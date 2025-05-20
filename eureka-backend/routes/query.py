import os
import json
from typing import List
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "eureka-test")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER_ACCESS_PATH = os.path.join(BASE_DIR, "authentication", "users_access.json")

router = APIRouter(prefix="/query", tags=["Query"])

class QueryRequest(BaseModel):
    user_id: str
    query: str

def get_user_scopes(user_id: str) -> List[str]:
    try:
        with open(USER_ACCESS_PATH) as f:
            access_map = json.load(f)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to load access control file")

    if user_id not in access_map:
        raise HTTPException(status_code=403, detail="Unauthorized user")

    return access_map[user_id]["access"]

def get_query_embedding(query: str) -> List[float]:
    try:
        response = openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

@router.post("/")
def handle_query(body: QueryRequest):
    user_id = body.user_id
    query = body.query

    scopes = get_user_scopes(user_id)
    print(f"🔐 User: {user_id} | Namespaces: {scopes}")

    query_vector = get_query_embedding(query)

    pinecone_filter = {"namespace": {"$in": scopes}}

    try:
        results = index.query(
            vector=query_vector,
            top_k=5,
            filter=pinecone_filter,
            include_metadata=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone query failed: {str(e)}")

    matches = results.get("matches", [])
    print(f"📦 Pinecone matches found: {len(matches)}")

    context_chunks = []

    if matches:
        for i, match in enumerate(matches):
            text = match["metadata"].get("text")
            if text:
                context_chunks.append(text)
                print(f"✅ Match {i} added: {text[:80]}...")
            else:
                print(f"⚠️ Match {i} has no 'text' field:", match["metadata"])
    else:
        print("❌ Pinecone returned zero matches")

        context = "\n\n".join(context_chunks)

    if not context:
        return {"answer": "No relevant data found within your access scope."}

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Answer the question using only the context provided."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")

    return {"answer": answer}
