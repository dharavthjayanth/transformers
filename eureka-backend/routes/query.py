import os
from typing import List
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from jose import jwt, JWTError
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "eureka")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

router = APIRouter(prefix="/query", tags=["Query"])

class QueryRequest(BaseModel):
    query: str

def get_user_scopes_from_request(request: Request) -> List[str]:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("scopes", [])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

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
def handle_query(request: Request, body: QueryRequest):
    scopes = get_user_scopes_from_request(request)
    query = body.query

    query_vector = get_query_embedding(query)

    try:
        pinecone_filter = {"domain": {"$in": scopes}}
        results = index.query(
            vector=query_vector,
            top_k=5,
            filter=pinecone_filter,
            include_metadata=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone query failed: {str(e)}")

    context_chunks = [match["metadata"]["text"] for match in results["matches"]]
    context = "\n\n".join(context_chunks)

    if not context:
        return { "answer": "No relevant data found within your access scope." }

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Answer based only on the given context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")

    return { "answer": answer }
