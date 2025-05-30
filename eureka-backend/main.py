from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import pandas as pd
from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")

app = FastAPI()

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
weaviate_client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=AuthApiKey(WEAVIATE_API_KEY),
    additional_headers={"X-OpenAI-Api-Key": OPENAI_API_KEY}
)

# Ensure class exists
excel_chunk_class = {
    "class": "ExcelChunk",
    "description": "Chunks of text extracted from Excel files",
    "properties": [
        {
            "name": "text",
            "dataType": ["text"]
        }
    ]
}

if not weaviate_client.schema.contains(excel_chunk_class):
    weaviate_client.schema.create_class(excel_chunk_class)


# Helper: Chunk Excel file into text segments
def chunk_excel(file_path, chunk_size=500):
    df = pd.read_excel(file_path, engine="openpyxl")
    text = df.astype(str).to_string(index=False)
    tokens = text.split()
    return [" ".join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]


# Endpoint: Upload Excel and embed
@app.post("/embed/")
async def embed_excel(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    path = f"tmp_{file.filename}"
    try:
        with open(path, "wb") as f:
            f.write(await file.read())

        # Chunk and embed
        chunks = chunk_excel(path)
        for chunk in chunks:
            try:
                embedding = openai_client.embeddings.create(
                    input=chunk,
                    model="text-embedding-3-small"
                ).data[0].embedding

                weaviate_client.data_object.create(
                    data_object={"text": chunk},
                    class_name="ExcelChunk",
                    vector=embedding
                )
            except Exception as weaviate_error:
                raise HTTPException(status_code=500, detail=f"Weaviate error: {str(weaviate_error)}")

        return {"status": f"✅ Successfully embedded {len(chunks)} chunks"}

    finally:
        if os.path.exists(path):
            os.remove(path)


# Endpoint: Query embedded data
@app.get("/query/")
def query_excel(q: str):
    response = weaviate_client.query.get("ExcelChunk", ["text"]).with_near_text({
        "concepts": [q]
    }).with_limit(3).do()

    print("📦 Raw Weaviate Response:", response)

    try:
        chunks = response["data"]["Get"]["ExcelChunk"]
    except KeyError:
        raise HTTPException(status_code=404, detail="❌ No matching data found in Weaviate.")

    context = "\n---\n".join([d["text"] for d in chunks])
    prompt = f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"

    answer = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return {"answer": answer.choices[0].message.content}
