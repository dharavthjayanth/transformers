# import os
# import json
# import time
# import numpy as np
# from glob import glob
# from openai import OpenAI
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException, Form
# from fastapi.middleware.cors import CORSMiddleware
# from sentence_transformers import SentenceTransformer

# from routes import auth, query
# from vector_db.insert import upsert_to_pinecone
# from azure.ingestion_pipeline import ingest_selected_chunks


# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# load_dotenv()
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# app.include_router(auth.router)
# app.include_router(query.router)

# CHUNKS_FILE = "data/chunks_raw.json"
# EMBEDDED_FILE = "data/embedded_chunks.json"
# EMBEDDING_DIR = "data/embeddings"
# BATCH_SIZE = 100

# os.makedirs(EMBEDDING_DIR, exist_ok=True)
# os.makedirs("data", exist_ok=True)

# @app.get("/ingest")
# def ingest_all_chunks():
#     chunks = ingest_selected_chunks()
#     with open(CHUNKS_FILE, "w") as f:
#         json.dump(chunks, f, indent=2)
#     return {
#         "total_chunks": len(chunks),
#         "sample": chunks[0] if chunks else "No chunks generated"
#     }

# @app.get("/embed")
# def embed_all_chunks():
#     if not os.path.exists(CHUNKS_FILE):
#         raise HTTPException(status_code=400, detail="Run /ingest first")

#     with open(CHUNKS_FILE, "r") as f:
#         chunks = json.load(f)

#     if not chunks:
#         raise HTTPException(status_code=400, detail="No chunks found to embed.")

#     total = len(chunks)
#     total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
#     success_count = 0

#     for i in range(0, total, BATCH_SIZE):
#         batch = chunks[i:i + BATCH_SIZE]
#         texts = [chunk["text"] for chunk in batch]

#         try:
#             response = client.embeddings.create(input=texts, model="text-embedding-3-small")
#             embeddings = [res.embedding for res in response.data]
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"OpenAI embedding failed at batch {i // BATCH_SIZE + 1}: {e}")

#         for j, chunk in enumerate(batch):
#             namespace = chunk["metadata"].get("namespace", "default")
#             source_file = chunk["metadata"].get("source_file", f"batch_{i // BATCH_SIZE + 1}")
#             folder_path = os.path.join(EMBEDDING_DIR, namespace)
#             os.makedirs(folder_path, exist_ok=True)
#             output_file = os.path.join(folder_path, f"{source_file}.json")

#             embedding_record = {
#                 "id": f"chunk_{i + j}",
#                 "values": embeddings[j],
#                 "metadata": chunk["metadata"]
#             }

#             with open(output_file, "a") as out_f:
#                 out_f.write(json.dumps(embedding_record) + "\n")

#             success_count += 1

#         print(f"✅ Embedded batch {i // BATCH_SIZE + 1} of {total_batches}")

#     return {
#         "status": "✅ Embedding complete",
#         "total_embedded": success_count,
#         "output_dir": EMBEDDING_DIR
#     }

# CHECKPOINT_DIR = "checkpoints"
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# @app.get("/push")
# def push_all_embeddings():
#     if not os.path.exists(EMBEDDING_DIR):
#         raise HTTPException(status_code=400, detail="No embeddings directory found. Run /embed first.")

#     embedded_files = glob(os.path.join(EMBEDDING_DIR, "*", "*.json"))

#     if not embedded_files:
#         raise HTTPException(status_code=400, detail="No embedded files found in structured folders.")

#     total_vectors = 0

#     for file_path in embedded_files:
#         namespace = os.path.basename(os.path.dirname(file_path))
#         checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{namespace}.txt")

#         with open(file_path, "r") as f:
#             lines = f.readlines()

#         if not lines:
#             continue

#         records = [json.loads(line) for line in lines]
#         total = len(records)
#         total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

#         # Load last checkpoint
#         resume_from = 0
#         if os.path.exists(checkpoint_file):
#             with open(checkpoint_file, "r") as cf:
#                 resume_from = int(cf.read().strip() or "0")

#         for i in range(resume_from, total, BATCH_SIZE):
#             batch = records[i:i + BATCH_SIZE]
#             try:
#                 upsert_to_pinecone(batch, namespace=namespace)
#                 print(f"✅ Inserted batch {i // BATCH_SIZE + 1} of {total_batches} into namespace '{namespace}'")
#                 # Save checkpoint after successful batch
#                 with open(checkpoint_file, "w") as cf:
#                     cf.write(str(i + BATCH_SIZE))
#                 time.sleep(1)
#             except Exception as e:
#                 print(f"❌ Failed at batch {i // BATCH_SIZE + 1}: {e}")
#                 raise HTTPException(status_code=500, detail=f"Failed at batch {i // BATCH_SIZE + 1}: {e}")

#         total_vectors += total

#     return {
#         "status": "✅ All embeddings pushed to Pinecone",
#         "files_processed": len(embedded_files),
#         "total_vectors": total_vectors
#     }

# @app.post("/query")
# def query_chunks(prompt: str = Form(...), namespace: str = Form("default")):
#     try:
#         # Step 1: Embed the query prompt
#         response = client.embeddings.create(input=prompt, model="text-embedding-3-small")
#         query_vector = response.data[0].embedding

#         # Step 2: Query Pinecone
        
#         result = index.query(vector=query_vector, top_k=5, include_metadata=True, namespace=namespace)

#         if not result.matches:
#             return {"answer": "No relevant results found."}

#         # Step 3: Extract the top matches
#         matched_rows = [match.metadata for match in result.matches]
#         return {"matches": matched_rows}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


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
