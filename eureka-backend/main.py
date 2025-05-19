import os
import json
import numpy as np
from glob import glob
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from routes import auth, query
from vector_db.insert import upsert_to_pinecone
from azure.ingestion_pipeline import ingest_selected_chunks


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(query.router)

CHUNKS_FILE = "data/chunks_raw.json"
EMBEDDED_FILE = "data/embedded_chunks.json"
EMBEDDING_DIR = "data/embeddings"
BATCH_SIZE = 100

os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

@app.get("/ingest")
def ingest_all_chunks():
    chunks = ingest_selected_chunks()
    with open(CHUNKS_FILE, "w") as f:
        json.dump(chunks, f, indent=2)
    return {
        "total_chunks": len(chunks),
        "sample": chunks[0] if chunks else "No chunks generated"
    }

@app.get("/embed")
def embed_all_chunks():
    if not os.path.exists(CHUNKS_FILE):
        raise HTTPException(status_code=400, detail="Run /ingest first")

    with open(CHUNKS_FILE, "r") as f:
        chunks = json.load(f)

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks found to embed.")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    total = len(chunks)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    success_count = 0

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [chunk["text"] for chunk in batch]
        embeddings = model.encode(texts, show_progress_bar=False)

        for j, chunk in enumerate(batch):
            namespace = chunk["metadata"].get("namespace", "default")
            source_file = chunk["metadata"].get("source_file", f"batch_{i // BATCH_SIZE + 1}")

            folder_path = os.path.join(EMBEDDING_DIR, namespace)
            os.makedirs(folder_path, exist_ok=True)

            output_file = os.path.join(folder_path, f"{source_file}.json")

            chunk["metadata"]["text"] = batch[j]["text"]
            embedding_record = {
                "id": f"chunk_{i + j}",
                "values": embeddings[j].tolist() if isinstance(embeddings[j], np.ndarray) else list(embeddings[j]),
                "metadata": chunk["metadata"]
            }

            # Append as JSONL
            with open(output_file, "a") as out_f:
                out_f.write(json.dumps(embedding_record) + "\n")

            success_count += 1

        print(f"✅ Embedded batch {i // BATCH_SIZE + 1} of {total_batches}")

    return {
        "status": "✅ Embedding complete",
        "total_embedded": success_count,
        "output_dir": EMBEDDING_DIR
    }

@app.get("/push")
def push_all_embeddings():
    if not os.path.exists(EMBEDDING_DIR):
        raise HTTPException(status_code=400, detail="No embeddings directory found. Run /embed first.")

    embedded_files = glob(os.path.join(EMBEDDING_DIR, "*", "*.json"))

    if not embedded_files:
        raise HTTPException(status_code=400, detail="No embedded files found in structured folders.")

    total_vectors = 0

    for file_path in embedded_files:
        namespace = os.path.basename(os.path.dirname(file_path))
        with open(file_path, "r") as f:
            lines = f.readlines()

        if not lines:
            continue

        records = [json.loads(line) for line in lines]
        total = len(records)
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        for i in range(0, total, BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            upsert_to_pinecone(batch, namespace=namespace)
            print(f"✅ Inserted batch {i // BATCH_SIZE + 1} of {total_batches} into namespace '{namespace}'")

        total_vectors += total

    return {
        "status": "✅ All embeddings pushed to Pinecone",
        "files_processed": len(embedded_files),
        "total_vectors": total_vectors
    }

