import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pc_index_name = os.getenv("PINECONE_INDEX_NAME")

if pc_index_name not in pc.list_indexes().names():
    pc.create_index(
        name=pc_index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(pc_index_name)

def upsert_to_pinecone(vectors: list, namespace: str = "default"):
    """
    Takes a list of dicts like:
    {
        "id": ...,  # SHA256 or UUID
        "values": [...],  # 1536-dim OpenAI embedding
        "metadata": {...}  # domain, root, etc.
    }
    """
    items = [(v["id"], v["values"], v["metadata"]) for v in vectors]

    for i in range(0, len(items), 100):
        batch = items[i:i + 100]
        index.upsert(vectors=batch, namespace=namespace)

    print(f"✅ Inserted {len(items)} vectors into namespace '{namespace}'")

def upsert_to_pinecone(vectors, namespace="eureka"):
    index.upsert(vectors=vectors, namespace=namespace)