import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "eureka")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-west-2")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,  # OpenAI text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
    )

# Connect to index
index = pc.Index(PINECONE_INDEX_NAME)

# 🧠 Unified function
def upsert_to_pinecone(vectors: list, namespace: str = "eureka"):
    """
    Upserts a list of vectors to Pinecone.

    Each vector must be a dict:
    {
        "id": str,
        "values": list[float],  # 384-dim
        "metadata": dict
    }
    """
    items = [(v["id"], v["values"], v["metadata"]) for v in vectors]
    for i in range(0, len(items), 100):
        batch = items[i:i + 100]
        index.upsert(vectors=batch, namespace=namespace)

    print(f"✅ Inserted {len(items)} vectors into namespace '{namespace}'")
