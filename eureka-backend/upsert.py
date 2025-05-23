import os
import uuid
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CSV_PATH = "data/output.csv"  # <- Update this path

# 1. Load and chunk CSV
def load_csv_as_documents(path):
    df = pd.read_csv(path)
    docs = []
    for _, row in df.iterrows():
        content = "\n".join(f"{col}: {row[col]}" for col in df.columns)
        docs.append(Document(page_content=content))
    return docs

def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

# 2. Setup and insert vectors into Pinecone
def upsert_chunks_to_pinecone(chunks):
    print("🧠 Connecting to Pinecone...")
    pc = PineconeClient(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print("🛠️ Creating Pinecone index...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
        )

    index = pc.Index(PINECONE_INDEX_NAME)

    print(f"✅ Index ready: {PINECONE_INDEX_NAME}")
    print("🔢 Generating embeddings...")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [{"text": chunk.page_content} for chunk in chunks]  # ✅ Required!

    embeddings = embedding_model.embed_documents(texts)

    # Build vector payloads
    vectors = []
    for i, emb in enumerate(embeddings):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": emb,
            "metadata": metadatas[i]
        })

    print("📤 Uploading vectors in batches...")
    BATCH_SIZE = 100
    for i in tqdm(range(0, len(vectors), BATCH_SIZE), desc="Uploading"):
        batch = vectors[i:i + BATCH_SIZE]
        index.upsert(vectors=batch)

    print("✅ Done! All vectors inserted into Pinecone.")

def main():
    print("📄 Loading and chunking CSV...")
    docs = load_csv_as_documents(CSV_PATH)
    chunks = chunk_documents(docs)
    print(f"📦 Total chunks: {len(chunks)}")

    upsert_chunks_to_pinecone(chunks)

if __name__ == "__main__":
    main()
