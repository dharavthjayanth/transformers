import hashlib
import os
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = "text-embedding-3-small"


def generate_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed_chunks(chunks: List[Dict]) -> List[Dict]:
    embedded = []

    for chunk in chunks:
        try:
            response = client.embeddings.create(
                model=EMBED_MODEL,
                input=chunk["text"]
            )
            vector = response.data[0].embedding
            embedded.append({
                "id": generate_id(chunk["text"]),
                "values": vector,
                "metadata": chunk["metadata"]
            })
        except Exception as e:
            print(f"❌ Embedding failed for chunk: {chunk['metadata'].get('filename', 'unknown')} — {e}")

    return embedded
