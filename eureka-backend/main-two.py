import os
import uuid
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_community.llms import OpenAI as OpenAI_LLM
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangchainPinecone

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west-2")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "eureka-data")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CSV_PATH = "data/output.csv"
QUERY = "What is total spend on Additive in year 2024?"

def load_csv_as_documents(csv_path):
    df = pd.read_csv(csv_path)
    documents = []
    for _, row in df.iterrows():
        content = "\n".join(f"{col}: {row[col]}" for col in df.columns)
        documents.append(Document(page_content=content))
    return documents

def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def setup_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

def setup_pinecone_index(embedding_model, chunks):
    print("🧠 Starting Pinecone integration...")
    pc = PineconeClient(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"🛠️ Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
        )

    index = pc.Index(PINECONE_INDEX_NAME)
    print(f"✅ Using Pinecone Index: {PINECONE_INDEX_NAME}")
    print(f"📦 Preparing to insert {len(chunks)} chunks...")

    # Prepare data
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    print("🔢 Creating OpenAI embeddings...")
    embeddings = embedding_model.embed_documents(texts)

    vectors = []
    for i, vector in enumerate(embeddings):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": vector,
            "metadata": metadatas[i] if i < len(metadatas) else {}
        })

    print(f"📤 Upserting vectors in batches...")
    BATCH_SIZE = 100
    for i in tqdm(range(0, len(vectors), BATCH_SIZE), desc="📤 Uploading to Pinecone"):
        batch = vectors[i:i + BATCH_SIZE]
        index.upsert(vectors=batch)

    print("✅ All vectors inserted successfully.")
    return index  # optionally return LangChain wrapper later

def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = OpenAI_LLM(openai_api_key=OPENAI_API_KEY)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def describe_pinecone_index(index_name):
    print("\n📄 Index description:")

    pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    desc = pc.describe_index(index_name)

    if desc is None:
        print("❌ Failed to get index description — got None.")
        return

    # Convert to dict if supported, else print directly
    if hasattr(desc, "model_dump"):
        desc_dict = desc.model_dump()
    elif hasattr(desc, "__dict__"):
        desc_dict = desc.__dict__
    else:
        print(desc)
        return

    for k, v in desc_dict.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    print(f"✅ Loaded Pinecone Index Name: {PINECONE_INDEX_NAME}")

    documents = load_csv_as_documents(CSV_PATH)
    chunks = chunk_documents(documents)
    embedding_model = setup_embeddings()
    vectorstore = setup_pinecone_index(embedding_model, chunks)
    
    describe_pinecone_index(PINECONE_INDEX_NAME)
    
    qa_chain = create_qa_chain(vectorstore)
    answer = qa_chain.run(QUERY)

    print("\n💬 Answer:")
    print(answer)
