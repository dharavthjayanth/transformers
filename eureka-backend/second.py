import os
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI as PandasAIOpenAI

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CSV_PATH = "output.csv"
QUERY = "which plant spend more on Packaging & Supplies for year 2024?"

def fetch_relevant_chunks(query):
    pc = PineconeClient(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(query)
    print(f"📦 Retrieved {len(docs)} documents from Pinecone")
    return docs

def analyze_with_pandasai(docs, query):
    print("\n📊 Converting retrieved chunks into DataFrame for analysis...")
    raw_data = [doc.page_content for doc in docs]

    records = []
    for text in raw_data:
        row = {}
        for line in text.split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                row[k.strip()] = v.strip()
        records.append(row)

    df = pd.DataFrame(records)

    # Optional cleaning
    if "PO_Net_Amount" in df.columns:
        df["PO_Net_Amount"] = df["PO_Net_Amount"].replace('[\$,]', '', regex=True).astype(float)

    if "Purchase_Order_Date" in df.columns:
        df["Year"] = pd.to_datetime(df["Purchase_Order_Date"], errors="coerce", dayfirst=True).dt.year

    print(df.head(3))

    pandasai_llm = PandasAIOpenAI(api_token=OPENAI_API_KEY, model="gpt-3.5-turbo")
    smart_df = SmartDataframe(df, config={"llm": pandasai_llm})

    instruction = f"""
You are working with a DataFrame containing purchase order data.

Columns include:
- Company Code
- Purchase Order
- Purchase Order Date
- Material
- Material Description
- Material_Group
- Material_Group description
- Plant
- Plant_Description
- Supplier
- Supplier_description
- Supplier_Country description
- PO_Net_Amount (monetary values)
- Quantity and UOM
- Purchase_Order_Date (format: DD.MM.YYYY)

You may need to look for rows where 'Material_Group description' contains the word 'Additive' (case-insensitive match).
You must:
1. Extract the year from Purchase_Order_Date if filtering by time.
2. Filter by conditions like 'PET' in Material_Group.
3. Convert PO_Net_Amount to float (remove symbols).
4. Sum values when asked about total spend.

Now answer this:
{query}
"""

    return smart_df.chat(instruction)

def main():
    print(f"✅ Using Pinecone Index: {PINECONE_INDEX_NAME}")
    docs = fetch_relevant_chunks(QUERY)
    if not docs:
        print("❌ No documents retrieved.")
        return
    result = analyze_with_pandasai(docs, QUERY)
    print("\n💬 Answer from PandasAI:")
    print(result)

if __name__ == "__main__":
    main()
