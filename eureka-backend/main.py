from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import json
import openai
import torch
from sentence_transformers import SentenceTransformer, util
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI as PandasOpenAI
from dotenv import load_dotenv

# Load env vars
load_dotenv()

app = FastAPI()

# CORS (for React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
API_KEY = os.getenv("OPENAI_API_KEY") or "sk-..."  # Replace securely
openai.api_key = API_KEY

FOLDER_PATH = "C:\\Users\\azureadmin\\Desktop\\data"
USERS_FILE = "user.json"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Request Schema
class QueryInput(BaseModel):
    username: str
    password: str
    query: str

def authenticate_user(username, password):
    try:
        with open(USERS_FILE, "r") as f:
            users = json.load(f)
        user_info = users.get(username)
        if user_info and user_info["password"] == password:
            return user_info["datasets"]
    except Exception as e:
        print("❌ Error reading users.json:", e)
    return None

def generate_embeddings_for_user_datasets(datasets):
    vectors = []
    texts = []
    dfs = []

    for file in datasets:
        path = os.path.join(FOLDER_PATH, file)
        if not os.path.exists(path):
            continue

        try:
            if file.endswith(".csv"):
                df = pd.read_csv(path, encoding="ISO-8859-1")
            elif file.endswith(".xlsx"):
                df = pd.read_excel(path, engine="openpyxl")
            else:
                continue

            dfs.append(df)
            row_texts = df.astype(str).apply(lambda r: " | ".join(r.values), axis=1).tolist()
            embeddings = embedding_model.encode(row_texts, convert_to_tensor=True)
            texts.extend(row_texts)
            vectors.append(embeddings)

        except Exception as e:
            print(f"❌ Error reading {file}: {e}")

    if not dfs:
        return None, None, None

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_embeddings = torch.cat(vectors, dim=0)
    return combined_df, combined_embeddings, texts

@app.post("/query")
async def process_query(input: QueryInput):
    username = input.username
    password = input.password
    user_query = input.query.strip()

    datasets = authenticate_user(username, password)
    if not datasets:
        raise HTTPException(status_code=403, detail="❌ Invalid credentials or access denied")

    df, embeddings, text_rows = generate_embeddings_for_user_datasets(datasets)
    if df is None:
        raise HTTPException(status_code=500, detail="❌ Could not load datasets.")

    try:
        query_embedding = embedding_model.encode(user_query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(10, len(df)))

        top_indices = top_results.indices.cpu().numpy()
        filtered_df = df.iloc[top_indices]

        formatting_prompt = f"""
        You are an AI assistant analyzing a user's query.

        User Query: "{user_query}"

        - Determine whether the query requires dataset calculations.
        - If dataset calculations are needed, structure the query for PandasAI.
        """

        format_response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": formatting_prompt}],
            temperature=0.5,
            max_tokens=200
        )

        formatted_query = format_response["choices"][0]["message"]["content"].strip()

        if "no dataset calculations are required" in formatted_query.lower():
            return {"response": f"<ul><li>{formatted_query}</li></ul>"}

        llm = PandasOpenAI(api_token=API_KEY)
        smart_df = SmartDataframe(filtered_df, config={
            "llm": llm,
            "enable_cache": False,
            "enable_plotting": True,
            "enforce_privacy": True
        })

        pandasai_result = smart_df.chat(formatted_query)

        summary_prompt = f"""
        You are an AI assistant reviewing dataset results.

        User Query: {user_query}
        Extracted Data from PandasAI: {pandasai_result}

        Provide key insights, trends, and recommended actions based on the data in HTML bullet points format using <ul><li>...</li></ul>. Only return HTML.
        """

        summary_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.5,
            max_tokens=500
        )

        insights = summary_response["choices"][0]["message"]["content"]

        return {
            "query": user_query,
            "formatted_query": formatted_query,
            "pandasai_result": str(pandasai_result),
            "insights": insights
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error: {str(e)}")
