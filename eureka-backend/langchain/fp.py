# ✅ Environment Setup
import os
import pandas as pd
from dotenv import load_dotenv
import spacy
from spacy.matcher import PhraseMatcher
from rapidfuzz import process, fuzz
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent, AgentType
from langchain.memory import ConversationBufferMemory

# 🔐 Load API Keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY is not loaded from .env")

print("✅ API key successfully loaded")

csv_path = "/Users/keshavsaraogi/Desktop/indorama/eureka-data/clean-csv/cleaned_finance_packaging.csv"
df = pd.read_csv(csv_path)

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5
)

finance_csv_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
finance_sql_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def normalize_query(query: str, df: pd.DataFrame) -> str:
    nlp = spacy.blank("en")
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    # Get company names
    company_names = df["Company Name"].dropna().unique().tolist()
    patterns = [nlp.make_doc(name) for name in company_names]
    matcher.add("COMPANY", patterns)

    # Run spaCy matcher
    doc = nlp(query)
    matches = matcher(doc)

    if matches:
        match_id, start, end = sorted(matches, key=lambda x: x[2] - x[1], reverse=True)[0]
        span_text = doc[start:end].text
        matched_phrase = next((name for name in company_names if name.lower() == span_text.lower()), None)

        if not matched_phrase:
            matched_phrase = next((name for name in company_names if span_text.lower() in name.lower()), None)

        if matched_phrase:
            print(f"✅ [spaCy] Replacing '{span_text}' → '{matched_phrase}'")
            return query.replace(span_text, matched_phrase)

    # Fallback: fuzzy n-gram matching
    stopwords = {"for", "of", "in", "the", "a", "an", "on", "to", "with", "by", "and"}
    tokens = query.split()
    max_ngram = 4
    best_match = None
    best_score = 0
    best_ngram = ""

    for n in range(1, max_ngram + 1):
        for i in range(len(tokens) - n + 1):
            ngram_tokens = tokens[i:i+n]
            ngram = " ".join(ngram_tokens)

            # Skip short, common, or meaningless n-grams
            if all(tok.lower() in stopwords or len(tok) <= 2 for tok in ngram_tokens):
                continue

            result = process.extractOne(ngram, company_names, scorer=fuzz.WRatio, score_cutoff=80)
            if result:
                match, score, _ = result
                if score > best_score:
                    best_match = match
                    best_score = score
                    best_ngram = ngram

    if best_match and best_ngram:
        print(f"✅ [Fuzzy N-gram] Replacing '{best_ngram}' → '{best_match}'")
        return query.replace(best_ngram, best_match)

    print("⚠️ No match found via spaCy or fuzzy matching.")
    return query


query = "What is the total spend for Indorama?"
normalized_query = normalize_query(query, df)
print(normalized_query)

print(df["Company Name"].unique())

csv_agent = create_csv_agent(
    llm,
    csv_path,
    verbose=True,
    allow_dangerous_code=True,
    handle_parsing_errors=True,
    max_iterations=20,
    max_execution_time=60,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    memory=finance_csv_memory
)

engine = create_engine("sqlite:///finance_packaging.db")
df.to_sql("finance_packaging_data", engine, index=False, if_exists="replace")
db = SQLDatabase(engine)

sql_agent = create_sql_agent(
    llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=20,
    max_execution_time=60,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    memory=finance_sql_memory
)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Optional: cache or re-use this prompt if desired
routing_prompt = PromptTemplate.from_template(
    """
You are a routing assistant that decides how to process data queries.

Given the following user query:
"{query}"

Decide whether it should be handled using SQL (for operations like aggregation, filtering, grouping, or numeric analysis),
or using CSV (for visualization, listing, or non-aggregated exploration).

Respond with only one word: "sql" or "csv".
"""
)

def route_query(query: str) -> str:
    chain = routing_prompt | llm | StrOutputParser()
    result = chain.invoke({"query": query}).strip().lower()

    if result not in {"sql", "csv"}:
        print(f"⚠️ Unexpected route output: {result} — defaulting to 'csv'")
        return "csv"
    
    return result


def master_agent(query: str):
    query = normalize_query(query, df)
    source = route_query(query)
    if source == "sql":
        return sql_agent.run(query)
    else:
        return csv_agent.run(query)
    

# 🧠 Recall Last Interaction
def recall_last_interaction(memory):
    messages = memory.chat_memory.messages
    if len(messages) >= 2:
        user_msg = messages[-2].content
        ai_msg = messages[-1].content
        return f"🧠 Last Question: {user_msg}\n💬 Last Answer: {ai_msg}"
    return "🧠 No previous memory found."


def show_full_chat(memory):
    print("🧠 Chat History:")
    for msg in memory.chat_memory.messages:
        role = msg.type.capitalize()
        print(f"{role}: {msg.content}")
        
master_agent("What is the total Adv To Sup Local sent by Bevpak?")