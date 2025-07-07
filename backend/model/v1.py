## IMPORT STATEMENTS

import os
import json
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.agents import create_csv_agent
from langchain_core.exceptions import OutputParserException
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

from dotenv import load_dotenv
load_dotenv()

## WARNINGS
import warnings
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

## ENV VARIABLES
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# FastAPI app
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QueryRequest(BaseModel):
    query: str

## LLM CONFIGURATION
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(temperature=0.5, model="gpt-4o")
persistent_instructions = []

## READ THE FILES   

finance_file = os.getenv("F_URL")
inventory_file = os.getenv("I_URL")
spend_file = os.getenv("SP_URL")
sales_file = os.getenv("SA_URL")

## CREATING DATAFRAMES

finance_df = pd.read_csv(finance_file, low_memory=False, dtype={9: str})
inventory_df = pd.read_csv(inventory_file, low_memory=False, dtype={9: str})
spend_df = pd.read_csv(spend_file, low_memory=False, dtype={9: str})
sales_df = pd.read_csv(sales_file, low_memory=False, dtype={9: str})

## DATA FORMAT CONVERSION

spend_df['Purchase_Order_Date'] = pd.to_datetime(spend_df['Purchase_Order_Date'], format="%d.%m.%Y")
spend_df['Purchase_Order_Date'] = spend_df['Purchase_Order_Date'].dt.strftime("%Y-%m-%d")

## DATABASE CONFIGURATION

def create_in_memory_db(df, table_name):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    df.to_sql(table_name, conn, index=False, if_exists="replace")
    return conn

finance_db = create_in_memory_db(finance_df, "finance")
inventory_db = create_in_memory_db(inventory_df, "inventory")
spend_db = create_in_memory_db(spend_df, "spend")
sales_db = create_in_memory_db(sales_df, "sales")

finance_engine = create_engine("sqlite://", creator=lambda: finance_db)
inventory_engine = create_engine("sqlite://", creator=lambda: inventory_db)
spend_engine = create_engine("sqlite://", creator=lambda: spend_db)
sales_engine = create_engine("sqlite://", creator=lambda: sales_db)

finance_sql_db = SQLDatabase(engine=finance_engine, sample_rows_in_table_info=5)
inventory_sql_db = SQLDatabase(engine=inventory_engine, sample_rows_in_table_info=5)
spend_sql_db = SQLDatabase(engine=spend_engine, sample_rows_in_table_info=5)
sales_sql_db = SQLDatabase(engine=sales_engine, sample_rows_in_table_info=5)

finance_toolkit = SQLDatabaseToolkit(db=finance_sql_db, llm=llm)
inventory_toolkit = SQLDatabaseToolkit(db=inventory_sql_db, llm=llm)
spend_toolkit = SQLDatabaseToolkit(db=spend_sql_db, llm=llm)
sales_toolkit = SQLDatabaseToolkit(db=sales_sql_db, llm=llm)


## PROMPT TEMPLATES

intent_prompt = PromptTemplate.from_template("""
You are a query router for a data assistant.

Classify this user query into one of the following categories:

- "memory": if the question refers to previous questions, previous answers, or instructions like "compare with what I asked earlier", "what did I ask before", "remember", etc.
- "dataset": if the question is related to data exploration, analysis, charting, SQL, or CSV.
- "other": if the question does not match any of the above.

Query:
"{query}"

Only respond with one word: memory, dataset, or other.
""")


insight_prompt = PromptTemplate.from_template("""
You are a data analyst. A user has asked the following query:

{query}

Here is the result table:
{table}

Write 2–3 insightful bullet points that:
- Compare time periods, categories, or trends
- Highlight anomalies, spikes, or outliers
- Avoid simply rephrasing or summarizing the values
- Use quantitative language when possible
""")

sql_agent_prompt_prefix = """
You are a SQL expert agent following ReAct reasoning.

- You must ALWAYS output Thought -> Action -> Observation -> Final Answer.
- DO NOT output meta commentary.
- After seeing Observation results (SQL query output), ALWAYS extract concrete values.
- ALWAYS summarize the result table to answer the user’s original question directly.
- NEVER say "the query successfully identifies..." — always give actual values.
- DO NOT wrap SQL code in markdown formatting or backticks.
- ONLY output valid SQL without formatting.
- If column names contain spaces, enclose them in double quotes.
- The SQL dialect is SQLite.
- ALWAYS use the available tools (sql_db_query) to execute your queries.
- NEVER just write SQL queries.
- ALWAYS call the action sql_db_query with the query as input.
- You are allowed to chain multiple queries to answer the question.
- If you encounter repeated errors or cannot execute the SQL query, still follow the ReAct format.
- When unable to answer, output:
Thought: I am unable to answer.
Final Answer: Unable to retrieve the data due to internal error.
- Do not write freeform explanations.
- Never write paragraphs describing failure.
- NEVER output markdown formatting.
- NEVER output queries inside triple backticks or code fences.
- ONLY output raw SQL text.
- SQLite does not support '%q' for quarters.
- To compute quarter, use strftime('%m', "Date") and CASE WHEN statements.
- NEVER use '%q' inside strftime() queries.

### ✅ PATCHED RULES:
- Use TRIM(LOWER(...)) for **ALL string comparisons** to avoid case/whitespace mismatches.
- Use CAST(... AS FLOAT) for **ALL numeric aggregations** (e.g., SUM, AVG) to prevent NULL-type results.
- Use `"Column" IS NOT NULL` for all **aggregated or filtered numeric columns**.
- If a SUM or COUNT returns NULL, assume fallback value 0 and proceed with reasoning.
- If query output is empty or all values are NULL, return 0 instead of NULL.
"""


dataset_routing_prompt = PromptTemplate.from_template("""
You are a dataset routing assistant.

Here are the columns for each dataset:

- finance: {finance_cols}
- inventory: {inventory_cols}
- spend: {spend_cols}
- sales: {sales_cols}

Given the following user query:
"{query}"

Decide which dataset this query should be routed to.

Respond ONLY with one word: finance, inventory, spend, or sales.
""")

agent_type_prompt = PromptTemplate.from_template("""
You are a routing assistant that decides how to process data queries.
Given the following user query:
"{query}"
Decide whether it should be handled using SQL (for aggregation, filtering, grouping, numeric analysis),
or using CSV (for visualization, listing, non-aggregated exploration).
Read dates as DD-MM-YYYY.
Respond only with: sql or csv.
""")

PACKAGING_KNOWLEDGE = """
When asked for packaging spend, you should filter "Material Group Description" column using these values:
'PET', 'BOTTLES', 'LABELS', 'CAPS', 'SEALS'.
If no values match, return 0.
Always handle column names with spaces by using double quotes.
The column Purchase_Order_Date is stored in format YYYY-MM-DD.
"""

visualization_prompt = PromptTemplate.from_template("""
You are a data visualization expert.

Based on the user's query and the given data preview, suggest the most suitable chart type.

Supported chart types:
- bar
- line
- scatter
- pie
- histogram
- none (if no chart is appropriate)

User query:
{query}

Data preview:
{data}

Respond with one word only: bar, line, scatter, pie, histogram, or none.
""")

## CUSTOM CHAINS

visualization_chain = visualization_prompt | llm | StrOutputParser()
insight_chain = insight_prompt | llm | StrOutputParser()

## AGENTS

finance_csv_agent = create_csv_agent(llm, finance_file, verbose=True, allow_dangerous_code=True)
inventory_csv_agent = create_csv_agent(llm, finance_file, verbose=True, allow_dangerous_code=True)
spend_csv_agent = create_csv_agent(llm, finance_file, verbose=True, allow_dangerous_code=True)
sales_csv_agent = create_csv_agent(llm, finance_file, verbose=True, allow_dangerous_code=True)

finance_sql_agent = create_sql_agent(
    llm=llm, toolkit=finance_toolkit, verbose=True, allow_dangerous_code=True,
    max_iterations=40, max_execution_time=120, memory=memory,
    handle_parsing_errors=True, early_stopping_method="generate",
    prefix=sql_agent_prompt_prefix
)

inventory_sql_agent = create_sql_agent(
    llm=llm, toolkit=inventory_toolkit, verbose=True, allow_dangerous_code=True,
    max_iterations=40, max_execution_time=120, memory=memory,
    handle_parsing_errors=True, early_stopping_method="generate",
    prefix=sql_agent_prompt_prefix
)

spend_sql_agent = create_sql_agent(
    llm=llm, toolkit=spend_toolkit, verbose=True, allow_dangerous_code=True,
    max_iterations=40, max_execution_time=120, handle_parsing_errors=True, 
    early_stopping_method="generate", prefix=sql_agent_prompt_prefix
)

sales_sql_agent = create_sql_agent(
    llm=llm, toolkit=sales_toolkit, verbose=True, allow_dangerous_code=True,
    max_iterations=40, max_execution_time=120,
    handle_parsing_errors=True, early_stopping_method="generate",
    prefix=sql_agent_prompt_prefix
)


## AGENT FUNCTIONS

def classify_intent_llm(query: str) -> str:
    return (intent_prompt | llm | StrOutputParser()).invoke({"query": query}).strip().lower()


def generate_insights(query: str, df: pd.DataFrame) -> str:
    if df.empty:
        return "No insights available — the result table is empty."
    try:
        table_snippet = df.head(5).to_markdown(index=False)
        return insight_chain.invoke({"query": query, "table": table_snippet}).strip()
    except Exception as e:
        print(f"⚠️ Insight generation failed: {e}")
        return "Insight generation failed due to an internal error."    

def get_column_names(filepath):
    df = pd.read_csv(filepath, nrows=1)
    return list(df.columns)

def classify_visualization_type(user_query: str, df_result: pd.DataFrame) -> str:
    try:
        # Prepare data preview
        data_sample = df_result.head(5).to_markdown(index=False)

        # Call LLM chain
        result = visualization_chain.invoke({
            "query": user_query,
            "data": data_sample
        })

        cleaned_result = result.strip().lower()
        valid_types = {"bar", "line", "scatter", "pie", "histogram", "none"}

        if cleaned_result in valid_types:
            return cleaned_result
        else:
            print(f"⚠️ Unrecognized chart type from LLM: {cleaned_result}")
            return "none"

    except Exception as e:
        print(f"❌ Visualization type classification failed: {e}")
        return "none"

finance_columns = get_column_names(finance_file)
inventory_columns = get_column_names(inventory_file)
spend_columns = get_column_names(spend_file)
sales_columns = get_column_names(sales_file)

def route_dataset(user_query: str) -> str:
    chain = dataset_routing_prompt | llm | StrOutputParser()

    prompt_input = {
        "query": user_query,
        "finance_cols": ", ".join(finance_columns),
        "inventory_cols": ", ".join(inventory_columns),
        "spend_cols": ", ".join(spend_columns),
        "sales_cols": ", ".join(sales_columns),
    }

    result = chain.invoke(prompt_input).strip().lower()

    if result not in {"finance", "inventory", "spend", "sales"}:
        print(f"⚠️ Unexpected dataset output: {result} — defaulting to 'finance'")
        return "finance"
    
    return result

def route_agent_type(query: str) -> str:
    chain = agent_type_prompt | llm | StrOutputParser()
    result = chain.invoke({"query": query}).strip().lower()
    if result not in {"sql", "csv"}:
        return "csv"
    return result

def convert_to_langchain_messages(messages):
    lc_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
    return lc_messages

def update_instructions_if_needed(query: str):
    """Detect and store persistent instructions from user queries."""
    lowered = query.lower()
    if lowered.startswith("from now on") or "always" in lowered:
        persistent_instructions.append(query.strip())

def apply_instruction_prefix(query: str) -> str:
    if persistent_instructions:
        prefix = "\n".join(persistent_instructions).strip()
        return f"{prefix}\n\n{query.strip()}"
    return query.strip()

def apply_instruction_prefix(query: str) -> str:
    if persistent_instructions:
        prefix = "\n".join(persistent_instructions).strip()
        return f"{prefix}\n\n{query.strip()}"
    return query.strip()

def master_agent(user_query):
    try:
        print("🧠 Chat memory (last turn):")
        print(memory.chat_memory.messages[-1] if memory.chat_memory.messages else "No memory yet.")

        # 🔍 Intent
        intent_output = classify_intent_llm(user_query)
        print("📍 Detected intent:", intent_output)

        if intent_output == "memory":
            try:
                memory_text = "\n\n".join([
                    f"🧍 **Human:**\n{m.content}" if m.type == "human" else f"🤖 **AI Assistant:**\n{m.content}"
                    for m in memory.chat_memory.messages[-10:]
                ])
                return {
                    "response": f"🧠 Here's your recent conversation:\n\n{memory_text}",
                    "dataframe": None
                }
            except Exception as e:
                return {
                    "response": f"⚠️ Failed to fetch memory: {e}",
                    "dataframe": None
                }

        # 🔁 Apply instructions
        user_query = apply_instruction_prefix(user_query)

        # 🔀 Routing
        dataset = route_dataset(user_query)
        agent_type = route_agent_type(user_query)
        print(f"📦 Dataset: {dataset}, Agent type: {agent_type}")

        agents = {
            "finance": {"sql": finance_sql_agent, "csv": finance_csv_agent},
            "inventory": {"sql": inventory_sql_agent, "csv": inventory_csv_agent},
            "spend": {"sql": spend_sql_agent, "csv": spend_csv_agent},
            "sales": {"sql": sales_sql_agent, "csv": sales_csv_agent},
        }

        df_map = {
            "finance": finance_df,
            "inventory": inventory_df,
            "spend": spend_df,
            "sales": sales_df,
        }

        db_map = {
            "finance": finance_db,
            "inventory": inventory_db,
            "spend": spend_db,
            "sales": sales_db,
        }

        if dataset == "spend" and agent_type == "sql":
            user_query += f"\n\n{PACKAGING_KNOWLEDGE}"

        agent = agents[dataset][agent_type]
        df_result = df_map[dataset]

        # 🧠 Run
        try:
            if isinstance(agent, AgentExecutor):
                result = agent.invoke(user_query, handle_parsing_errors=True)
            else:
                result = agent.invoke(user_query)
        except Exception as e:
            return {
                "response": "❌ Agent failed after retry.",
                "dataframe": None,
                "error": str(e)
            }

        # 🧾 SQL result capture
        if agent_type == "sql":
            try:
                import re
                sql_output = result.get("output") if isinstance(result, dict) else result
                conn = db_map[dataset]
                sql_pattern = r"Action: sql_db_query\s+Action Input:\s*(.*?)\n(?:Observation|Thought|Final Answer):"
                match = re.search(sql_pattern, sql_output, re.DOTALL)
                sql_query = match.group(1).strip() if match else None
                if sql_query and sql_query.lower().startswith("select"):
                    df_result = pd.read_sql_query(sql_query, conn)
            except Exception as e:
                print(f"⚠️ SQL extraction error: {e}")

        # 💡 Insights
        agent_response = result.get("output") if isinstance(result, dict) else result
        insight = generate_insights(user_query, df_result)

        return {
            "response": f"""🧾 **Answer:**\n{agent_response}\n\n🔍 **Insights:**\n{insight}""",
            "dataframe": df_result.to_json(orient="split") if not df_result.empty else None
        }

    except Exception as e:
        print(f"❌ Unhandled error: {e}")
        return {
            "response": "An unexpected error occurred.",
            "dataframe": None,
            "error": str(e)
        }

# # List of questions to query
# questions = [
#     "What is the sum of quantity mt for company TH14 in 2024?",
#     "Show total sales invoice net value for 2023 and 2024.",
#     "Which customers contributed most to sales in 2024?",
#     "Which customer name had the highest total ending balance in global currency for company code TH14?",
#     "What is the sum of ending balance of cusomter PEPSI COLA PRODUCTS PHILIPPINES INC in 2024?",if 
#     "Which company showed the most fluctuation for each quarter in ending balance in 2024?",
#     "Give me top 5 customers with the highest ending balance in 2024?",
#     "Compare Sales Performance for different quarters in 2024 for each company",
# ]

# FastAPI endpoint
@app.post("/query")
def query_endpoint(data: QueryRequest):
    result = master_agent(data.query)
    if isinstance(result, dict):
        return {"response": result.get("output") or result.get("error") or "Unknown error"}
    return {"response": result}

@app.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    lc_chat_history = convert_to_langchain_messages(messages)
    memory.chat_memory.messages = lc_chat_history
    
    print("🧠 Chat memory (last turn):")
    for m in memory.chat_memory.messages[-2:]:
        print(m)

    latest_user_msg = next(
        (msg["content"] for msg in reversed(messages) if msg["role"] == "user"), None
    )
    
    if not latest_user_msg:
        return {"message": {"role": "assistant", "content": "No user message found."}}

    result = master_agent(latest_user_msg)

    output = result.get("response", "No response.")
    image = result.get("image", None)
    dataframe_json = result.get("dataframe", None)

    if isinstance(result, dict):
        output = result.get("response") or result.get("error") or "Unknown error"
        image = result.get("image", None)
    else:
        output = str(result)
        image = None

    return {
    "message": {
        "role": "assistant",
        "content": output
    },
    "image": image,
    "dataframe": dataframe_json
}

@app.get("/chat/history")
def chat_history():
    messages = memory.chat_memory.messages
    formatted = "\n\n".join([
        f"🧍 **Human:**\n{m.content}" if m.type == "human" else f"🤖 **AI Assistant:**\n{m.content}"
        for m in messages
    ])
    return {"chat_history": formatted}

@app.get("/instructions")
def get_instructions():
    return {"instructions": persistent_instructions}

@app.post("/instructions/clear")
def clear_instructions():
    persistent_instructions.clear()
    return {"message": "✅ Instructions cleared."}

@app.post("/instructions/add")
def add_instruction(payload: dict):
    instruction = payload.get("instruction", "").strip()
    if instruction:
        persistent_instructions.append(instruction)
        return {"message": "Instruction added successfully."}
    return {"error": "Instruction cannot be empty."}