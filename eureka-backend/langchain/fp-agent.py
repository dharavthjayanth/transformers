import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_experimental.agents import create_pandas_dataframe_agent

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o", temperature=0.5, openai_api_key=api_key)

csv_path = "/Users/keshavsaraogi/Desktop/indorama/eureka-data/clean-csv/cleaned_finance_packaging.csv"
df = pd.read_csv(csv_path)

engine = create_engine("sqlite:///finance.db")
df.to_sql("finance_packaging_data", engine, index=False, if_exists="replace")
db = SQLDatabase(engine)

finance_csv_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
finance_sql_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
viz_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


csv_agent_executor = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    agent_type="openai-functions",
    allow_dangerous_code=True
)

sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an agent who uses SQL to answer questions based on a given database."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

sql_agent = create_openai_functions_agent(
    llm=llm,
    prompt=sql_prompt,
    tools=sql_toolkit.get_tools()
)

sql_agent_executor = AgentExecutor(
    agent=sql_agent,
    tools=sql_toolkit.get_tools(),
    memory=finance_sql_memory,
    verbose=True
)

viz_agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    agent_type="openai-functions",
    memory=viz_memory,
    allow_dangerous_code=True
)

def visualize_with_agent(prompt: str):
    return viz_agent.invoke({"input": prompt})

def route_query(query: str) -> str:
    keywords = ["sum", "average", "group", "balance", "debit", "credit", "filter"]
    return "sql" if any(word in query.lower() for word in keywords) else "csv"

def master_agent(query: str) -> str:
    source = route_query(query)
    if source == "sql":
        return sql_agent_executor.invoke({"input": query}).get("output", "")
    return csv_agent_executor.invoke({"input": query}).get("output", "")

def recall_last_interaction(memory):
    msgs = memory.chat_memory.messages
    if len(msgs) >= 2:
        return f"🧠 Last Question: {msgs[-2].content}\n💬 Last Answer: {msgs[-1].content}"
    return "🧠 No previous memory found."

def trace_memory(memory):
    print("🧠 Full Conversation:")
    for msg in memory.chat_memory.messages:
        print(f"{msg.type.upper()}: {msg.content}")

if __name__ == "__main__":
    # print(master_agent("What is the total balance for Adv To Sup Local for company TH14?"))
    # print(master_agent("What are the top 5 companies with highest total balance for Adv To Sup Local. Sort by balance."))
    
    # print("🔎 SQL MEMORY:")
    # trace_memory(finance_sql_memory)

    # print("🔎 CSV MEMORY:")
    # trace_memory(finance_csv_memory)
    # print(visualize_with_agent("Create a bar chart showing total Ending Balance in Global Currency for each Company Name."))
    print(visualize_with_agent("Show a bar chart for the top 5 companies by total Ending Balance in Global Currency."))
