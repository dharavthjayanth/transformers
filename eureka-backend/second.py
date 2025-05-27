# import os
# from dotenv import load_dotenv
# from pinecone import Pinecone
# from openai import OpenAI

# # Load environment variables
# load_dotenv()

# # ENV variables
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# # Init Pinecone
# pc = Pinecone(api_key=PINECONE_API_KEY)
# index = pc.Index(PINECONE_INDEX_NAME)

# # Init OpenAI client
# client = OpenAI(api_key=OPENAI_API_KEY)

# # Query
# query = "What is the total spend on Additive in 2024?"

# # Create embedding (new SDK usage)
# embedding_response = client.embeddings.create(
#     model="text-embedding-3-small",
#     input=query
# )
# embedding = embedding_response.data[0].embedding

# # Query Pinecone
# res = index.query(vector=embedding, top_k=5, include_metadata=True)
# contexts = [match["metadata"]["text"] for match in res["matches"]]
# context_text = "\n".join(contexts)

# # Final prompt
# final_prompt = f"""You are a helpful assistant. Based on the following context, answer the question:

# Context:
# {context_text}

# Question: {query}
# Answer:"""

# # Chat completion (new SDK usage)
# response = client.chat.completions.create(
#     model="gpt-4",
#     messages=[{"role": "user", "content": final_prompt}]
# )

# # Output
# print("📤 RAG Answer:\n", response.choices[0].message.content)

from openai import OpenAI

client = OpenAI(api_key="your-api-key")
res = client.embeddings.create(model="text-embedding-3-small", input="hello world")
print(res.data[0].embedding[:5])
