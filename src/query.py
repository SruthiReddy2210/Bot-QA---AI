import os
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=API_KEY)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="db")

collection = client.get_collection("documents")

llm = genai.GenerativeModel("gemini-2.5-flash")

while True:

    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        break

    query_embedding = embed_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    context = ""

    for doc in results["documents"][0]:
        context += doc + "\n\n"

    prompt = f"""
Use ONLY the context below.

If answer is not present say:
'I cannot find the answer in the documents.'

Context:
{context}

Question:
{question}
"""

    response = llm.generate_content(prompt)

    print("\nANSWER:\n")
    print(response.text)

    print("\nCITATIONS:")

    for meta in results["metadatas"][0]:
        print(f"{meta['source']} Page {meta['page']}")