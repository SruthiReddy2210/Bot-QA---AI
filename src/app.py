import streamlit as st
import os
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Load API Key from Streamlit Secrets
API_KEY = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=API_KEY)

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB
client = chromadb.PersistentClient(path="db")
collection = client.get_collection("documents")

# Faster Gemini model
llm = genai.GenerativeModel("gemini-pro")

st.title("📄 Document Q&A Bot")
st.write("Ask questions about the uploaded documents.")

question = st.text_input("Ask a question")

if st.button("Get Answer"):

    if question.strip():

        try:
            # Create query embedding
            query_embedding = embed_model.encode(question).tolist()

            # Retrieve only top result
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=1
            )

            # Limit context size
            context = ""

            for doc in results["documents"][0]:
                context += doc[:500] + "\n\n"

            prompt = f"""
Answer ONLY from the provided context.

If the answer is not present, reply:
"I cannot find the answer in the documents."

Context:
{context}

Question:
{question}
"""

            response = llm.generate_content(prompt)

            st.subheader("Answer")
            st.write(response.text)

            st.subheader("Citations")

            for meta in results["metadatas"][0]:
                st.write(
                    f"📄 {meta['source']} | Page {meta['page']}"
                )

        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.warning("Please enter a question.")