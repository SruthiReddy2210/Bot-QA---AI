import os
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="db")

collection = client.get_or_create_collection(
    name="documents"
)

model = SentenceTransformer("all-MiniLM-L6-v2")

all_chunks = []
all_meta = []
all_ids = []

chunk_id = 0

for file in os.listdir("data"):

    if file.endswith(".pdf"):

        reader = PdfReader(os.path.join("data", file))

        for page_num, page in enumerate(reader.pages):

            text = page.extract_text()

            if not text:
                continue

            chunk_size = 1000
            overlap = 200

            start = 0

            while start < len(text):

                end = min(start + chunk_size, len(text))

                chunk = text[start:end]

                all_chunks.append(chunk)

                all_meta.append({
                    "source": file,
                    "page": page_num + 1
                })

                all_ids.append(str(chunk_id))

                chunk_id += 1

                start += chunk_size - overlap

embeddings = model.encode(all_chunks).tolist()

collection.add(
    documents=all_chunks,
    embeddings=embeddings,
    metadatas=all_meta,
    ids=all_ids
)

print("Indexing Complete")