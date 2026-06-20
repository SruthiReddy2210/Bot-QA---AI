# Document QA Bot

## Tech Stack
Python
Gemini
ChromaDB
PyPDF

## Architecture
Documents
→ Chunking
→ Embedding
→ ChromaDB
→ Retrieval
→ Gemini
→ Answer

## Chunking Strategy
1000 chars
200 overlap

## Setup

pip install -r requirements.txt

python src/ingest.py

python src/query.py

## Environment Variables

GEMINI_API_KEY

## Example Queries

1. What is machine learning?
2. Summarize climate report
3. AI trends
4. Business findings
5. Technology overview

## Limitations

Small document collection.