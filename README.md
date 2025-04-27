# Project Structure

```plaintext
backend/                  # Node.js API server
├── src/
│   ├── controllers/       # Business logic
│   ├── routes/            # Express route definitions
│   ├── services/          # PythonShell bridge & helpers
│   ├── models/            # Mongoose schemas
│   ├── scripts/           # CSV import script
│   └── app.js             # Express app entrypoint
├── .env                   # Env vars for backend
├── package.json           # Node.js dependencies & scripts

python/                   # RAG + LLM services
├── notebooks/             
│   ├── data_prep.ipynb     # Jupyter experiments
│   └── train_rag.ipynb     # 
├── services/
│   ├── embedder.py         # Embed & index FAISS
│   ├── retriever.py        # Retrieval + LLM generation
├── requirements.txt       # Python dependencies

indexes/                  # FAISS index files (ignored)
├── faiss.idx

.gitignore
README.md
