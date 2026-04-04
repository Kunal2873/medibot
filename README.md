# MediBot — Agentic Medical RAG Chatbot 🩺

MediBot is a **Corrective RAG (CRAG)** based medical chatbot built with **LangGraph**, **FastAPI**, **FAISS**, **Groq-hosted LLMs**, and **Docker**.

It retrieves answers strictly from a medical knowledge base, grades retrieved documents for relevance, rewrites weak queries, and refuses to answer from general knowledge — no hallucinations.

---

## 🌐 Live API

🔗 **FastAPI Backend (Render)**
https://medibot-backend-hym8.onrender.com

📖 **Interactive API Docs (Swagger UI)**
https://medibot-backend-hym8.onrender.com/docs

> ⚠️ Free tier cold start may take 30–50 seconds on first request.

---

## 🚀 Features

* 🤖 **Agentic CRAG pipeline** built with **LangGraph**
* 🔍 Semantic search using **FAISS** + **FastEmbed** (no PyTorch dependency)
* ✅ **Document grading** — filters irrelevant retrieved chunks before generation
* 🔄 **Query rewriting** — rephrases weak queries and retries retrieval once
* 🚫 **Hard context guard** — returns "I don't know" when no relevant docs found
* ⚡ **Dual-model strategy** — 8B model for grading, 70B for generation
* 🌐 **FastAPI REST backend** with Pydantic validation and Swagger docs
* 🐳 **Fully containerized** with Docker and Docker Compose
* 🚀 **Deployed on Render** via Docker

---

## 🏗️ Architecture
```
User Query
   ↓
FastAPI /chat endpoint
   ↓
LangGraph CRAG Pipeline
   ├── retrieve         → FAISS top-k retrieval
   ├── grade_documents  → LLM relevance grading (8B model)
   ├── rewrite_query    → Query rewriting + retry (if all docs fail grading)
   └── generate         → Final answer generation (70B model)
   ↓
JSON Response (answer + sources)
   ↓
Streamlit Frontend
```

---

## 🛠️ Tech Stack

* **Python 3.13**
* **LangGraph** — agentic CRAG graph
* **LangChain** — LCEL pipeline, FAISS integration
* **FastAPI** — REST API backend
* **FAISS** — vector similarity search
* **FastEmbed** — lightweight embeddings (BAAI/bge-small-en-v1.5)
* **Groq API** — LLaMA 3.3 70B (generation) + LLaMA 3.1 8B (grading)
* **Streamlit** — chat frontend
* **Docker + Docker Compose** — containerization
* **Render** — cloud deployment

---

## 📂 Project Structure
```
medibot/
│
├── app/
│   ├── main.py              # FastAPI app — /chat and /health endpoints
│   ├── crag_pipeline.py     # LangGraph CRAG pipeline
│   └── schemas.py           # Pydantic request/response models
│
├── frontend.py              # Streamlit chat UI
├── memory.py                # FAISS index creation script (run once)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env                     # (ignored) API keys
├── data/                    # (ignored) source PDFs
└── vectorstore/             # (ignored) FAISS index
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 📦 Running Locally with Docker
```bash
git clone https://github.com/Kunal2873/medibot.git
cd medibot
```

Add your `.env` file, then:
```bash
docker compose up --build
```

- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- Swagger docs: http://localhost:8000/docs

---

## 📦 Running Locally without Docker
```bash
pip install -r requirements.txt
python memory.py  # build FAISS index once
```

Terminal 1:
```bash
python -m uvicorn app.main:app --port 8000
```

Terminal 2:
```bash
streamlit run frontend.py
```

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.
It is **not a medical diagnostic tool** and should not replace professional medical advice.

---

## 👤 Author

**Kunal Salaria**
B.Tech | IT | AI / ML / GenAI
GitHub: [https://github.com/Kunal2873](https://github.com/Kunal2873)
