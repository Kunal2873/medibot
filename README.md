# MediBot — Medical RAG Chatbot 🩺

MediBot is a **Retrieval-Augmented Generation (RAG)** based medical chatbot built using **LangChain**, **FAISS**, **Groq-hosted LLMs**, and **Streamlit**.

It allows users to ask medical questions and receive answers grounded strictly in a pre-indexed medical knowledge base, avoiding hallucinations and unsupported claims.

---

## 🚀 Features

* 🔍 Semantic search using **FAISS vector store**
* 🧠 Context-aware answers via **Groq LLM (LLaMA-based)**
* 📄 Answers restricted strictly to retrieved documents
* 💬 Interactive chat UI built with **Streamlit**
* 🔐 Secure API key handling using environment variables

---

## 🏗️ Architecture (High Level)

```
User Query
   ↓
FAISS Retriever (Top-k documents)
   ↓
Prompt + Context
   ↓
Groq LLM
   ↓
Grounded Answer
```

This follows a **modern explicit RAG pipeline** (no deprecated LangChain chains).

---

## 🛠️ Tech Stack

* **Python 3.12**
* **LangChain**
* **FAISS**
* **Groq API**
* **HuggingFace Embeddings**
* **Streamlit**

---

## 📂 Project Structure

```
medibot/
│
├── medibot.py                  # Streamlit app entry point
├── memory.py                   # Vector store creation logic
├── connect_memory_with_llm.py  # RAG pipeline wiring
├── requirements.txt
├── .gitignore
├── README.md
│
├── data/            # (ignored) source documents
├── vectorstore/     # (ignored) FAISS index
└── venv/            # (ignored) virtual environment
```

> ⚠️ `data/`, `vectorstore/`, `.env`, and `venv/` are intentionally excluded from GitHub.

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

**Do NOT commit this file.**
It is already ignored via `.gitignore`.

---

## 📦 Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Kunal2873/medibot.git
cd medibot
```

### 2️⃣ Create virtual environment (Python 3.12)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
streamlit run medibot.py
```

Then open the browser at:

```
http://localhost:8501
```

---

## 📌 Notes

* The FAISS vector store must be generated locally before running queries.
* This project intentionally avoids deprecated LangChain abstractions.
* Answers are **strictly grounded in retrieved documents** — no hallucinations.

---

## ⚠️ Disclaimer

This project is for **educational and experimental purposes only**.
It is **not a medical diagnostic tool** and should not be used as a substitute for professional medical advice.

---

## 👤 Author

**Kunal Salaria**
B.Tech | AI / ML / GenAI
GitHub: [https://github.com/Kunal2873](https://github.com/Kunal2873)
