import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableMap
from langchain_groq import ChatGroq


# =========================
# 1. SETUP LLM (HF API)
# =========================


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.5
)





# =========================
# 2. PROMPT
# =========================

CUSTOM_PROMPT_TEMPLATE = """You are an informational medical assistant.
You are NOT a doctor.

Rules:
- Use ONLY the provided context.
- If the answer is not in the context, say you do not know.
- Do NOT provide diagnosis, treatment, or medical advice.
- For serious medical conditions, clearly state that a qualified medical professional must be consulted.

Context:
{context}

Question:
{question}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful medical information assistant."),
    ("user", CUSTOM_PROMPT_TEMPLATE)
])


# =========================
# 3. LOAD FAISS VECTOR STORE
# =========================

DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)


# =========================
# 4. RETRIEVER
# =========================

retriever = db.as_retriever(search_kwargs={"k": 3})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# =========================
# 5. LCEL RAG PIPELINE
# =========================

qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# =========================
# 6. RUN QUERY
# =========================

while True:
    user_query = input("\nWrite Query (or 'exit'): ")
    if user_query.lower() == "exit":
        break

    docs = retriever.invoke(user_query)

    if not docs:
        print("\nANSWER:\nI don't know. The information is not available in my documents.")
        continue

    answer = qa_chain.invoke(user_query)
    print("\nANSWER:\n", answer)

    print("\nSOURCE DOCUMENTS:")
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Source {i} ---")
        print(doc.page_content[:500])