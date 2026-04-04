import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

_VECTORSTORE = None
_LLM = None
_RETRIEVER = None
_QA_CHAIN = None


def _initialize():
    global _VECTORSTORE, _LLM, _RETRIEVER, _QA_CHAIN
    if _VECTORSTORE is None:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        _VECTORSTORE = FAISS.load_local(
            DB_FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True,
        )
    if _LLM is None:
        _LLM = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    if _RETRIEVER is None:
        _RETRIEVER = _VECTORSTORE.as_retriever(search_kwargs={"k": 3})
    if _QA_CHAIN is None:
        _QA_CHAIN = (
            {"context": _RETRIEVER | format_docs, "question": RunnablePassthrough()}
            | PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])
            | _LLM
            | StrOutputParser()
        )


def get_vectorstore():
    _initialize()
    return _VECTORSTORE


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run_rag_query(query: str) -> dict:
    _initialize()
    if _QA_CHAIN is None or _RETRIEVER is None:
        raise RuntimeError("RAG pipeline is not initialized")
    answer = _QA_CHAIN.invoke(query)
    source_documents = _RETRIEVER.invoke(query)
    sources = [doc.page_content[:300] for doc in source_documents]

    return {"answer": answer, "sources": sources}
