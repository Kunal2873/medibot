import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

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
_GRADER_LLM = None
_RETRIEVER = None
_ANSWER_CHAIN = None
_GRAPH = None


class CRAGState(TypedDict):
    query: str
    documents: list
    answer: str
    rewrite: bool
    retry_count: int


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def _initialize():
    global _VECTORSTORE, _LLM, _GRADER_LLM, _RETRIEVER, _ANSWER_CHAIN, _GRAPH
    if _VECTORSTORE is None:
        embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
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
    if _GRADER_LLM is None:
        _GRADER_LLM = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.0,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    if _RETRIEVER is None:
        _RETRIEVER = _VECTORSTORE.as_retriever(search_kwargs={"k": 3})
    if _ANSWER_CHAIN is None:
        _ANSWER_CHAIN = PromptTemplate(
            template=CUSTOM_PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        ) | _LLM | StrOutputParser()
    if _GRAPH is None:
        _GRAPH = _build_graph()


def retrieve(state: CRAGState) -> dict:
    documents = _RETRIEVER.invoke(state["query"])
    return {"documents": documents}


def grade_documents(state: CRAGState) -> dict:
    relevant_docs = []
    question = state["query"]

    for doc in state["documents"]:
        grade_prompt = (
            "You are grading whether a retrieved document is relevant to a user question. "
            "Respond with only 'yes' or 'no'.\n\n"
            f"Question: {question}\n\n"
            f"Document:\n{doc.page_content}"
        )
        decision = _GRADER_LLM.invoke(grade_prompt).content.strip().lower()
        if decision == "yes":
            relevant_docs.append(doc)

    rewrite = len(relevant_docs) == 0
    return {"documents": relevant_docs, "rewrite": rewrite}


def rewrite_query(state: CRAGState) -> dict:
    rewrite_prompt = (
        "Rewrite the user query to be more specific and better suited for document retrieval. "
        "Return only the rewritten query.\n\n"
        f"Original query: {state['query']}"
    )
    rewritten_query = _LLM.invoke(rewrite_prompt).content.strip()
    return {"query": rewritten_query, "retry_count": state["retry_count"] + 1, "rewrite": False}


def generate(state: CRAGState) -> dict:
    if not state["documents"]:
        return {"answer": "I don't know. The information is not available in my knowledge base."}
    answer = _ANSWER_CHAIN.invoke(
        {
            "context": format_docs(state["documents"]),
            "question": state["query"],
        }
    )
    return {"answer": answer}


def route_after_grading(state: CRAGState) -> str:
    if state["rewrite"] and state["retry_count"] < 1:
        return "rewrite_query"
    return "generate"


def _build_graph():
    graph = StateGraph(CRAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "rewrite_query": "rewrite_query",
            "generate": "generate",
        },
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


def run_crag_query(query: str) -> dict:
    _initialize()
    final_state = _GRAPH.invoke(
        {
            "query": query,
            "documents": [],
            "answer": "",
            "rewrite": False,
            "retry_count": 0,
        }
    )
    sources = [doc.page_content[:300] for doc in final_state["documents"]]
    return {"answer": final_state["answer"], "sources": sources}
