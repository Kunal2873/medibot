from fastapi import FastAPI, HTTPException

from app.crag_pipeline import run_crag_query
from app.schemas import ChatRequest, ChatResponse

app = FastAPI()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = run_crag_query(request.query)
        return ChatResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
