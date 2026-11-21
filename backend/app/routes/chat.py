"""Chat endpoint: answer a question from the uploaded documents."""
from fastapi import APIRouter, Request

from backend.app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    return request.app.state.rag_service.answer(payload.question, payload.history)
