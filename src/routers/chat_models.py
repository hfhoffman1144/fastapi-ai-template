import torch
import time
import asyncio
import json
from fastapi import APIRouter
from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import Annotated
from models.auth import User
from models.chat_models import ChatCompletionRequest, ChatMessage
from auth.auth_utils import (
    get_current_active_user,
)
from chat_models.chat_model_interface import ChatModelInterface
from config import Config

CONFIG = Config()

MODEL_INTERFACE = ChatModelInterface(
    model_id=CONFIG.CHAT_HF_MODEL_ID, torch_dtype=torch.float16
)

chat_models_router = APIRouter(prefix="/chat", tags=["chat"])


async def _model_stream_async_generator(request: ChatCompletionRequest):

    message = " ".join([m.content for m in request.messages])

    model_content_gen = MODEL_INTERFACE.stream(
        message=message,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    for i, chunk_text in enumerate(model_content_gen):
        chunk = {
            "id": i,
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": CONFIG.CHAT_HF_MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk_text or ""},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"


@chat_models_router.post("/completions")
async def complete_chat(
    request: ChatCompletionRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """Open AI-compatible chat completion endpoint"""

    print(request)

    try:

        if request.stream:
            return StreamingResponse(
                _model_stream_async_generator(request),
                media_type="application/x-ndjson",
            )

        message = " ".join([m.content for m in request.messages])

        model_content = MODEL_INTERFACE.invoke(
            message=message,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        response = {
            "id": "0",
            "object": "chat.completion",
            "created": time.time(),  # TODO: Make utc
            "model": CONFIG.CHAT_HF_MODEL_ID,
            "choices": [
                {"message": ChatMessage(role="assistant", content=model_content)}
            ],
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=e)
