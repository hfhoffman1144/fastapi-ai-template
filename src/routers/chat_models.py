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
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from config import Config

CONFIG = Config()

engine_args = AsyncEngineArgs(
    model=CONFIG.CHAT_HF_MODEL_ID,
    enforce_eager=True,
    dtype="float16",
    gpu_memory_utilization=1,  # TODO: Parameterize,
    max_model_len=1904,
)
MODEL_INTERFACE = AsyncLLMEngine.from_engine_args(engine_args)


chat_models_router = APIRouter(prefix="/chat", tags=["chat"])


async def _model_stream_async_generator(request: ChatCompletionRequest):

    message = " ".join([m.content for m in request.messages])

    sampling_params = SamplingParams(
        temperature=request.temperature, max_tokens=request.max_tokens
    )

    model_content_gen = MODEL_INTERFACE.generate(
        message, sampling_params, request_id=time.monotonic()
    )

    i = 0
    previous_text = ""

    async for request_output in model_content_gen:

        curr_full_text = request_output.outputs[0].text
        chunk_text = curr_full_text[len(previous_text) :]
        previous_text = curr_full_text

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

        i += 1

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
