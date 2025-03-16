from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "mock-gpt-model"
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.1
    stream: bool = False
