from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class User(SQLModel, table=True):
    __tablename__ = "auth.users"
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(unique=True)
    api_key_hash: str
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    is_active: bool
