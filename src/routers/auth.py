from fastapi import APIRouter
from fastapi import Depends
from typing import Annotated
from models.auth import User
from auth.auth_utils import get_current_active_user
from config import Config

CONFIG = Config()

auth_router = APIRouter(prefix="/auth", tags=["auth"])


@auth_router.get("/users/me/", response_model=User)
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    return current_user
