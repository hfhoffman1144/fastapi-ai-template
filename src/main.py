from fastapi import Depends, FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from auth.auth_utils import (
    create_db_and_tables,
    get_current_active_user,
)
from contextlib import asynccontextmanager
from config import Config
from routers.auth import auth_router
from routers.chat_models import chat_models_router

CONFIG = Config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield
    # TODO: Release GPU resources


app = FastAPI(
    title="FastAPI AI Template",
    description="Deploy open-source AI models that you control",
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)

app.include_router(auth_router)
app.include_router(chat_models_router)


@app.get("/")
async def get_status(token: str = Depends(get_current_active_user)):
    return {"status": "running"}


@app.get("/docs")
async def get_documentation(token: str = Depends(get_current_active_user)):
    return get_swagger_ui_html(openapi_url="/openapi.json", title="docs")


@app.get("/openapi.json")
async def openapi(token: str = Depends(get_current_active_user)):
    return get_openapi(title="FastAPI", version="0.1.0", routes=app.routes)
