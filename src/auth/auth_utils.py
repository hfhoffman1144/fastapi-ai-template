from sqlmodel import SQLModel, create_engine, Session, select
from datetime import datetime, timezone
from google.cloud import bigquery
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import Annotated
from models.auth import User
from config import Config
from utils.gcp import create_big_query_dataset

CONFIG = Config()
DB_ENGINE = create_engine(CONFIG.DB_CONNECTION_STR, echo=True)
PWD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")
BQ_CLIENT = bigquery.Client()
OAUTH2_SCHEME = OAuth2PasswordBearer(tokenUrl="token")

create_big_query_dataset(
    bq_client=BQ_CLIENT,
    project_id=CONFIG.GCP_PROJECT_ID,
    dataset_id="auth",
    exists_ok=True,
)


def create_db_and_tables() -> None:
    """Initialize SQLModel engine"""
    SQLModel.metadata.create_all(DB_ENGINE)


def verify_api_key(plain_token: str, hashed_token: str) -> bool:
    """Verify that a plaintext API token matches a hashed API token"""
    return PWD_CONTEXT.verify(plain_token, hashed_token)


def get_api_key_hash(api_key: str) -> str:
    """Hash an API token"""
    return PWD_CONTEXT.hash(api_key)


def get_max_user_id() -> int | None:
    """Get the maximum id from the User table"""
    with Session(DB_ENGINE) as session:
        # Select the max id from the User table
        statement = select(User.id).order_by(User.id.desc()).limit(1)
        result = session.exec(statement).first()

    return result if result is not None else None


def update_user(user: User, **kwargs):
    """Update a user record"""
    with Session(DB_ENGINE) as session:

        for key, value in kwargs.items():
            setattr(user, key, value)

        user.updated_at = datetime.now(timezone.utc)

        session.add(user)
        session.commit()
        session.refresh(user)


def get_user_from_api_key(api_key: str) -> User | None:
    """Search the database for an active API token"""

    user_id = api_key.split("-")[0]

    with Session(DB_ENGINE) as session:
        statement = select(User).where(
            (User.user_id == user_id) & (User.is_active == True)
        )
        results = session.exec(statement).all()

    if len(results) == 0:
        return None

    user = results[0]

    if not verify_api_key(api_key, user.api_key_hash):
        return None

    expires_at = user.expires_at

    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if datetime.now(timezone.utc) >= expires_at:
        update_user(user, updated_at=datetime.now(timezone.utc), is_active=False)
        return None

    return user


def authenticate_api_key(api_key: str) -> User | None:
    """Authenticate an API token"""

    user = get_user_from_api_key(api_key)

    return user


async def get_current_user(api_key: Annotated[str, Depends(OAUTH2_SCHEME)]) -> User:
    """Get the current user from a JWT"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired API key",
        headers={"WWW-Authenticate": "Bearer"},
    )
    user = get_user_from_api_key(api_key)

    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Get the current active user"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
