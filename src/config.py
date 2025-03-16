from pydantic_settings import BaseSettings
from google.auth import default as get_default_credentials


class Config(BaseSettings):
    GCP_PROJECT_ID: str = "medium-demo-proj"
    GOOGLE_APPLICATION_CREDENTIALS: str | None = None
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 90
    AUTH_SECRET_KEY: str
    DB_CONNECTION_STR: str
    CHAT_HF_MODEL_ID: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.GOOGLE_APPLICATION_CREDENTIALS is None:
            _, project = get_default_credentials()
            if project:
                self.GCP_PROJECT_ID = project
