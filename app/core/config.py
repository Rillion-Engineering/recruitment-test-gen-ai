from pydantic import SecretStr
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    AZURE_OPENAI_API_KEY: SecretStr
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "RAG Query API"
    VERSION: str = "0.1.0"
    QDRANT_URL: str = "127.0.0.1"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "data"
    AZURE_OPENAI_ENDPOINT: str = "https://devrillionopenai.openai.azure.com"
    # CORS Configuration
    BACKEND_CORS_ORIGINS: list[str] = ["*"]  # In production, replace with specific origins

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 