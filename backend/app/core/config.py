from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    APP_NAME: str = "Counterfactual Vision Auditor API"
    APP_ENV: str = "development"
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"]
    MAX_IMAGE_UPLOAD_MB: int = 10
    MODEL_DEVICE: str = "auto"
    CHECKPOINT_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "outputs", "checkpoints"))
    METRICS_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "outputs", "metrics"))
    FIGURES_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "outputs", "figures"))
    MASKS_CACHE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "outputs", "masks"))

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()
