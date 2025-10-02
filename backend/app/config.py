"""
Configurazione dell'applicazione AI Video Chat
"""

import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Configurazioni dell'applicazione"""
    
    # Database
    database_url: str = "sqlite:///./ai_video_chat.db"
    
    # OpenAI
    openai_api_key: str = "sk-your-openai-api-key-here"
    
    # Security
    secret_key: str = "ai-video-chat-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # Video processing
    max_video_size_mb: int = 100
    supported_video_formats: str = "mp4,avi,mov,mkv,webm"
    
    # App settings
    debug: bool = True
    
    @property
    def supported_video_formats_list(self ) -> List[str]:
        return self.supported_video_formats.split(",")
    
    class Config:
        env_file = ".env"

# Istanza globale delle impostazioni
settings = Settings()
