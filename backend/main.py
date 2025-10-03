"""
AI Video Chat - Backend API
Piattaforma per conversazioni intelligenti con contenuti video
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.routers import auth, upload, chat
import uvicorn
import structlog
from app.config import settings
from dotenv import load_dotenv
load_dotenv()  # Carica le variabili dal file .env


# Configurazione logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Creazione app FastAPI
app = FastAPI(
    title="AI Video Chat API",
    description="API per conversazioni intelligenti con contenuti video",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurazione CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gestione errori globale
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Internal server error", error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Endpoint di base
@app.get("/")
async def root():
    """Endpoint di benvenuto"""
    return {
        "message": "AI Video Chat API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check per monitoraggio"""
    return {
        "status": "healthy",
        "service": "ai-video-chat-backend"
    }

# Eventi di startup e shutdown
@app.on_event("startup")
async def startup_event():
    """Inizializzazione dell'applicazione"""
    logger.info("AI Video Chat API starting up...")
    
    # Qui aggiungeremo l'inizializzazione del database
    # e altri servizi quando necessario
    
    logger.info("AI Video Chat API started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Pulizia risorse all'arresto"""
    logger.info("AI Video Chat API shutting down...")

# Importazione router (li aggiungeremo dopo)
# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(chat.router, prefix="/api", tags=["chat"])


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    )
