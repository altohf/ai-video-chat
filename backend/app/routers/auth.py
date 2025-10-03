from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt
from datetime import datetime, timedelta

router = APIRouter()
security = HTTPBearer()

# Mock secret key - in produzione usare variabile d'ambiente
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

@router.post("/login")
async def login(request: LoginRequest):
    # Mock login - in produzione verificare con database
    if request.username == "demo" and request.password == "demo":
        # Crea token JWT
        token_data = {
            "sub": request.username,
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": {"username": request.username, "email": "demo@example.com"}
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenziali non valide"
        )

@router.post("/register")
async def register(request: RegisterRequest):
    # Mock registration
    return {
        "message": "Utente registrato con successo",
        "user": {"username": request.username, "email": request.email}
    }

@router.get("/me")
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Token non valido")
        return {"username": username, "email": "demo@example.com"}
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Token non valido")
