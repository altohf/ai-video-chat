from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import uuid
from pathlib import Path

router = APIRouter()

# Crea directory per upload se non esiste
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    # Verifica tipo file
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File deve essere un video")
    
    # Verifica dimensione (max 100MB)
    if file.size > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File troppo grande (max 100MB)")
    
    # Genera nome file unico
    file_extension = file.filename.split('.')[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = UPLOAD_DIR / unique_filename
    
    # Salva file
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return JSONResponse({
            "message": "Video caricato con successo",
            "filename": unique_filename,
            "size": len(content),
            "status": "uploaded"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel salvare il file: {str(e)}")
