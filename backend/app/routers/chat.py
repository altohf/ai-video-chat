from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
from pathlib import Path
from app.services.video_analyzer import VideoAnalyzer

router = APIRouter()

class ChatMessage(BaseModel):
    message: str
    video_id: str = None

class VideoAnalysisRequest(BaseModel):
    video_filename: str

# Storage per le analisi video (in produzione usare database)
video_analyses = {}

@router.post("/chat")
async def chat_with_ai(chat_message: ChatMessage):
    try:
        if not chat_message.video_id:
            return {"response": "Carica prima un video per iniziare la conversazione!"}
        
        # Verifica se abbiamo l'analisi del video
        if chat_message.video_id not in video_analyses:
            return {"response": "Video non ancora analizzato. Usa prima l'endpoint /api/analyze per analizzare il video."}
        
        video_analysis = video_analyses[chat_message.video_id]
        
        if not video_analysis.get("success"):
            return {"response": "Errore nell'analisi del video. Riprova ad analizzarlo."}
        
        # 🚀 CHRIS: Crea client OpenAI dentro la funzione per evitare errori di import
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # 🚀 CHRIS: Prompt migliorato con trascrizione audio
        analysis_text = video_analysis['analysis']
        transcription = video_analysis.get('transcription', '')
        
        system_prompt = f"""Sei un assistente AI esperto nell'analisi di video. Hai già analizzato completamente un video e hai a disposizione:

📹 ANALISI VISIVA:
{analysis_text}

🎤 TRASCRIZIONE AUDIO:
{transcription}

Rispondi alle domande dell'utente basandoti su queste informazioni complete. Puoi fare riferimento sia al contenuto visivo che a quello audio. Sii preciso e dettagliato, citando specifici momenti o elementi quando possibile."""

        # Usa GPT-4o per compatibilità garantita
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chat_message.message}
            ],
            max_tokens=800,
            temperature=0.3
        )
        
        ai_response = response.choices[0].message.content
        
        return {
            "response": ai_response,
            "video_id": chat_message.video_id,
            "has_transcription": bool(transcription),
            "analysis_quality": video_analysis.get('performance', {}).get('quality_score', 'unknown')
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nella chat: {str(e)}")

@router.post("/analyze")
async def analyze_video(request: VideoAnalysisRequest):
    try:
        # Percorso del video caricato
        video_path = Path("uploads") / request.video_filename
        
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video non trovato")
        
        # Inizializza l'analyzer
        analyzer = VideoAnalyzer()
        
        # Analizza il video
        analysis_result = await analyzer.analyze_video(str(video_path))
        
        if analysis_result.get("success"):
            # 🚀 CHRIS: Salva l'analisi completa con tutti i dati
            video_id = request.video_filename.split('.')[0]
            video_analyses[video_id] = analysis_result
            
            # Informazioni dettagliate per il response
            frames_count = analysis_result.get('frames_analyzed', 0)
            has_audio = bool(analysis_result.get('transcription', ''))
            architecture = analysis_result.get('architecture', 'unknown')
            cost_efficiency = analysis_result.get('cost_efficiency', 'unknown')
            
            return {
                "status": "analyzed",
                "summary": f"Video analizzato con successo! {frames_count} frame processati.",
                "video_id": video_id,
                "analysis_preview": analysis_result["analysis"][:300] + "...",
                "features": {
                    "audio_transcription": has_audio,
                    "frames_analyzed": frames_count,
                    "architecture": architecture,
                    "cost_efficiency": cost_efficiency
                },
                "ready_for_chat": True
            }
        else:
            raise HTTPException(status_code=500, detail=analysis_result.get("error", "Errore sconosciuto"))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nell'analisi: {str(e)}")

@router.get("/debug/video/{video_id}")
async def debug_video_analysis(video_id: str):
    """
    🔍 CHRIS: Endpoint di debug per verificare l'analisi salvata
    """
    if video_id not in video_analyses:
        raise HTTPException(status_code=404, detail="Video non trovato")
    
    analysis = video_analyses[video_id]
    
    return {
        "video_id": video_id,
        "success": analysis.get("success", False),
        "has_analysis": bool(analysis.get("analysis", "")),
        "has_transcription": bool(analysis.get("transcription", "")),
        "analysis_length": len(analysis.get("analysis", "")),
        "transcription_length": len(analysis.get("transcription", "")),
        "frames_analyzed": analysis.get("frames_analyzed", 0),
        "architecture": analysis.get("architecture", "unknown"),
        "performance": analysis.get("performance", {}),
        "full_analysis": analysis.get("analysis", "")[:500] + "..." if len(analysis.get("analysis", "")) > 500 else analysis.get("analysis", ""),
        "full_transcription": analysis.get("transcription", "")[:500] + "..." if len(analysis.get("transcription", "")) > 500 else analysis.get("transcription", "")
    }

@router.get("/debug/storage")
async def debug_storage():
    """
    🔍 CHRIS: Endpoint per vedere tutti i video in storage
    """
    return {
        "videos_count": len(video_analyses),
        "video_ids": list(video_analyses.keys()),
        "storage_details": {
            video_id: {
                "success": data.get("success", False),
                "has_analysis": bool(data.get("analysis", "")),
                "has_transcription": bool(data.get("transcription", "")),
                "frames_analyzed": data.get("frames_analyzed", 0)
            }
            for video_id, data in video_analyses.items()
        }
    }
