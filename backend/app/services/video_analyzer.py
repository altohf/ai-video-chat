import cv2
import base64
import os
import json
import numpy as np
from pathlib import Path
from openai import OpenAI
from PIL import Image
import io
import subprocess
import tempfile
from typing import List, Dict, Tuple, Optional
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """
    🚀 CHRIS ADAPTIVE VIDEO ANALYZER 2.0
    Content Intelligence + Adaptive Strategy Framework
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY non trovata nelle variabili d'ambiente")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Content Type Patterns
        self.content_patterns = {
            "MATH_MODE": [
                "formula", "equazione", "calcolo", "teorema", "dimostrazione",
                "matematica", "algebra", "geometria", "derivata", "integrale",
                "funzione", "grafico", "variabile", "coefficiente", "radice"
            ],
            "AUDIO_MODE": [
                "intervista", "podcast", "conversazione", "dialogo", "discussione",
                "penso che", "secondo me", "la mia opinione", "credo che",
                "personalmente", "dal mio punto di vista", "ascoltatori"
            ],
            "PRESENTATION_MODE": [
                "slide", "presentazione", "grafico", "dati", "risultati",
                "statistica", "report", "analisi", "business", "azienda",
                "vendite", "marketing", "strategia", "obiettivi", "kpi"
            ],
            "DOCUMENTARY_MODE": [
                "guerra", "storia", "evento", "accadde", "secolo", "anno",
                "documentario", "racconto", "narrativa", "cronaca", "testimonianza",
                "archivio", "storico", "memoria", "passato", "epoca"
            ],
            "TUTORIAL_MODE": [
                "tutorial", "lezione", "impara", "come fare", "passo", "procedura",
                "istruzioni", "guida", "metodo", "tecnica", "esempio",
                "pratica", "esercizio", "dimostrazione", "spiegazione"
            ]
        }
        
        logger.info("🚀 CHRIS Adaptive VideoAnalyzer 2.0 initialized - Content Intelligence Active")

    def extract_audio_and_transcribe(self, video_path: str) -> Dict:
        """
        STEP 1: Estrazione audio PERFETTA e trascrizione prioritaria
        """
        try:
            logger.info("🎵 STEP 1: PRIORITY Audio extraction and transcription...")
            
            # Estrai audio dal video
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                audio_path = temp_audio.name
            
            # 🔧 CHRIS: FFmpeg con parametri ottimizzati per qualità audio
            ffmpeg_path = "C:\\ffmpeg\\bin\\ffmpeg.exe"
            
            if not os.path.exists(ffmpeg_path):
                logger.error(f"❌ FFmpeg not found at {ffmpeg_path}")
                return {"success": False, "transcription": "", "content_type": "UNKNOWN"}
            
            logger.info(f"✅ FFmpeg found - Extracting high-quality audio...")
            
            # Parametri ottimizzati per trascrizione perfetta
            cmd = [
                ffmpeg_path, "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit
                "-ar", "16000",  # Sample rate ottimale per Whisper
                "-ac", "1",  # Mono
                "-af", "highpass=f=80,lowpass=f=8000",  # Filtro per voce umana
                audio_path, "-y", "-loglevel", "error"
            ]
            
            logger.info(f"🔧 Executing optimized audio extraction...")
            result = subprocess.run(cmd, capture_output=True, shell=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"❌ Audio extraction failed: {result.stderr}")
                return {"success": False, "transcription": "", "content_type": "UNKNOWN"}
            
            # Verifica qualità audio estratto
            audio_size = os.path.getsize(audio_path)
            logger.info(f"✅ High-quality audio extracted: {audio_size} bytes")
            
            if audio_size == 0:
                logger.error("❌ Audio file is empty")
                return {"success": False, "transcription": "", "content_type": "UNKNOWN"}
            
            # Trascrizione con Whisper - parametri ottimizzati
            logger.info("🎤 Transcribing with Whisper (optimized parameters)...")
            
            with open(audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"],
                    language="it",  # Forza italiano per migliore accuratezza
                    prompt="Trascrivi accuratamente questo contenuto in italiano, mantenendo terminologia tecnica e nomi propri."
                )
            
            # Cleanup
            os.unlink(audio_path)
            
            # Analizza il tipo di contenuto dalla trascrizione
            content_type = self.analyze_content_type(transcript.text)
            
            # Estrai momenti chiave basati sul tipo di contenuto
            key_moments = self._extract_adaptive_key_moments(transcript, content_type)
            
            logger.info(f"✅ PERFECT Audio processed:")
            logger.info(f"📝 Transcription: {len(transcript.text)} characters")
            logger.info(f"🧠 Content Type: {content_type}")
            logger.info(f"🎯 Key Moments: {len(key_moments)} identified")
            
            return {
                "success": True,
                "transcription": transcript.text,
                "content_type": content_type,
                "key_moments": key_moments,
                "duration": getattr(transcript, 'duration', 0),
                "segments": getattr(transcript, 'segments', [])
            }
            
        except Exception as e:
            logger.error(f"❌ Audio processing failed: {str(e)}")
            return {"success": False, "transcription": "", "content_type": "UNKNOWN"}

    def analyze_content_type(self, transcription: str) -> str:
        """
        🧠 CONTENT INTELLIGENCE: Analizza il tipo di contenuto
        """
        transcription_lower = transcription.lower()
        
        # Conta occorrenze per ogni tipo
        type_scores = {}
        
        for content_type, keywords in self.content_patterns.items():
            score = 0
            for keyword in keywords:
                score += transcription_lower.count(keyword)
            type_scores[content_type] = score
        
        # Trova il tipo con score più alto
        best_type = max(type_scores, key=type_scores.get)
        best_score = type_scores[best_type]
        
        # Se nessun pattern forte, usa BALANCED
        if best_score < 3:
            return "BALANCED_MODE"
        
        logger.info(f"🧠 Content Analysis: {best_type} (score: {best_score})")
        return best_type

    def _extract_adaptive_key_moments(self, transcript, content_type: str) -> List[float]:
        """
        Estrai momenti chiave adattivi basati sul tipo di contenuto
        """
        key_moments = []
        
        if content_type == "MATH_MODE":
            # Per matematica: cerca quando si introducono nuovi concetti
            math_keywords = ["formula", "teorema", "dimostrazione", "esempio", "calcolo"]
            key_moments = self._find_keyword_moments(transcript, math_keywords)
            
        elif content_type == "AUDIO_MODE":
            # Per podcast: cerca cambi di argomento e punti salienti
            topic_keywords = ["primo punto", "secondo", "inoltre", "però", "quindi", "infine"]
            key_moments = self._find_keyword_moments(transcript, topic_keywords)
            
        elif content_type == "PRESENTATION_MODE":
            # Per presentazioni: cerca transizioni tra slide
            slide_keywords = ["slide", "grafico", "dati", "prossimo", "vediamo", "analizziamo"]
            key_moments = self._find_keyword_moments(transcript, slide_keywords)
            
        elif content_type == "DOCUMENTARY_MODE":
            # Per documentari: cerca eventi e date
            event_keywords = ["anno", "quando", "evento", "accadde", "storia", "periodo"]
            key_moments = self._find_keyword_moments(transcript, event_keywords)
            
        elif content_type == "TUTORIAL_MODE":
            # Per tutorial: cerca passi e istruzioni
            step_keywords = ["passo", "prima", "poi", "adesso", "facciamo", "vediamo come"]
            key_moments = self._find_keyword_moments(transcript, step_keywords)
        
        # Fallback: distribuzione uniforme
        if not key_moments and hasattr(transcript, 'duration'):
            duration = transcript.duration
            for i in range(5):
                key_moments.append((duration / 5) * i)
        
        return sorted(list(set(key_moments)))[:15]  # Max 15 momenti

    def _find_keyword_moments(self, transcript, keywords: List[str]) -> List[float]:
        """
        Trova momenti temporali basati su parole chiave
        """
        moments = []
        
        if hasattr(transcript, 'words') and transcript.words:
            for word in transcript.words:
                if any(keyword in word.word.lower() for keyword in keywords):
                    moments.append(word.start)
        
        return moments

    def adaptive_scene_detection(self, video_path: str, content_type: str) -> List[float]:
        """
        STEP 2: Scene detection adattiva basata sul tipo di contenuto
        """
        logger.info(f"🎬 STEP 2: Adaptive scene detection for {content_type}...")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Parametri adattivi
        if content_type == "MATH_MODE":
            # Matematica: rileva ogni cambio di formula/lavagna
            threshold = 0.8
            analysis_interval = int(fps * 0.5)  # Ogni 0.5 secondi
        elif content_type == "AUDIO_MODE":
            # Podcast: scene detection minimo (focus su audio)
            threshold = 0.5
            analysis_interval = int(fps * 5)  # Ogni 5 secondi
        elif content_type == "PRESENTATION_MODE":
            # Presentazioni: rileva cambi slide
            threshold = 0.7
            analysis_interval = int(fps * 1)  # Ogni secondo
        else:
            # Default bilanciato
            threshold = 0.7
            analysis_interval = int(fps * 2)  # Ogni 2 secondi
        
        scene_changes = []
        prev_hist = None
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % analysis_interval == 0:
                hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                
                if prev_hist is not None:
                    diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    
                    if diff < threshold:
                        timestamp = frame_count / fps
                        scene_changes.append(timestamp)
                        logger.info(f"📸 Scene change detected at {timestamp:.2f}s")
                
                prev_hist = hist
            
            frame_count += 1
        
        cap.release()
        
        logger.info(f"✅ Adaptive scene detection: {len(scene_changes)} changes ({content_type})")
        return scene_changes

    def adaptive_keyframe_selection(self, video_path: str, scene_changes: List[float], 
                                  audio_moments: List[float], content_type: str) -> List[Tuple[float, np.ndarray]]:
        """
        STEP 3: Selezione keyframe adattiva
        """
        logger.info(f"🎯 STEP 3: Adaptive keyframe selection for {content_type}...")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Strategia adattiva
        if content_type == "MATH_MODE":
            # Matematica: priorità ai cambi visivi (formule, grafici)
            moments = scene_changes + audio_moments[:3]  # Pochi momenti audio
            max_frames = 20
        elif content_type == "AUDIO_MODE":
            # Podcast: minimi frame, focus su audio
            moments = audio_moments + scene_changes[:2]  # Pochissimi cambi scena
            max_frames = 5
        elif content_type == "PRESENTATION_MODE":
            # Presentazioni: bilanciato slide + audio
            moments = scene_changes + audio_moments
            max_frames = 15
        elif content_type == "DOCUMENTARY_MODE":
            # Documentari: correlazione audio-visual
            moments = sorted(set(scene_changes + audio_moments))
            max_frames = 12
        else:
            # Bilanciato
            moments = sorted(set(scene_changes + audio_moments))
            max_frames = 10
        
        # Seleziona momenti più significativi
        selected_moments = sorted(set(moments))[:max_frames]
        
        selected_frames = []
        
        for timestamp in selected_moments:
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                quality = self._assess_frame_quality(frame)
                if quality > 0.2:  # Soglia bassa per adattività
                    selected_frames.append((timestamp, frame))
                    logger.info(f"🖼️ Keyframe selected at {timestamp:.2f}s (quality: {quality:.2f})")
        
        cap.release()
        
        logger.info(f"✅ Adaptive selection: {len(selected_frames)} frames for {content_type}")
        return selected_frames

    def _assess_frame_quality(self, frame: np.ndarray) -> float:
        """
        Valutazione qualità frame migliorata
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Nitidezza (Laplaciano)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(1.0, laplacian_var / 1000.0)
        
        # Contrasto
        contrast = gray.std() / 255.0
        
        # Luminosità (evita frame troppo scuri/chiari)
        brightness = gray.mean() / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2
        
        # Score combinato
        quality = (sharpness * 0.5 + contrast * 0.3 + brightness_score * 0.2)
        
        return quality

    def frame_to_base64(self, frame: np.ndarray) -> str:
        """
        Conversione frame ottimizzata
        """
        height, width = frame.shape[:2]
        max_size = 768  # Aumentato per migliore qualità
        
        if max(height, width) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)
            
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=75, optimize=True)
        
        base64_string = base64.b64encode(buffer.getvalue()).decode()
        size_kb = len(base64_string) / 1024
        logger.info(f"🔧 Frame: {frame.shape[1]}x{frame.shape[0]} → {size_kb:.1f}KB")
        
        return f"data:image/jpeg;base64,{base64_string}"

    async def analyze_video(self, video_path: str) -> Dict:
        """
        MAIN: Analisi adattiva completa
        """
        try:
            logger.info(f"🚀 CHRIS AI 2.0: Starting adaptive analysis of {Path(video_path).name}")
            
            # STEP 1: Audio-first analysis
            audio_result = self.extract_audio_and_transcribe(video_path)
            
            if not audio_result["success"]:
                logger.warning("⚠️ Audio failed, using visual-only mode")
                return await self._visual_only_fallback(video_path)
            
            content_type = audio_result["content_type"]
            transcription = audio_result["transcription"]
            
            # STEP 2: Adaptive scene detection
            scene_changes = self.adaptive_scene_detection(video_path, content_type)
            
            # STEP 3: Adaptive keyframe selection
            audio_moments = audio_result.get("key_moments", [])
            keyframes = self.adaptive_keyframe_selection(
                video_path, scene_changes, audio_moments, content_type
            )
            
            if not keyframes:
                logger.warning("⚠️ No keyframes, using fallback")
                return await self._visual_only_fallback(video_path, audio_result)
            
            # STEP 4: Adaptive multi-modal analysis
            logger.info(f"🧠 STEP 4: Adaptive analysis for {content_type}...")
            
            # Prompt adattivo basato sul tipo di contenuto
            analysis_prompt = self._generate_adaptive_prompt(content_type, transcription)
            
            content = [{"type": "text", "text": analysis_prompt}]
            
            # Aggiungi frame
            for i, (timestamp, frame) in enumerate(keyframes):
                base64_image = self.frame_to_base64(frame)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": base64_image, "detail": "high"}
                })
                logger.info(f"🔧 Added frame {i+1}/{len(keyframes)} at {timestamp:.1f}s")
            
            # Chiamata GPT-4o
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                max_tokens=3000,
                temperature=0.2
            )
            
            analysis_text = response.choices[0].message.content
            
            logger.info(f"✅ Adaptive analysis completed: {len(analysis_text)} chars")
            
            return {
                "success": True,
                "analysis": analysis_text,
                "transcription": transcription,
                "content_type": content_type,
                "frames_analyzed": len(keyframes),
                "architecture": f"Adaptive-{content_type}",
                "performance": {
                    "scene_changes": len(scene_changes),
                    "audio_moments": len(audio_moments),
                    "keyframes_selected": len(keyframes),
                    "quality_score": "adaptive-optimized"
                },
                "strategy": self._get_strategy_description(content_type)
            }
            
        except Exception as e:
            logger.error(f"❌ Adaptive analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "analysis": "",
                "transcription": "",
                "frames_analyzed": 0
            }

    def _generate_adaptive_prompt(self, content_type: str, transcription: str) -> str:
        """
        Genera prompt adattivo basato sul tipo di contenuto
        """
        base_prompt = f"""TRASCRIZIONE AUDIO COMPLETA:
{transcription}

TIPO DI CONTENUTO RILEVATO: {content_type}

"""
        
        if content_type == "MATH_MODE":
            return base_prompt + """Analizza questo contenuto matematico/scientifico:
1. Identifica formule, teoremi e concetti chiave
2. Correla le spiegazioni audio con le immagini (lavagne, grafici)
3. Evidenzia passaggi dimostrativi e esempi
4. Spiega la progressione logica del contenuto
5. Nota eventuali errori o imprecisioni"""
            
        elif content_type == "AUDIO_MODE":
            return base_prompt + """Analizza questo contenuto audio-centrico (podcast/intervista):
1. Identifica i punti chiave della discussione
2. Riassumi gli argomenti principali
3. Nota cambi di tono o enfasi
4. Evidenzia opinioni e fatti
5. Le immagini sono secondarie, focus sull'audio"""
            
        elif content_type == "PRESENTATION_MODE":
            return base_prompt + """Analizza questa presentazione business/professionale:
1. Correla slide e contenuto audio
2. Identifica dati, grafici e statistiche
3. Evidenzia messaggi chiave e conclusioni
4. Nota la struttura della presentazione
5. Valuta efficacia comunicativa"""
            
        elif content_type == "DOCUMENTARY_MODE":
            return base_prompt + """Analizza questo contenuto documentaristico/storico:
1. Identifica eventi, date e personaggi
2. Correla narrazione e immagini d'archivio
3. Evidenzia fatti storici e testimonianze
4. Nota la progressione cronologica
5. Valuta accuratezza storica"""
            
        elif content_type == "TUTORIAL_MODE":
            return base_prompt + """Analizza questo tutorial/contenuto educativo:
1. Identifica passi e procedure
2. Correla istruzioni audio e dimostrazioni visive
3. Evidenzia tecniche e metodi
4. Nota la progressione didattica
5. Valuta chiarezza esplicativa"""
        
        else:
            return base_prompt + """Fornisci un'analisi completa e bilanciata che includa:
1. Contenuto principale e tema
2. Correlazione audio-video
3. Messaggi chiave e conclusioni
4. Struttura e organizzazione
5. Qualità comunicativa"""

    def _get_strategy_description(self, content_type: str) -> str:
        """
        Descrizione della strategia utilizzata
        """
        strategies = {
            "MATH_MODE": "Visual-priority: Focus su formule e dimostrazioni",
            "AUDIO_MODE": "Audio-priority: Focus su contenuto parlato",
            "PRESENTATION_MODE": "Balanced: Slide + narrazione",
            "DOCUMENTARY_MODE": "Narrative: Eventi + immagini",
            "TUTORIAL_MODE": "Step-by-step: Istruzioni + demo",
            "BALANCED_MODE": "Balanced: Multi-modal standard"
        }
        return strategies.get(content_type, "Standard multi-modal")

    async def _visual_only_fallback(self, video_path: str, audio_result: Dict = None) -> Dict:
        """
        Fallback solo visivo
        """
        logger.info("🔄 Visual-only fallback mode...")
        
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frames_to_extract = [
                total_frames // 6,
                total_frames // 3,
                total_frames // 2,
                2 * total_frames // 3,
                5 * total_frames // 6
            ]
            
            content = [{
                "type": "text",
                "text": "Analizza questi frame del video e descrivi dettagliatamente il contenuto."
            }]
            
            for frame_num in frames_to_extract:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    base64_image = self.frame_to_base64(frame)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": base64_image, "detail": "high"}
                    })
            
            cap.release()
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                max_tokens=2000,
                temperature=0.3
            )
            
            return {
                "success": True,
                "analysis": response.choices[0].message.content,
                "transcription": audio_result.get("transcription", "") if audio_result else "",
                "content_type": "VISUAL_ONLY",
                "frames_analyzed": len(frames_to_extract),
                "architecture": "Visual-Only-Fallback",
                "performance": {"quality_score": "fallback"}
            }
            
        except Exception as e:
            logger.error(f"❌ Visual fallback failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "analysis": "Analisi fallita completamente",
                "transcription": "",
                "frames_analyzed": 0
            }
