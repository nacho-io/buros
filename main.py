"""
API FastAPI per l'assistente documenti burocratici.

Endpoint:
  POST /api/documents/upload      -> carica un file, esegue la pipeline, ritorna l'analisi
  GET  /api/documents/{doc_id}    -> rilegge un'analisi già fatta
  POST /api/documents/{doc_id}/ask -> domanda libera (RAG) sul documento

Avvio: uvicorn main:app --reload

Il frontend (buros-fe) è servito separatamente: in locale dal dev server di Vite,
in produzione da un container nginx che fa da reverse proxy verso questa API.
"""

import os
import uuid
import shutil
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import Depends, FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import extraction
import rag
import storage
from auth import get_current_user
from graph import run_pipeline
from schemas import AnalysisResult

app = FastAPI(title="Assistente Documenti Burocratici")

# In locale il frontend gira su un'origine diversa (Vite su :5173) dal backend (:8000).
# In produzione, dietro il reverse proxy nginx, FE e BE condividono la stessa origine
# e il CORS non serve — ma resta configurabile via env var per altri scenari di deploy.
_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
allowed_origins = os.getenv("ALLOWED_ORIGINS", _default_origins).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).parent / "storage"
UPLOAD_DIR.mkdir(exist_ok=True)

SUPPORTED_EXT = {".pdf", ".txt", ".docx", ".jpg", ".jpeg", ".png", ".webp"}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise HTTPException(400, f"Formato non supportato: {ext}")

    doc_id = str(uuid.uuid4())
    doc_dir = UPLOAD_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    local_path = doc_dir / file.filename

    with open(local_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        text = extraction.extract_text(str(local_path))
        if not text.strip():
            raise HTTPException(422, "Non è stato possibile estrarre testo dal documento.")

        pipeline_result = run_pipeline(text)

        analysis = AnalysisResult(
            doc_id=doc_id,
            tipo=pipeline_result["doc_type"],
            extracted=pipeline_result["extracted"],
            red_flags=pipeline_result["red_flags"],
            spiegazione_semplice=pipeline_result["spiegazione_semplice"],
            checklist=pipeline_result["checklist"],
        )

        storage.save_analysis(doc_id, analysis.model_dump(), user["id"])
        rag.index_text(doc_id, text)  # indicizza per il Q&A libero successivo

        return analysis.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Errore durante l'analisi: {e}")


@app.get("/api/documents/{doc_id}")
async def get_analysis(doc_id: str, user: dict = Depends(get_current_user)):
    analysis = storage.load_analysis(doc_id)
    if analysis is None or analysis.get("user_id") != user["id"]:
        raise HTTPException(404, "Documento non trovato.")
    analysis.pop("user_id", None)
    return analysis


class QuestionRequest(BaseModel):
    question: str


@app.post("/api/documents/{doc_id}/ask")
async def ask_question(doc_id: str, req: QuestionRequest, user: dict = Depends(get_current_user)):
    analysis = storage.load_analysis(doc_id)
    if analysis is None or analysis.get("user_id") != user["id"]:
        raise HTTPException(404, "Documento non trovato.")
    answer = rag.answer_question(doc_id, req.question)
    return {"answer": answer}
