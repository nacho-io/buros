"""Schemi Pydantic usati per forzare output strutturato dal modello nei vari nodi della pipeline."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    BOLLETTA = "bolletta"
    CONTRATTO = "contratto"
    MULTA = "multa"
    PEC = "pec"
    ALTRO = "altro"


class Classification(BaseModel):
    tipo: DocumentType = Field(description="Tipo di documento individuato")
    confidenza: str = Field(description="alta / media / bassa")


class ExtractedInfo(BaseModel):
    mittente: Optional[str] = Field(None, description="Chi ha emesso il documento (azienda, ente, controparte)")
    destinatario: Optional[str] = Field(None, description="A chi è indirizzato, se specificato")
    importo: Optional[str] = Field(None, description="Importo economico rilevante, se presente (con valuta)")
    scadenza: Optional[str] = Field(None, description="Data entro cui è richiesta un'azione, in formato leggibile")
    azione_richiesta: Optional[str] = Field(None, description="Cosa deve fare il destinatario (pagare, rispondere, fare ricorso, nulla...)")
    riferimenti_normativi: List[str] = Field(default_factory=list, description="Leggi, articoli o normative citate nel documento")
    riepilogo: str = Field(description="Riassunto in 2-3 frasi del contenuto del documento")


class RedFlag(BaseModel):
    descrizione: str = Field(description="Cosa è stato notato e perché potrebbe essere un problema")
    gravita: str = Field(description="bassa / media / alta")


class RedFlagList(BaseModel):
    red_flags: List[RedFlag] = Field(default_factory=list)


class Explanation(BaseModel):
    spiegazione_semplice: str = Field(description="Spiegazione del documento in linguaggio semplice, senza legalese")
    checklist: List[str] = Field(description="Lista di azioni concrete e ordinate che l'utente dovrebbe fare")


class AnalysisResult(BaseModel):
    doc_id: str
    tipo: DocumentType
    extracted: ExtractedInfo
    red_flags: List[RedFlag]
    spiegazione_semplice: str
    checklist: List[str]
    disclaimer: str = (
        "Questa analisi è generata automaticamente e ha scopo puramente informativo: "
        "non sostituisce una consulenza legale, fiscale o professionale. "
        "Per decisioni importanti rivolgiti a un avvocato, commercialista o ente competente."
    )
