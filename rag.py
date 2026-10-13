"""RAG per rispondere a domande libere sul documento, un indice per doc_id.

Embeddings locali con fastembed (ONNX, niente PyTorch) e ricerca per similarità coseno
in numpy: per documenti di poche pagine un vector DB sarebbe solo peso in più.
"""

import json
import os
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

STORAGE_DIR = Path(__file__).parent / "storage"
STORAGE_DIR.mkdir(exist_ok=True)

_CACHE_DIR = os.getenv("FASTEMBED_CACHE_PATH", str(Path(__file__).parent / ".fastembed_cache"))

_embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_dir=_CACHE_DIR)
_llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0.3)


def _embed(texts: list[str]) -> np.ndarray:
    vectors = np.array(list(_embedder.embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def index_text(doc_id: str, text: str) -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)

    doc_dir = STORAGE_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    np.save(doc_dir / "embeddings.npy", _embed(chunks))
    (doc_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")


def _retrieve(doc_id: str, question: str, k: int) -> list[str]:
    doc_dir = STORAGE_DIR / doc_id
    embeddings = np.load(doc_dir / "embeddings.npy")
    chunks = json.loads((doc_dir / "chunks.json").read_text(encoding="utf-8"))

    scores = embeddings @ _embed([question])[0]
    best = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in best]


def answer_question(doc_id: str, question: str, k: int = 4) -> str:
    context = "\n\n---\n\n".join(_retrieve(doc_id, question, k))

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Rispondi alla domanda usando esclusivamente il CONTESTO estratto dal documento. "
         "Se non trovi la risposta nel contesto, dillo chiaramente.\n\nCONTESTO:\n{context}"),
        ("human", "{question}"),
    ])
    chain = prompt | _llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})
