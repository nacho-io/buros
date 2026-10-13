# Assistente Documenti Burocratici

App web che analizza bollette, contratti, multe e PEC e restituisce:
- un riepilogo in linguaggio semplice
- le informazioni chiave (scadenze, importi, azione richiesta)
- eventuali elementi critici da notare ("red flag"), specifici per tipo di documento
- una checklist di cosa fare
- un Q&A libero sul documento (RAG)

## Architettura

Pipeline **LangGraph** a 4 nodi (`app/graph.py`):

```
classify → extract → redflag → explain
```

- **classify**: individua il tipo di documento (bolletta/contratto/multa/PEC/altro)
- **extract**: estrae dati strutturati (Pydantic) — mittente, importo, scadenza, azione richiesta...
- **redflag**: analizza il documento con un prompt specifico per il tipo individuato (le red flag di un contratto sono diverse da quelle di una multa)
- **explain**: genera la spiegazione in linguaggio semplice e la checklist finale

Il routing condizionale (prompt diverso in base al tipo di documento) è il motivo per cui qui LangGraph è più adatto di una semplice catena lineare LangChain.

Per le **foto di documenti**, il testo viene estratto direttamente con la vision del modello (`app/extraction.py`), senza bisogno di un motore OCR locale.

Il **Q&A libero** (`rag.py`) è un RAG minimale: embeddings locali e gratuite con `fastembed` (ONNX, niente PyTorch), un piccolo indice per documento (`storage/<doc_id>/embeddings.npy` + `chunks.json`) e ricerca per similarità coseno in numpy. Per documenti di poche pagine non serve un vector DB.

## Setup

```bash
cd document_assistant
python -m venv venv
source venv/bin/activate  # su Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# inserisci la tua ANTHROPIC_API_KEY in .env
```

## Avvio

```bash
cd app
uvicorn main:app --reload
```

Apri `http://localhost:8000` nel browser: carica un documento (drag&drop o click) e dopo qualche secondo vedrai l'analisi completa, con la possibilità di fare domande libere in fondo alla pagina.

## Struttura del progetto

```
document_assistant/
├── app/
│   ├── main.py         # API FastAPI
│   ├── graph.py         # pipeline LangGraph
│   ├── schemas.py        # modelli Pydantic (output strutturato)
│   ├── extraction.py    # estrazione testo (file + immagini via vision)
│   ├── rag.py            # Q&A libero sul documento
│   └── storage.py        # storage JSON minimale
├── static/
│   └── index.html        # frontend vanilla JS
├── storage/               # documenti caricati, indici vettoriali, analisi (creato a runtime)
├── requirements.txt
└── .env.example
```

## Limiti di questo scheletro (da sapere prima di usarlo con dati reali)

- **Storage**: JSON su disco, va bene per prototipare ma non per produzione — sostituire con un vero DB (Postgres) se il progetto cresce.
- **Nessuna autenticazione**: chiunque abbia accesso all'app può caricare/leggere documenti. Da aggiungere prima di esporlo pubblicamente, specialmente trattandosi di documenti potenzialmente sensibili (dati fiscali, contratti).
- **Nessuna cancellazione automatica**: i documenti restano su disco indefinitamente. Per dati sensibili, aggiungi una policy di retention/cancellazione.
- **Un solo LLM call per nodo, senza retry**: in produzione andrebbero aggiunti retry e validazione più robusta in caso di output malformato.

## Possibili estensioni

- **Notifiche scadenze**: integrare un reminder (email/calendar) quando viene rilevata una scadenza vicina.
- **Storico documenti per utente**: con autenticazione, mostrare tutti i documenti caricati da un utente nel tempo.
- **Confronto bollette**: se l'utente carica più bollette dello stesso fornitore, confrontarle per rilevare aumenti anomali reali (non solo dedotti dal singolo documento).
- **Export PDF del riepilogo**, utile da allegare o stampare.
