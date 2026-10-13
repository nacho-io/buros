"""
Pipeline LangGraph a 4 nodi. Il routing dopo la classificazione seleziona il
prompt specifico per tipo di documento (contratto/multa/bolletta hanno red flag
completamente diverse), ma tutti i nodi condividono lo stesso stato.
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from schemas import DocumentType, Classification, ExtractedInfo, RedFlagList, Explanation, RedFlag

_llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)

# Prompt aggiuntivo specifico per tipo, usato nel nodo di analisi red-flag
_REDFLAG_HINTS = {
    DocumentType.CONTRATTO: (
        "Cerca in particolare: clausole di rinnovo automatico poco visibili, penali sproporzionate, "
        "clausole vessatorie, condizioni di recesso poco chiare o sfavorevoli, costi nascosti."
    ),
    DocumentType.MULTA: (
        "Cerca in particolare: possibili vizi di forma (dati mancanti, notifica fuori termine), "
        "termini per il ricorso, sconti per pagamento entro breve termine."
    ),
    DocumentType.BOLLETTA: (
        "Cerca in particolare: aumenti anomali rispetto a bollette precedenti (se menzionati), "
        "voci di costo poco chiare, cambi di fornitore/tariffa non richiesti esplicitamente."
    ),
    DocumentType.PEC: (
        "Cerca in particolare: termini di risposta stretti, richieste di conferma implicite, "
        "conseguenze del silenzio/non risposta."
    ),
    DocumentType.ALTRO: "Segnala qualunque elemento che richieda attenzione o azione da parte del destinatario.",
}


class PipelineState(TypedDict):
    text: str
    doc_type: DocumentType
    extracted: ExtractedInfo
    red_flags: List[RedFlag]
    spiegazione_semplice: str
    checklist: List[str]


def classify_node(state: PipelineState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Classifica il tipo di documento tra: bolletta, contratto, multa, pec, altro."),
        ("human", "{text}"),
    ])
    structured_llm = _llm.with_structured_output(Classification)
    result = (prompt | structured_llm).invoke({"text": state["text"][:4000]})
    return {"doc_type": result.tipo}


def extract_node(state: PipelineState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Estrai le informazioni chiave dal documento seguente (tipo: {doc_type}). "
         "Se un'informazione non è presente, lasciala vuota invece di inventarla."),
        ("human", "{text}"),
    ])
    structured_llm = _llm.with_structured_output(ExtractedInfo)
    result = (prompt | structured_llm).invoke({"text": state["text"], "doc_type": state["doc_type"].value})
    return {"extracted": result}


def redflag_node(state: PipelineState) -> dict:
    hint = _REDFLAG_HINTS[state["doc_type"]]
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Analizza il documento e individua eventuali elementi che l'utente dovrebbe notare con attenzione. "
         f"{hint} "
         "Se non trovi nulla di problematico, restituisci una lista vuota: non inventare problemi che non ci sono."),
        ("human", "{text}"),
    ])
    structured_llm = _llm.with_structured_output(RedFlagList)
    result = (prompt | structured_llm).invoke({"text": state["text"]})
    return {"red_flags": result.red_flags}


def explain_node(state: PipelineState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Sulla base delle informazioni estratte e degli elementi critici individuati, scrivi: "
         "1) una spiegazione in linguaggio semplice e diretto (niente legalese, come se parlassi a un amico) "
         "2) una checklist ordinata di azioni concrete da fare, con eventuali scadenze in evidenza."),
        ("human",
         "Info estratte: {extracted}\n\nElementi critici: {red_flags}\n\nTesto originale (per contesto): {text}"),
    ])
    structured_llm = _llm.with_structured_output(Explanation)
    result = (prompt | structured_llm).invoke({
        "extracted": state["extracted"].model_dump(),
        "red_flags": [rf.model_dump() for rf in state["red_flags"]],
        "text": state["text"][:3000],
    })
    return {"spiegazione_semplice": result.spiegazione_semplice, "checklist": result.checklist}


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("classify", classify_node)
    graph.add_node("extract", extract_node)
    graph.add_node("redflag", redflag_node)
    graph.add_node("explain", explain_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "extract")
    graph.add_edge("extract", "redflag")
    graph.add_edge("redflag", "explain")
    graph.add_edge("explain", END)

    return graph.compile()


_compiled_graph = build_graph()


def run_pipeline(text: str) -> PipelineState:
    return _compiled_graph.invoke({"text": text})
