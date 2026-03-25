from typing import Any, Dict, List, Literal, Optional
import os

from ..builder.kg_to_docs import build_docs_from_kg
from ..filter.disease_cards import filter_disease_cards
from ..index.faiss_index import build_faiss_index
from ..retrieve.faiss import FaissRetriever
from ..schema import Query


def ensure_faiss_artifacts(
    *,
    nodes_csv: str,
    edges_csv: str,
    docs_path: str,
    index_path: str,
    table_path: str,
    client,
    model_id: str,
    topn: int = 50,
) -> None:
    if not os.path.exists(docs_path):
        build_docs_from_kg(nodes_csv, edges_csv, docs_path, topn=topn)
    if not os.path.exists(index_path) or not os.path.exists(table_path):
        build_faiss_index(docs_path, index_path, table_path, client=client, model_id=model_id)


def _standardize_cards(cards: List[Dict[str, Any]], filter_topk: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for card in cards[:filter_topk]:
        data = dict(card)
        title = data.get("title") or data.get("disease") or "(no title)"
        data["title"] = title
        data.setdefault("text", "")
        out.append(data)
    return out


def run_retrieval_pipeline(
    *,
    query: Query,
    index_path: str,
    table_path: str,
    client,
    model_id: str,
    topk: int = 20,
    filter_topk: int = 8,
    top_n_diseases: Optional[int] = 3,
    top_k_per_disease: Optional[int] = 3,
    symptom_score_threshold: Optional[float] = None,
    disease_rank_by: Literal["sum", "max", "count"] = "sum",
    min_symptoms_per_disease: int = 1,
    per_term_k: int = 10,
    agg: Literal["max", "mean"] = "max",
    w_text: float = 0.6,
    w_terms: float = 0.4,
) -> List[Dict[str, Any]]:
    query.validate()
    retriever = FaissRetriever(index_path=index_path, table_path=table_path, client=client, model_id=model_id)

    if query.mode == "text":
        hits = retriever.search_text(query.text or "", topk=topk)
    elif query.mode == "terms":
        hits = retriever.search_terms(query.terms or [], topk=topk, per_term_k=per_term_k, agg=agg)
    else:
        hits = retriever.search(
            {"text": query.text or "", "terms": query.terms or []},
            mode="hybrid",
            topk=topk,
            per_term_k=per_term_k,
            agg=agg,
            w_text=w_text,
            w_terms=w_terms,
        )

    cards = filter_disease_cards(
        hits,
        top_n_diseases=top_n_diseases,
        top_k_per_disease=top_k_per_disease,
        symptom_score_threshold=symptom_score_threshold,
        disease_rank_by=disease_rank_by,
        min_symptoms_per_disease=min_symptoms_per_disease,
    )
    return _standardize_cards(cards, filter_topk=filter_topk)
