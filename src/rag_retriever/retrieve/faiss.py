from typing import Any, Dict, List, Literal, Tuple
import os

from .base import BaseRetriever
from .faiss_backend import search_faiss


def _normalize_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normed: List[Dict[str, Any]] = []
    for i, hit in enumerate(hits):
        doc_id = hit.get("doc_id", hit.get("id", i))
        score = float(hit.get("score", 0.0))
        text = hit.get("text")
        title = hit.get("title", hit.get("disease_name"))
        meta = {k: v for k, v in hit.items() if k not in ("doc_id", "id", "score", "text", "title")}
        normed.append({"doc_id": doc_id, "score": score, "text": text, "title": title, "meta": meta})
    return normed


def _merge(a: List[Dict[str, Any]], b: List[Dict[str, Any]], topk: int, w_a: float = 0.6, w_b: float = 0.4) -> List[Dict[str, Any]]:
    pool: Dict[Any, Dict[str, Any]] = {}

    def update(items: List[Dict[str, Any]], weight: float) -> None:
        for item in items:
            did = item["doc_id"]
            score = item.get("score", 0.0) * weight
            if did not in pool:
                pool[did] = dict(item)
                pool[did]["score"] = score
            else:
                pool[did]["score"] += score

    update(a, w_a)
    update(b, w_b)
    return sorted(pool.values(), key=lambda x: x["score"], reverse=True)[:topk]


class FaissRetriever(BaseRetriever):
    def __init__(self, index_path: str, table_path: str, *, client, model_id: str):
        if not os.path.exists(index_path) or not os.path.exists(table_path):
            raise FileNotFoundError(f"FAISS index/table not found: {index_path}, {table_path}")
        self.index_path = index_path
        self.table_path = table_path
        self.client = client
        self.model_id = model_id

    def search_text(self, text: str, topk: int = 20) -> List[Dict[str, Any]]:
        hits = search_faiss(
            self.index_path,
            self.table_path,
            text,
            client=self.client,
            model_id=self.model_id,
            top_k=topk,
        )
        return _normalize_hits(hits)

    def search_terms(
        self,
        terms: List[str],
        topk: int = 20,
        per_term_k: int = 10,
        agg: Literal["max", "mean"] = "max",
    ) -> List[Dict[str, Any]]:
        if not terms:
            return []
        buckets: Dict[Any, List[float]] = {}
        cache: Dict[Any, Dict[str, Any]] = {}
        for term in terms:
            if not term.strip():
                continue
            term_hits = _normalize_hits(
                search_faiss(
                    self.index_path,
                    self.table_path,
                    term,
                    client=self.client,
                    model_id=self.model_id,
                    top_k=per_term_k,
                )
            )
            for hit in term_hits:
                did = hit["doc_id"]
                buckets.setdefault(did, []).append(float(hit.get("score", 0.0)))
                if did not in cache:
                    cache[did] = hit

        scored: List[Tuple[Any, float]] = []
        for did, scores in buckets.items():
            if agg == "mean":
                score = sum(scores) / len(scores)
            else:
                score = max(scores)
            score += 0.01 * (len(scores) - 1)
            scored.append((did, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        out: List[Dict[str, Any]] = []
        for did, score in scored[:topk]:
            item = dict(cache[did])
            item["score"] = float(score)
            out.append(item)
        return out

    def search(
        self,
        query,
        mode: Literal["text", "terms", "hybrid"] = "text",
        topk: int = 20,
        per_term_k: int = 10,
        agg: Literal["max", "mean"] = "max",
        w_text: float = 0.6,
        w_terms: float = 0.4,
    ) -> List[Dict[str, Any]]:
        if mode == "text":
            assert isinstance(query, str)
            return self.search_text(query, topk=topk)
        if mode == "terms":
            assert isinstance(query, list)
            return self.search_terms(query, topk=topk, per_term_k=per_term_k, agg=agg)
        if mode == "hybrid":
            assert isinstance(query, dict) and "text" in query and "terms" in query
            text_hits = self.search_text(query["text"], topk=topk)
            term_hits = self.search_terms(query["terms"], topk=topk, per_term_k=per_term_k, agg=agg)
            return _merge(text_hits, term_hits, topk=topk, w_a=w_text, w_b=w_terms)
        raise ValueError("mode must be one of: text | terms | hybrid")
