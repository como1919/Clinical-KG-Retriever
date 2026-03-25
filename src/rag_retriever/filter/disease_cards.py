from typing import Any, Dict, List, Literal, Optional, Tuple
import re

_NUM_RE = r"[0-9]+(?:\.[0-9]+)?"
_HDR_RE = re.compile(r"^\s*#\s*(.+?)\s*$")
META_KEYS = ("disease", "icd", "id", "code", "uri", "type", "label", "name")
_META_RE = re.compile(rf"^\s*(?:[•\-\–—]\s*)?(?:{'|'.join(META_KEYS)})\s*[:=]\s*", re.IGNORECASE)
_TOPSYM_RE = re.compile(r"^\s*(?:[•\-\–—]\s*)?top\s+symptoms\b", re.IGNORECASE)


def _parse_bullet_symptom(line: str) -> Tuple[str, Optional[float]]:
    text = line.strip().lstrip("•-–— ").strip()
    if not text or _META_RE.match(text) or _TOPSYM_RE.match(text):
        return "", None

    match = re.match(rf"^(.*?)\s*\([^)]*?({_NUM_RE})\)\s*$", text)
    if match:
        try:
            return match.group(1).strip(), float(match.group(2))
        except Exception:
            return match.group(1).strip(), None

    match = re.match(rf"^(.*?)[\s:]+({_NUM_RE})\s*$", text)
    if match:
        return match.group(1).strip(), float(match.group(2))

    return text, None


def _extract_triples_from_text(text: str) -> List[Tuple[str, str, Optional[float]]]:
    out: List[Tuple[str, str, Optional[float]]] = []
    if not text:
        return out
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        symptom, relation, disease, score_text = parts
        relation = relation.replace(" ", "")
        if relation not in ("has_symptom", "has_sign_or_symptom"):
            continue
        try:
            score = float(score_text)
        except Exception:
            score = None
        if disease and symptom:
            out.append((disease, symptom, score))
    return out


def _group_from_headers(text: str) -> Dict[str, Dict[str, Optional[float]]]:
    buckets: Dict[str, Dict[str, Optional[float]]] = {}
    if not text:
        return buckets

    current_disease: Optional[str] = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _HDR_RE.match(line)
        if match:
            current_disease = match.group(1).strip()
            buckets.setdefault(current_disease, {})
            continue

        if current_disease and (line.startswith(("•", "-", "–", "—")) or "(" in line or ":" in line):
            name, score = _parse_bullet_symptom(line)
            if name:
                prev = buckets[current_disease].get(name)
                if prev is None or (score is not None and (prev is None or score > prev)):
                    buckets[current_disease][name] = score
            continue

        if current_disease:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                symptom, relation, disease, score_text = parts
                relation = relation.replace(" ", "")
                if relation in ("has_symptom", "has_sign_or_symptom") and symptom:
                    target = disease or current_disease
                    try:
                        score = float(score_text)
                    except Exception:
                        score = None
                    if target:
                        prev = buckets.setdefault(target, {}).get(symptom)
                        if prev is None or (score is not None and (prev is None or score > prev)):
                            buckets[target][symptom] = score
    return buckets


def _extract_bullets(text: str) -> List[Tuple[str, Optional[float]]]:
    out: List[Tuple[str, Optional[float]]] = []
    if not text:
        return out
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(("•", "-", "–", "—")) or "(" in stripped or ":" in stripped:
            name, score = _parse_bullet_symptom(stripped)
            if name:
                out.append((name, score))
    return out


def _coerce_symptoms_from_text(text: str) -> List[Tuple[str, Optional[float]]]:
    out = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if _META_RE.match(stripped) or _TOPSYM_RE.match(stripped):
            continue
        if stripped.startswith(("•", "-", "–", "—")) or re.search(rf"\(\s*{_NUM_RE}\s*\)\s*$", stripped):
            name, score = _parse_bullet_symptom(stripped)
            if name:
                out.append((name, score))
    return out


def _guess_title_from_kv(text: str) -> Optional[str]:
    if not text:
        return None
    for line in text.splitlines():
        match = re.match(r"^\s*disease\s*[:=]\s*(.+?)\s*$", line.strip(), flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def filter_disease_cards(
    hits: List[Dict[str, Any]],
    top_n_diseases: Optional[int] = 3,
    top_k_per_disease: Optional[int] = 3,
    symptom_score_threshold: Optional[float] = None,
    disease_rank_by: Literal["sum", "max", "count"] = "sum",
    min_symptoms_per_disease: int = 1,
) -> List[Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Optional[float]]] = {}

    for hit in hits:
        grouped = _group_from_headers(hit.get("text", "") or "")
        for disease, symptom_map in grouped.items():
            agg = buckets.setdefault(disease, {})
            for name, score in symptom_map.items():
                prev = agg.get(name)
                if prev is None or (score is not None and (prev is None or score > prev)):
                    agg[name] = score

    if not buckets:
        for hit in hits:
            for disease, symptom, score in _extract_triples_from_text(hit.get("text", "") or ""):
                agg = buckets.setdefault(disease.strip(), {})
                prev = agg.get(symptom.strip())
                if prev is None or (score is not None and (prev is None or score > prev)):
                    agg[symptom.strip()] = score

    if not buckets:
        for hit in hits:
            title = (hit.get("title") or hit.get("meta", {}).get("disease") or "").strip()
            if not title:
                first = ((hit.get("text") or "").splitlines() or [""])[0]
                match = _HDR_RE.match(first or "")
                if match:
                    title = match.group(1).strip()
            if not title:
                continue
            for line in (hit.get("text", "") or "").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith(("•", "-", "–", "—")) or "(" in stripped or ":" in stripped:
                    name, score = _parse_bullet_symptom(stripped)
                    if name:
                        agg = buckets.setdefault(title, {})
                        prev = agg.get(name)
                        if prev is None or (score is not None and (prev is None or score > prev)):
                            agg[name] = score

    if not buckets:
        for i, hit in enumerate(hits, 1):
            text = hit.get("text", "") or ""
            title = (
                (hit.get("meta", {}) or {}).get("disease")
                or _guess_title_from_kv(text)
                or (hit.get("title") or "").strip()
                or str(hit.get("doc_id") or "").strip()
                or f"Candidate {i}"
            )
            symptom_pairs = _extract_bullets(text) or _coerce_symptoms_from_text(text)
            if not symptom_pairs:
                continue
            agg = buckets.setdefault(title, {})
            for name, score in symptom_pairs:
                prev = agg.get(name)
                if prev is None or (score is not None and (prev is None or score > prev)):
                    agg[name] = score

    if not buckets:
        return []

    if symptom_score_threshold is not None:
        buckets = {
            disease: {name: score for name, score in symptoms.items() if score is not None and score >= symptom_score_threshold}
            for disease, symptoms in buckets.items()
        }

    buckets = {disease: symptoms for disease, symptoms in buckets.items() if len(symptoms) >= max(0, min_symptoms_per_disease)}
    if not buckets:
        return []

    def score_sum(symptoms: Dict[str, Optional[float]]) -> float:
        return sum((score or 0.0) for score in symptoms.values())

    def score_max(symptoms: Dict[str, Optional[float]]) -> float:
        return max(((score or 0.0) for score in symptoms.values()), default=0.0)

    def score_count(symptoms: Dict[str, Optional[float]]) -> int:
        return len(symptoms)

    if disease_rank_by == "max":
        key_fn = lambda kv: (score_max(kv[1]), len(kv[1]))
    elif disease_rank_by == "count":
        key_fn = lambda kv: (score_count(kv[1]), score_sum(kv[1]))
    else:
        key_fn = lambda kv: (score_sum(kv[1]), len(kv[1]))

    ranked = sorted(buckets.items(), key=key_fn, reverse=True)
    if top_n_diseases and top_n_diseases > 0:
        ranked = ranked[:top_n_diseases]

    out: List[Dict[str, Any]] = []
    for disease, symptom_map in ranked:
        pairs = list(symptom_map.items())
        pairs.sort(key=lambda x: (x[1] is None, -(x[1] or 0.0), x[0]))
        if top_k_per_disease and top_k_per_disease > 0:
            pairs = pairs[:top_k_per_disease]
        out.append({"disease": disease, "title": disease, "symptoms": pairs, "text": ""})
    return out
