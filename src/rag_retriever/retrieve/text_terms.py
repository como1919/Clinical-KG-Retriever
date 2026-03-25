import re
from typing import List, Tuple

_PUNCT_RE = re.compile(r"[^\w\s가-힣·\-/%]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_STOP_EN = set(
    """the a an of to and or for in on with without by from at as is are was were be been being
patient pt hx hpi pmh sh fh day days week weeks month months year years very much many
no not mild severe moderate normal abnormal within about""".split()
)
_STOP_NUM = re.compile(r"^\d+(\.\d+)?([a-zA-Z%/]+)?$")


def _clean_text(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _tokenize_ko_en(text: str) -> List[str]:
    tokens = [w for w in text.split(" ") if w]
    out = []
    for token in tokens:
        token = token.strip("-_/")
        if not token:
            continue
        if _STOP_NUM.match(token):
            continue
        if token in _STOP_EN:
            continue
        if len(token) < 2 or len(token) > 30:
            continue
        out.append(token)
    return out


def _make_ngrams(tokens: List[str], n: int) -> List[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def extract_terms_from_text(text: str, max_terms: int = 12) -> List[str]:
    cleaned = _clean_text(text)
    tokens = _tokenize_ko_en(cleaned)
    if not tokens:
        return []

    candidates: List[Tuple[str, float]] = []
    for n, weight in [(3, 1.0), (2, 0.9), (4, 0.8), (5, 0.7)]:
        for gram in _make_ngrams(tokens, n):
            if any(token in _STOP_EN for token in gram.split()):
                continue
            if any(len(tok) == 1 and re.match(r"[가-힣]", tok) for tok in gram.split()):
                continue
            candidates.append((gram, weight))

    scored = {}
    for gram, weight in candidates:
        score = weight * (1.0 + 0.02 * len(gram))
        scored[gram] = max(scored.get(gram, 0.0), score)

    items = sorted(scored.items(), key=lambda x: (len(x[0]), x[1]), reverse=True)
    uniq: List[str] = []
    seen = set()
    for gram, _ in items:
        if gram in seen:
            continue
        seen.add(gram)
        uniq.append(gram)
        if len(uniq) >= max_terms:
            break
    return uniq
