from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal


class BaseRetriever(ABC):
    @abstractmethod
    def search_text(self, text: str, topk: int = 20) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def search_terms(
        self,
        terms: List[str],
        topk: int = 20,
        per_term_k: int = 10,
        agg: Literal["max", "mean"] = "max",
    ) -> List[Dict[str, Any]]:
        ...
