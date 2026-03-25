from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

QueryMode = Literal["text", "terms", "hybrid"]


@dataclass
class Query:
    mode: QueryMode
    text: Optional[str] = None
    terms: Optional[List[str]] = None
    meta: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        if self.mode == "text" and not self.text:
            raise ValueError("text mode requires 'text'.")
        if self.mode == "terms" and not self.terms:
            raise ValueError("terms mode requires 'terms'.")
        if self.mode == "hybrid" and not (self.text and self.terms):
            raise ValueError("hybrid mode requires both 'text' and 'terms'.")
