import json
from typing import Any, Dict, List

import faiss
import numpy as np
import pandas as pd


def titan_embed_query(text: str, client, model_id: str) -> np.ndarray:
    payload = {"inputText": text}
    resp = client.invoke_model(
        modelId=model_id,
        body=json.dumps(payload),
        contentType="application/json",
    )
    out = json.loads(resp["body"].read())
    vec = np.asarray(out["embedding"], dtype="float32")
    return vec / (np.linalg.norm(vec) + 1e-12)


def search_faiss(
    index_path: str,
    table_path: str,
    query: str,
    *,
    client,
    model_id: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    index = faiss.read_index(index_path)
    table = pd.read_parquet(table_path)
    qv = titan_embed_query(query, client=client, model_id=model_id).reshape(1, -1)
    distances, indices = index.search(qv, top_k)

    hits: List[Dict[str, Any]] = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        row = table.iloc[int(idx)].to_dict()
        row["distance"] = float(distance)
        row["score"] = float(1.0 / (1.0 + distance))
        hits.append(row)
    return hits
