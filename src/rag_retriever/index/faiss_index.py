import json
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm


def titan_embed_one(text: str, client, model_id: str) -> np.ndarray:
    payload = {"inputText": text}
    resp = client.invoke_model(
        modelId=model_id,
        body=json.dumps(payload),
        contentType="application/json",
    )
    out = json.loads(resp["body"].read())
    return np.asarray(out["embedding"], dtype="float32")


def build_faiss_index(
    docs_parquet: str,
    index_out: str,
    table_out: str,
    *,
    client,
    model_id: str,
    text_col: str = "text",
    normalize: bool = True,
    batch_size: Optional[int] = None,
) -> None:
    """Embed documents and build a FAISS flat index."""
    docs = pd.read_parquet(docs_parquet)
    if text_col not in docs.columns:
        raise ValueError(f"Column '{text_col}' not found in {docs_parquet}.")

    vecs = []
    iterator = docs[text_col].tolist()
    if batch_size is not None and batch_size <= 0:
        batch_size = None

    for text in tqdm(iterator, desc="Embedding documents"):
        vecs.append(titan_embed_one(text, client=client, model_id=model_id))

    embs = np.vstack(vecs).astype("float32")
    if normalize:
        faiss.normalize_L2(embs)

    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, index_out)
    docs.to_parquet(table_out, index=False)
