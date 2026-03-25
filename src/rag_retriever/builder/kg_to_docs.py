import numpy as np
import pandas as pd


def build_docs_from_kg(nodes_csv: str, edges_csv: str, out_parquet: str, topn: int = 50) -> pd.DataFrame:
    """Build retrieval-friendly disease documents from KG CSV exports."""
    nodes = pd.read_csv(nodes_csv, dtype=str).fillna("")
    edges = pd.read_csv(edges_csv, dtype=str).fillna("")

    id2name = nodes.set_index(":ID")["name:String"].to_dict()
    id2label = nodes.set_index(":ID")[":LABEL"].to_dict()
    id2icd = nodes.set_index(":ID")["icd:String"].to_dict() if "icd:String" in nodes.columns else {}

    if "importance:Double" in edges.columns:
        edges["importance:Double"] = pd.to_numeric(edges["importance:Double"], errors="coerce")
    else:
        edges["importance:Double"] = np.nan

    diseases = [nid for nid, lbl in id2label.items() if lbl == "Disease"]

    rows = []
    for disease_id in diseases:
        sub = edges[(edges[":TYPE"] == "CAUSES") & (edges[":START_ID"] == disease_id)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("importance:Double", ascending=False).head(topn)

        disease_name = id2name.get(disease_id, disease_id)
        icd = id2icd.get(disease_id, "")

        lines = [
            f"Disease: {disease_name}",
            f"ID: {disease_id}",
            f"ICD: {icd}",
            "",
            "Top symptoms (by importance):",
        ]
        for _, row in sub.iterrows():
            symptom_name = id2name.get(row[":END_ID"], row[":END_ID"])
            weight = "" if pd.isna(row["importance:Double"]) else f"{row['importance:Double']:.3f}"
            lines.append(f"- {symptom_name} (importance={weight})")

        rows.append(
            {
                "doc_id": disease_id,
                "disease_name": disease_name,
                "title": disease_name,
                "icd": icd,
                "text": "\n".join(lines),
            }
        )

    docs = pd.DataFrame(rows)
    docs.to_parquet(out_parquet, index=False)
    return docs
