#!/usr/bin/env python3
import argparse
from pathlib import Path

from rag_retriever.builder import build_docs_from_kg


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval docs from KG CSV files.")
    parser.add_argument("--nodes_csv", required=True)
    parser.add_argument("--edges_csv", required=True)
    parser.add_argument("--out_parquet", required=True)
    parser.add_argument("--topn", type=int, default=50)
    args = parser.parse_args()

    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    docs = build_docs_from_kg(args.nodes_csv, args.edges_csv, args.out_parquet, topn=args.topn)
    print(f"Saved {len(docs)} documents to {args.out_parquet}")


if __name__ == "__main__":
    main()
