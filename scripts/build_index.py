#!/usr/bin/env python3
import argparse
from pathlib import Path

import boto3

from rag_retriever.index import build_faiss_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from docs parquet.")
    parser.add_argument("--docs_parquet", required=True)
    parser.add_argument("--index_out", required=True)
    parser.add_argument("--table_out", required=True)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--embed_model_id", default="amazon.titan-embed-text-v2:0")
    args = parser.parse_args()

    Path(args.index_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.table_out).parent.mkdir(parents=True, exist_ok=True)

    client = boto3.client("bedrock-runtime", region_name=args.region)
    build_faiss_index(
        args.docs_parquet,
        args.index_out,
        args.table_out,
        client=client,
        model_id=args.embed_model_id,
    )
    print(f"Saved index to {args.index_out}")
    print(f"Saved table to {args.table_out}")


if __name__ == "__main__":
    main()
