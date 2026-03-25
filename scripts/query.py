#!/usr/bin/env python3
import argparse
import json

import boto3

from rag_retriever.pipeline import run_retrieval_pipeline
from rag_retriever.retrieve import extract_terms_from_text
from rag_retriever.schema import Query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval against a FAISS-backed disease-card index.")
    parser.add_argument("--mode", choices=["text", "terms", "hybrid"], default="text")
    parser.add_argument("--text", default=None)
    parser.add_argument("--terms", nargs="*", default=None)
    parser.add_argument("--auto_terms", action="store_true")
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--table_path", required=True)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--embed_model_id", default="amazon.titan-embed-text-v2:0")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--filter_topk", type=int, default=8)
    parser.add_argument("--topn_diseases", type=int, default=3)
    parser.add_argument("--topk_symptoms", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def build_query(args: argparse.Namespace) -> Query:
    terms = args.terms or []
    text = args.text
    if args.mode == "hybrid" and (args.auto_terms or not terms):
        if not text:
            raise SystemExit("hybrid mode requires --text")
        auto_terms = extract_terms_from_text(text, max_terms=12)
        terms = sorted({*terms, *auto_terms})
    return Query(mode=args.mode, text=text, terms=terms or None)


def main() -> None:
    args = parse_args()
    query = build_query(args)
    client = boto3.client("bedrock-runtime", region_name=args.region)
    cards = run_retrieval_pipeline(
        query=query,
        index_path=args.index_path,
        table_path=args.table_path,
        client=client,
        model_id=args.embed_model_id,
        topk=args.topk,
        filter_topk=args.filter_topk,
        top_n_diseases=None if args.topn_diseases == 0 else args.topn_diseases,
        top_k_per_disease=None if args.topk_symptoms == 0 else args.topk_symptoms,
    )
    if args.json:
        print(json.dumps(cards, ensure_ascii=False, indent=2))
    else:
        for idx, card in enumerate(cards, 1):
            print(f"[{idx}] {card['title']}")
            for name, score in card.get("symptoms", []):
                suffix = f" ({score:.2f})" if score is not None else ""
                print(f"  - {name}{suffix}")


if __name__ == "__main__":
    main()
