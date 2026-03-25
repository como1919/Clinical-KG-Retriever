# rag-retriever

A lightweight **clinical retrieval system** built on a disease–symptom knowledge graph.

This project implements a structured retrieval pipeline based on a health knowledge graph learned from Electronic Medical Records (EMR), and extends it into a FAISS-based retrieval engine for clinical reasoning tasks.

---

## Background

This system is based on the following paper:

> **Learning a Health Knowledge Graph from Electronic Medical Records**  
> Rotmensch et al., Scientific Reports (2017)  
> https://www.nature.com/articles/s41598-017-05778-z

The study demonstrates that:

- Disease–symptom relationships can be automatically learned from EMR data  
- Knowledge graphs can be constructed using probabilistic models (Naive Bayes, Logistic Regression, Noisy OR)
- The resulting graph achieves high clinical quality validated against physician knowledge and Google's health graph  

---

## What this project does

While the original paper focuses on **knowledge graph construction**,  
this project extends it into a **retrieval system**:

- Converts knowledge graph → retrieval documents
- Builds FAISS-based vector index
- Supports:
  - Text query (clinical narrative)
  - Term-based query (symptom list)
  - Hybrid query (text + terms)
- Aggregates results at **disease level (not chunk level)**

---

## Retrieval Pipeline

The system is structured into four stages:

1. builder   : knowledge graph → documents
2. index     : documents → embeddings → FAISS
3. retrieve  : query → candidate documents
4. filter    : documents → structured disease cards

---

## Project Structure

src/rag_retriever/
  builder/
  index/
  retrieve/
  filter/
  pipeline/

---

## Mapping to Research Methodology

| Paper Step                         | This Implementation                  |
|----------------------------------|-------------------------------------|
| Concept extraction               | Preprocessed nodes/edges CSV        |
| Disease–symptom graph            | nodes.csv / edges.csv               |
| Importance scoring               | symptom weights in documents        |
| Graph → usable format            | docs.parquet                        |
| Inference                        | FAISS retrieval + filtering         |

---

## Quick Start

### 1. Build retrieval documents

python scripts/build_docs.py \
  --nodes_csv data/sample/nodes.csv \
  --edges_csv data/sample/edges.csv \
  --out_parquet data/artifacts/docs.parquet

---

### 2. Build FAISS index

python scripts/build_index.py \
  --docs_parquet data/artifacts/docs.parquet \
  --index_out data/artifacts/faiss_index.bin \
  --table_out data/artifacts/faiss_docs.parquet \
  --region us-west-2 \
  --embed_model_id amazon.titan-embed-text-v2:0

---

### 3. Run retrieval

python scripts/query.py \
  --mode hybrid \
  --text "right lower quadrant pain with nausea" \
  --auto_terms \
  --index_path data/artifacts/faiss_index.bin \
  --table_path data/artifacts/faiss_docs.parquet \
  --region us-west-2 \
  --embed_model_id amazon.titan-embed-text-v2:0

---

## Example Output

{
  "disease": "appendicitis",
  "symptoms": [
    ["pain", 0.88],
    ["nausea", 0.40],
    ["abdominal pain", 0.36]
  ]
}

---

## Key Features

- Knowledge graph–based retrieval (not plain text RAG)
- Disease-level aggregation
- Hybrid query support (text + medical terms)
- Clinically interpretable output
- Modular pipeline (builder / index / retrieve / filter)

---

## Notes

- This package includes **retrieval only**
- LLM generation and diagnosis prompting are intentionally excluded
- OpenSearch backend is not included in this public version
- FAISS index and parquet artifacts should not be committed

---

## Future Work

- Integration with LLM-based diagnosis (DDX pipeline)
- Knowledge graph expansion (UMLS / SNOMED)
- Improved symptom normalization (e.g., "pain" → location-aware)
- Clinical weighting refinement

---

## Summary

This project is not a generic RAG pipeline.

It is a **structured clinical retrieval system** built on a knowledge graph,  
designed to support downstream medical reasoning tasks such as differential diagnosis.
