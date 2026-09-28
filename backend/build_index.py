"""
Builds the semantic search index + knowledge graph from the standards corpus.

Pipeline:
  1. Load standards (sample + any scraped data) into one corpus.
  2. Embed each standard's "title + scope" using a multilingual sentence
     embedding model, so Hindi/regional-language queries also match English
     standard text without a separate translation step.
  3. Store vectors in a FAISS index for fast nearest-neighbour search.
  4. Build a directed knowledge graph (networkx) of cross-reference edges:
     normative, test_method, related_product, terminology, safety,
     installation — so that once a primary standard is matched, allied
     standards can be pulled by graph traversal instead of a second search.

Run:  python build_index.py
Produces: index/faiss.index, index/id_map.json, index/graph.gpickle,
          index/corpus.json
"""

import json
import os
import pickle

import networkx as nx
import faiss

import embeddings

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "index")
os.makedirs(INDEX_DIR, exist_ok=True)


def load_corpus():
    corpus = []
    sample_path = os.path.join(DATA_DIR, "sample_standards.json")
    if os.path.exists(sample_path):
        with open(sample_path, encoding="utf-8") as f:
            corpus.extend(json.load(f))

    scraped_path = os.path.join(DATA_DIR, "scraped_standards.json")
    if os.path.exists(scraped_path):
        with open(scraped_path, encoding="utf-8") as f:
            corpus.extend(json.load(f))

    # de-duplicate by is_number, sample data taking priority (richer fields)
    seen = {}
    for rec in corpus:
        seen.setdefault(rec["is_number"], rec)
    return list(seen.values())


def build_graph(corpus):
    g = nx.DiGraph()
    for rec in corpus:
        g.add_node(rec["is_number"], title=rec.get("title", ""),
                   certification=rec.get("certification", "None"),
                   status=rec.get("status", "active"),
                   latest_amendment=rec.get("latest_amendment", "None"))
        for ref in rec.get("cross_references", []):
            g.add_edge(rec["is_number"], ref["is_number"],
                       relation=ref.get("relation", "related"))
    return g


def build_faiss_index(corpus):
    texts = [f"{rec['title']}. {rec.get('scope', '')}" for rec in corpus]
    print(f"Requesting embeddings for {len(texts)} standards from Hugging Face...")
    vectors = embeddings.embed_texts(texts)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized inner product
    index.add(vectors)
    id_map = [rec["is_number"] for rec in corpus]
    return index, id_map


def main():
    print("Loading corpus...")
    corpus = load_corpus()
    print(f"Loaded {len(corpus)} standards.")

    print("Building FAISS index...")
    index, id_map = build_faiss_index(corpus)
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "id_map.json"), "w", encoding="utf-8") as f:
        json.dump(id_map, f)

    print("Building knowledge graph...")
    graph = build_graph(corpus)
    with open(os.path.join(INDEX_DIR, "graph.gpickle"), "wb") as f:
        pickle.dump(graph, f)

    with open(os.path.join(INDEX_DIR, "corpus.json"), "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Index has {len(id_map)} standards, "
          f"graph has {graph.number_of_edges()} cross-reference edges.")


if __name__ == "__main__":
    main()
