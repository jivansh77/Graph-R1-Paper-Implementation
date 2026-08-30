"""
Build Knowledge HyperGraph from corpus documents.

Usage:
    python scripts/build_hypergraph.py --dataset 2WikiMultiHopQA --data_dir data/ --output_dir experiments/

Requires an OpenAI API key for GPT-4o-mini extraction, or use a local model.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph_r1.hypergraph import KnowledgeHyperGraph


def create_llm_fn(api_key: str | None = None, model: str = "gpt-4o-mini"):
    """Create an LLM function for n-ary relation extraction."""
    if api_key:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        def llm_fn(prompt: str) -> str:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4096,
            )
            return response.choices[0].message.content
        return llm_fn

    def local_llm_fn(prompt: str) -> str:
        """Fallback: simple regex-based extraction for testing."""
        import re
        sentences = re.split(r'[.!?]+', prompt.split("Text:")[-1] if "Text:" in prompt else prompt)
        output_parts = []
        entities_seen = set()

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue
            output_parts.append(f'("hyperedge"<|>{sent})')

            words = sent.split()
            for word in words:
                clean = word.strip(".,;:!?\"'()")
                if clean and clean[0].isupper() and len(clean) > 2 and clean.lower() not in entities_seen:
                    entities_seen.add(clean.lower())
                    output_parts.append(
                        f'("entity"<|>{clean}<|>unknown<|>{clean} mentioned in context)'
                    )

        return "##".join(output_parts) + "<|COMPLETE|>"

    return local_llm_fn


def main():
    parser = argparse.ArgumentParser(description="Build Knowledge HyperGraph")
    parser.add_argument("--dataset", default="2WikiMultiHopQA")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="experiments")
    parser.add_argument("--openai_api_key", default=None)
    parser.add_argument("--chunk_size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    corpus_path = os.path.join(args.data_dir, args.dataset, "corpus.jsonl")
    if not os.path.exists(corpus_path):
        print(f"Corpus not found at {corpus_path}")
        print("Will try to build from dataset documents...")
        raw_path = os.path.join(args.data_dir, args.dataset, "raw", "train.json")
        if os.path.exists(raw_path):
            with open(raw_path) as f:
                data = json.load(f)
            documents = []
            for item in data:
                ctx = item.get("gold_context", item.get("context", ""))
                if isinstance(ctx, list):
                    for c in ctx:
                        if isinstance(c, dict):
                            documents.append(c.get("content", c.get("text", str(c))))
                        else:
                            documents.append(str(c))
                elif ctx:
                    documents.append(str(ctx))
            documents = [d for d in documents if d.strip()]
            print(f"Extracted {len(documents)} documents from training data")
        else:
            print(f"No data found. Please prepare data first.")
            return
    else:
        documents = []
        with open(corpus_path) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    documents.append(item.get("contents", item.get("content", item.get("text", ""))))
        print(f"Loaded {len(documents)} documents from corpus")

    working_dir = os.path.join(args.output_dir, args.dataset)
    hg = KnowledgeHyperGraph(working_dir=working_dir)

    print("Step 1: Chunking documents...")
    chunks = hg.chunk_documents(documents, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"Created {len(chunks)} chunks")

    print("Step 2: Extracting n-ary relations...")
    llm_fn = create_llm_fn(args.openai_api_key)
    hg.extract_relations(chunks, llm_fn)
    print(f"Extracted {len(hg.entities)} entities, {len(hg.hyperedges)} hyperedges")

    print("Step 3: Building embeddings...")
    hg.build_embeddings(device=args.device)

    print("Step 4: Saving hypergraph...")
    hg.save()

    stats = hg.stats()
    print(f"\nHyperGraph Statistics:")
    print(f"  Entities: {stats['num_entities']}")
    print(f"  Hyperedges: {stats['num_hyperedges']}")
    print(f"  Avg entities/hyperedge: {stats['avg_entities_per_hyperedge']:.1f}")
    print(f"\nSaved to {working_dir}")


if __name__ == "__main__":
    main()
