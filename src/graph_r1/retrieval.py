"""
Dual-path Hypergraph Retrieval with Reciprocal Rank Fusion.

Implements Section 4.2 of the paper:
- Entity-based hyperedge retrieval (Eq. 7)
- Direct hyperedge retrieval (Eq. 8)
- Fusion via Reciprocal Rank Aggregation (Eq. 9)
"""

from collections import defaultdict

import faiss
import numpy as np


class HypergraphRetriever:
    """Retrieves n-ary relational facts from a knowledge hypergraph."""

    def __init__(self, hypergraph, embedding_model=None):
        self.hg = hypergraph
        self.embedding_model = embedding_model

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a query string into an embedding vector."""
        emb = self.embedding_model.encode_queries([query])
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb.astype(np.float32)

    def entity_retrieval(self, query: str, k_v: int = 5) -> list[dict]:
        """Entity-based hyperedge retrieval (Eq. 7).

        Find top-k_v entities similar to query, then collect
        all hyperedges connected to those entities.
        """
        if self.hg.entity_index is None or self.hg.entity_index.ntotal == 0:
            return []

        query_emb = self._encode_query(query)
        k_v = min(k_v, self.hg.entity_index.ntotal)
        scores, ids = self.hg.entity_index.search(query_emb, k_v)

        entity_keys = list(self.hg.entities.keys())
        retrieved_facts = []
        seen_he_ids = set()

        for idx in ids[0]:
            if idx < 0 or idx >= len(entity_keys):
                continue
            entity_key = entity_keys[idx]
            he_ids = self.hg.entity_to_hyperedges.get(entity_key, [])
            for he_id in he_ids:
                if he_id not in seen_he_ids and he_id in self.hg.hyperedges:
                    seen_he_ids.add(he_id)
                    he = self.hg.hyperedges[he_id]
                    entity_names = [
                        self.hg.entities[ek]["entity_name"]
                        for ek in he["entities"]
                        if ek in self.hg.entities
                    ]
                    retrieved_facts.append({
                        "he_id": he_id,
                        "content": he["content"],
                        "entities": entity_names,
                        "source": "entity",
                    })

        return retrieved_facts

    def hyperedge_retrieval(self, query: str, k_h: int = 5) -> list[dict]:
        """Direct hyperedge retrieval (Eq. 8).

        Find top-k_h hyperedges by query-hyperedge similarity.
        """
        if self.hg.hyperedge_index is None or self.hg.hyperedge_index.ntotal == 0:
            return []

        query_emb = self._encode_query(query)
        k_h = min(k_h, self.hg.hyperedge_index.ntotal)
        scores, ids = self.hg.hyperedge_index.search(query_emb, k_h)

        he_keys = list(self.hg.hyperedges.keys())
        retrieved_facts = []

        for idx in ids[0]:
            if idx < 0 or idx >= len(he_keys):
                continue
            he_id = he_keys[idx]
            he = self.hg.hyperedges[he_id]
            entity_names = [
                self.hg.entities[ek]["entity_name"]
                for ek in he["entities"]
                if ek in self.hg.entities
            ]
            retrieved_facts.append({
                "he_id": he_id,
                "content": he["content"],
                "entities": entity_names,
                "source": "hyperedge",
            })

        return retrieved_facts

    def retrieve(self, query: str, top_k: int = 5, k_v: int = 5, k_h: int = 5) -> list[dict]:
        """Fused retrieval via Reciprocal Rank Aggregation (Eq. 9).

        Combines entity-based and direct hyperedge retrieval results
        using reciprocal rank scoring: Score(f) = 1/r_V + 1/r_H
        """
        entity_facts = self.entity_retrieval(query, k_v=k_v)
        hyperedge_facts = self.hyperedge_retrieval(query, k_h=k_h)

        scores = defaultdict(float)
        fact_map = {}

        for rank, fact in enumerate(entity_facts):
            he_id = fact["he_id"]
            scores[he_id] += 1.0 / (rank + 1)
            fact_map[he_id] = fact

        for rank, fact in enumerate(hyperedge_facts):
            he_id = fact["he_id"]
            scores[he_id] += 1.0 / (rank + 1)
            if he_id not in fact_map:
                fact_map[he_id] = fact

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for he_id, score in ranked:
            fact = fact_map[he_id].copy()
            fact["rrf_score"] = score
            results.append(fact)

        return results

    def format_knowledge(self, facts: list[dict]) -> str:
        """Format retrieved facts into a knowledge string for the agent prompt."""
        if not facts:
            return "No relevant knowledge found."

        parts = []
        for i, fact in enumerate(facts):
            entities_str = ", ".join(fact.get("entities", []))
            parts.append(
                f"[{i + 1}] {fact['content']}"
                + (f" (Entities: {entities_str})" if entities_str else "")
            )

        return "\n".join(parts)


class RetrievalServer:
    """Lightweight retrieval server wrapping the HypergraphRetriever."""

    def __init__(self, retriever: HypergraphRetriever):
        self.retriever = retriever

    def search(self, queries: list[str], top_k: int = 5) -> list[str]:
        """Process a batch of queries and return formatted results."""
        import json
        results = []
        for query in queries:
            facts = self.retriever.retrieve(query, top_k=top_k)
            knowledge = self.retriever.format_knowledge(facts)
            results.append(json.dumps({"results": knowledge}))
        return results
