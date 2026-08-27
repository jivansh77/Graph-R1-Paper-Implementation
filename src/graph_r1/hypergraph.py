"""
Knowledge Hypergraph Construction for Graph-R1.

Implements the three-stage pipeline from Section 4.1 and Appendix J:
1. Document chunking (1200 tokens, 50-token overlap)
2. N-ary relation extraction via LLM (GPT-4o-mini compatible)
3. Entity/hyperedge embedding with bge-large-en-v1.5
"""

import json
import os
import re
import time
from typing import Any

import faiss
import numpy as np
import tiktoken
from tqdm import tqdm


EXTRACTION_PROMPT = """Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

-Steps-
1. Divide the text into several complete knowledge segments. For each knowledge segment, extract the following information:
-- knowledge_segment: A sentence that describes the content of the knowledge segment.
Format each knowledge segment as ("hyperedge"{tuple_delimiter}<knowledge_segment>)

2. Identify all entities in each knowledge segment. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: Type of the entity.
- entity_description: Comprehensive description of the entity's attributes and activities.
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

3. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter} as the list delimiter.

######################
-Examples-
######################
{examples}

######################
-Real Data-
######################
Text: {input_text}
"""

EXTRACTION_EXAMPLES = """Example 1:
Text: The Verdantia's Central Park is an urban oasis that features a large lake, walking trails, and the famous Verdantia Botanical Garden. It is maintained by the Verdantia Parks Department and was designed by landscape architect Frederick Olmsted.
Output:
("hyperedge"{tuple_delimiter}The Verdantia's Central Park features a large lake, walking trails, and the famous Verdantia Botanical Garden.)
##
("hyperedge"{tuple_delimiter}The park is maintained by the Verdantia Parks Department and designed by Frederick Olmsted.)
##
("entity"{tuple_delimiter}VERDANTIA'S CENTRAL PARK{tuple_delimiter}geo{tuple_delimiter}An urban oasis with a lake, trails, and botanical garden)
##
("entity"{tuple_delimiter}VERDANTIA BOTANICAL GARDEN{tuple_delimiter}geo{tuple_delimiter}A famous botanical garden located within Central Park)
##
("entity"{tuple_delimiter}VERDANTIA PARKS DEPARTMENT{tuple_delimiter}organization{tuple_delimiter}Government department maintaining the park)
##
("entity"{tuple_delimiter}FREDERICK OLMSTED{tuple_delimiter}person{tuple_delimiter}Landscape architect who designed the park)
{completion_delimiter}
"""

TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"


class KnowledgeHyperGraph:
    """Builds and manages a knowledge hypergraph from a document corpus."""

    def __init__(self, working_dir: str, embedding_model_name: str = "BAAI/bge-large-en-v1.5"):
        self.working_dir = working_dir
        self.embedding_model_name = embedding_model_name
        os.makedirs(working_dir, exist_ok=True)

        self.entities: dict[str, dict[str, Any]] = {}
        self.hyperedges: dict[str, dict[str, Any]] = {}
        self.entity_to_hyperedges: dict[str, list[str]] = {}

        self.entity_embeddings: np.ndarray | None = None
        self.hyperedge_embeddings: np.ndarray | None = None
        self.entity_index: faiss.IndexFlatIP | None = None
        self.hyperedge_index: faiss.IndexFlatIP | None = None

        self.entity_names: list[str] = []
        self.hyperedge_contents: list[str] = []

    def chunk_documents(self, documents: list[str], chunk_size: int = 1200,
                        overlap: int = 50) -> list[dict]:
        """Split documents into fixed-size token chunks with overlap."""
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            enc = None

        all_chunks = []
        for doc_idx, doc in enumerate(tqdm(documents, desc="Chunking documents")):
            if enc is not None:
                tokens = enc.encode(doc)
                for start in range(0, len(tokens), chunk_size - overlap):
                    end = min(start + chunk_size, len(tokens))
                    chunk_text = enc.decode(tokens[start:end])
                    all_chunks.append({
                        "doc_id": doc_idx,
                        "chunk_id": len(all_chunks),
                        "content": chunk_text,
                    })
                    if end >= len(tokens):
                        break
            else:
                words = doc.split()
                for start in range(0, len(words), chunk_size - overlap):
                    end = min(start + chunk_size, len(words))
                    chunk_text = " ".join(words[start:end])
                    all_chunks.append({
                        "doc_id": doc_idx,
                        "chunk_id": len(all_chunks),
                        "content": chunk_text,
                    })
                    if end >= len(words):
                        break

        return all_chunks

    def extract_relations(self, chunks: list[dict], llm_fn, batch_size: int = 50,
                          max_retries: int = 3) -> None:
        """Extract n-ary relational facts from chunks using an LLM.

        Args:
            chunks: List of chunk dicts with 'content' key.
            llm_fn: Callable(prompt: str) -> str that calls the extraction LLM.
        """
        for i in tqdm(range(0, len(chunks), batch_size), desc="Extracting relations"):
            batch = chunks[i:i + batch_size]
            for chunk in batch:
                prompt = EXTRACTION_PROMPT.format(
                    language="English",
                    tuple_delimiter=TUPLE_DELIMITER,
                    record_delimiter=RECORD_DELIMITER,
                    completion_delimiter=COMPLETION_DELIMITER,
                    examples=EXTRACTION_EXAMPLES.format(
                        tuple_delimiter=TUPLE_DELIMITER,
                        completion_delimiter=COMPLETION_DELIMITER,
                    ),
                    input_text=chunk["content"],
                )

                for attempt in range(max_retries):
                    try:
                        response = llm_fn(prompt)
                        self._parse_extraction(response, chunk["chunk_id"])
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            print(f"Failed chunk {chunk['chunk_id']}: {e}")
                        time.sleep(2)

    def _parse_extraction(self, response: str, chunk_id: int) -> None:
        """Parse LLM extraction output into entities and hyperedges."""
        response = response.replace(COMPLETION_DELIMITER, "")
        records = response.split(RECORD_DELIMITER)

        current_hyperedge_id = None
        for record in records:
            record = record.strip()
            if not record:
                continue

            match = re.search(r'\("hyperedge"' + re.escape(TUPLE_DELIMITER) + r'(.+?)\)', record)
            if match:
                content = match.group(1).strip()
                he_id = f"he_{len(self.hyperedges)}"
                self.hyperedges[he_id] = {
                    "id": he_id,
                    "content": content,
                    "chunk_id": chunk_id,
                    "entities": [],
                }
                current_hyperedge_id = he_id
                continue

            match = re.search(
                r'\("entity"' + re.escape(TUPLE_DELIMITER)
                + r'(.+?)' + re.escape(TUPLE_DELIMITER)
                + r'(.+?)' + re.escape(TUPLE_DELIMITER)
                + r'(.+?)\)',
                record,
            )
            if match:
                name = match.group(1).strip()
                etype = match.group(2).strip()
                desc = match.group(3).strip()

                entity_key = name.lower()
                if entity_key not in self.entities:
                    eid = f"ent_{len(self.entities)}"
                    self.entities[entity_key] = {
                        "id": eid,
                        "entity_name": name,
                        "entity_type": etype,
                        "description": desc,
                        "source_chunks": [chunk_id],
                    }
                    self.entity_to_hyperedges[entity_key] = []
                else:
                    existing = self.entities[entity_key]
                    if chunk_id not in existing["source_chunks"]:
                        existing["source_chunks"].append(chunk_id)
                    if len(existing["description"]) < len(desc):
                        existing["description"] = desc

                if current_hyperedge_id:
                    self.hyperedges[current_hyperedge_id]["entities"].append(entity_key)
                    if current_hyperedge_id not in self.entity_to_hyperedges.get(entity_key, []):
                        self.entity_to_hyperedges.setdefault(entity_key, []).append(current_hyperedge_id)

    def build_embeddings(self, device: str = "cpu") -> None:
        """Encode entities and hyperedges using bge-large-en-v1.5."""
        from FlagEmbedding import FlagAutoModel

        model = FlagAutoModel.from_finetuned(
            self.embedding_model_name,
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
            devices=device,
        )

        self.entity_names = []
        entity_texts = []
        for key in self.entities:
            ent = self.entities[key]
            self.entity_names.append(ent["entity_name"])
            entity_texts.append(f"{ent['entity_name']}: {ent['description']}")

        if entity_texts:
            self.entity_embeddings = model.encode(entity_texts)
            self.entity_embeddings = self.entity_embeddings / np.linalg.norm(
                self.entity_embeddings, axis=1, keepdims=True
            )
            dim = self.entity_embeddings.shape[1]
            self.entity_index = faiss.IndexFlatIP(dim)
            self.entity_index.add(self.entity_embeddings.astype(np.float32))

        self.hyperedge_contents = []
        hyperedge_texts = []
        for he_id in self.hyperedges:
            he = self.hyperedges[he_id]
            self.hyperedge_contents.append(he["content"])
            entities_str = ", ".join(he["entities"])
            hyperedge_texts.append(f"{he['content']} Entities: {entities_str}")

        if hyperedge_texts:
            self.hyperedge_embeddings = model.encode(hyperedge_texts)
            self.hyperedge_embeddings = self.hyperedge_embeddings / np.linalg.norm(
                self.hyperedge_embeddings, axis=1, keepdims=True
            )
            dim = self.hyperedge_embeddings.shape[1]
            self.hyperedge_index = faiss.IndexFlatIP(dim)
            self.hyperedge_index.add(self.hyperedge_embeddings.astype(np.float32))

        print(f"Built hypergraph: {len(self.entities)} entities, {len(self.hyperedges)} hyperedges")

    def save(self) -> None:
        """Persist the hypergraph to disk."""
        with open(os.path.join(self.working_dir, "kv_store_entities.json"), "w") as f:
            json.dump(self.entities, f)
        with open(os.path.join(self.working_dir, "kv_store_hyperedges.json"), "w") as f:
            json.dump(self.hyperedges, f)
        with open(os.path.join(self.working_dir, "entity_to_hyperedges.json"), "w") as f:
            json.dump(self.entity_to_hyperedges, f)
        with open(os.path.join(self.working_dir, "entity_names.json"), "w") as f:
            json.dump(self.entity_names, f)
        with open(os.path.join(self.working_dir, "hyperedge_contents.json"), "w") as f:
            json.dump(self.hyperedge_contents, f)

        if self.entity_index is not None:
            faiss.write_index(self.entity_index, os.path.join(self.working_dir, "index_entity.bin"))
        if self.hyperedge_index is not None:
            faiss.write_index(self.hyperedge_index, os.path.join(self.working_dir, "index_hyperedge.bin"))

        if self.entity_embeddings is not None:
            np.save(os.path.join(self.working_dir, "entity_embeddings.npy"), self.entity_embeddings)
        if self.hyperedge_embeddings is not None:
            np.save(os.path.join(self.working_dir, "hyperedge_embeddings.npy"), self.hyperedge_embeddings)

    def load(self) -> None:
        """Load a previously saved hypergraph."""
        with open(os.path.join(self.working_dir, "kv_store_entities.json")) as f:
            self.entities = json.load(f)
        with open(os.path.join(self.working_dir, "kv_store_hyperedges.json")) as f:
            self.hyperedges = json.load(f)
        with open(os.path.join(self.working_dir, "entity_to_hyperedges.json")) as f:
            self.entity_to_hyperedges = json.load(f)
        with open(os.path.join(self.working_dir, "entity_names.json")) as f:
            self.entity_names = json.load(f)
        with open(os.path.join(self.working_dir, "hyperedge_contents.json")) as f:
            self.hyperedge_contents = json.load(f)

        idx_entity_path = os.path.join(self.working_dir, "index_entity.bin")
        if os.path.exists(idx_entity_path):
            self.entity_index = faiss.read_index(idx_entity_path)
        idx_he_path = os.path.join(self.working_dir, "index_hyperedge.bin")
        if os.path.exists(idx_he_path):
            self.hyperedge_index = faiss.read_index(idx_he_path)

        emb_path = os.path.join(self.working_dir, "entity_embeddings.npy")
        if os.path.exists(emb_path):
            self.entity_embeddings = np.load(emb_path)
        he_emb_path = os.path.join(self.working_dir, "hyperedge_embeddings.npy")
        if os.path.exists(he_emb_path):
            self.hyperedge_embeddings = np.load(he_emb_path)

    def stats(self) -> dict:
        """Return summary statistics."""
        return {
            "num_entities": len(self.entities),
            "num_hyperedges": len(self.hyperedges),
            "avg_entities_per_hyperedge": (
                np.mean([len(h["entities"]) for h in self.hyperedges.values()])
                if self.hyperedges else 0
            ),
        }
