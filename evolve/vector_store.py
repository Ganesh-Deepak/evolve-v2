import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from evolve.models import Candidate


class VectorStore:
    def __init__(self, persist_dir: str = "./data/chromadb"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.ef = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="evolve_candidates",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add_candidate(self, candidate: Candidate) -> None:
        existing = self.collection.get(ids=[candidate.code_hash])
        if existing and existing["ids"]:
            old_fitness = (existing["metadatas"] or [{}])[0].get("fitness", 0)
            if candidate.fitness is not None and candidate.fitness > old_fitness:
                self.collection.update(
                    ids=[candidate.code_hash],
                    documents=[candidate.code],
                    metadatas=[{
                        "fitness": candidate.fitness or 0.0,
                        "generation": candidate.generation,
                        "mutation_type": candidate.mutation_type,
                    }],
                )
            return

        self.collection.add(
            ids=[candidate.code_hash],
            documents=[candidate.code],
            metadatas=[{
                "fitness": candidate.fitness or 0.0,
                "generation": candidate.generation,
                "mutation_type": candidate.mutation_type,
            }],
        )

    def get_similar(self, code: str, n: int = 3,
                    min_fitness: float = 0.0) -> list[tuple[str, float]]:
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[code],
            n_results=min(n * 2, self.collection.count()),
            where={"fitness": {"$gte": min_fitness}} if min_fitness > 0 else None,
        )

        pairs = []
        if results["documents"] and results["metadatas"]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                fitness = meta.get("fitness", 0.0)
                if fitness >= min_fitness:
                    pairs.append((doc, fitness))

        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:n]

    def is_duplicate(self, code: str, threshold: float = 0.98) -> bool:
        if self.collection.count() == 0:
            return False

        results = self.collection.query(
            query_texts=[code],
            n_results=1,
        )

        if results["distances"] and results["distances"][0]:
            cosine_distance = results["distances"][0][0]
            similarity = 1.0 - cosine_distance
            return similarity > threshold

        return False

    def get_cached_fitness(self, code_hash: str) -> float | None:
        results = self.collection.get(ids=[code_hash])
        if results and results["ids"]:
            meta = (results["metadatas"] or [{}])[0]
            return meta.get("fitness")
        return None

    def clear(self) -> None:
        self.client.delete_collection("evolve_candidates")
        self.collection = self.client.get_or_create_collection(
            name="evolve_candidates",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

    def seed_templates(self, templates: list[tuple[str, str]]) -> None:
        """Seed the DB with template code. templates = [(code, name), ...]"""
        if self.collection.count() > 0:
            return
        for code, name in templates:
            candidate = Candidate(
                code=code,
                generation=0,
                mutation_type="template",
                mutation_description=f"Template: {name}",
                fitness=0.0,
            )
            self.add_candidate(candidate)
