import csv
import hashlib
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
except ImportError:  # pragma: no cover
    Chem = None
    DataStructs = None
    RDLogger = None
    rdFingerprintGenerator = None
try:
    from rdkit.Chem.Draw import rdMolDraw2D
except ImportError:  # pragma: no cover
    rdMolDraw2D = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from cl_model import ContrastiveLearningModel
    from smile_to_graph import SMILESToGraph
except ImportError:  # pragma: no cover
    ContrastiveLearningModel = None
    SMILESToGraph = None


@dataclass
class SearchResult:
    rank: int
    smiles: str
    score: float
    metadata: dict[str, Any]


class SimilarityEngine:
    """Builds a searchable library from SMILES and serves nearest-neighbor queries."""

    CACHE_VERSION = "v4"

    def __init__(
        self,
        dataset_path: str | Path,
        model_path: str | Path | None = None,
        cache_dir: str | Path = "demo/cache",
        smiles_column: str = "SMILES",
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.model_path = Path(model_path) if model_path else None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.smiles_column = smiles_column
        self.mode = "unavailable"
        self.records: list[dict[str, Any]] = []
        self.matrix: np.ndarray | None = None
        self.fingerprints: list[Any] = []
        self.rerank_k = 200
        self.stats = {"invalid_smiles": 0, "loaded_records": 0}
        self.graph_converter = None
        self.model = None
        self.fp_generator = (
            rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            if rdFingerprintGenerator is not None
            else None
        )

    @property
    def ready(self) -> bool:
        if self.mode in {"fingerprint", "hybrid"}:
            return len(self.fingerprints) > 0 and len(self.records) > 0
        return self.matrix is not None and len(self.records) > 0

    def initialize(self) -> None:
        started_at = time.time()
        print(
            "[engine] initialize start",
            {
                "dataset": str(self.dataset_path),
                "model": str(self.model_path) if self.model_path else None,
                "cache_dir": str(self.cache_dir),
            },
            flush=True,
        )
        if Chem is None:
            self.mode = "unavailable"
            print("[engine] RDKit unavailable", flush=True)
            return
        if RDLogger is not None:
            RDLogger.DisableLog("rdApp.error")

        self.records = self._load_records()
        print("[engine] records loaded", {"count": len(self.records)}, flush=True)

        encoder_available = self._can_use_encoder()
        print("[engine] encoder availability", {"encoder_available": encoder_available}, flush=True)
        if self.dataset_path.suffix.lower() == ".smi" or len(self.records) > 100_000:
            self.mode = "hybrid" if encoder_available else "fingerprint"
        else:
            self.mode = "encoder" if encoder_available else "fingerprint"
        print("[engine] selected mode", {"mode": self.mode}, flush=True)

        self._load_or_build_index()
        self.stats["loaded_records"] = len(self.records)
        print(
            "[engine] initialize complete",
            {
                "ready": self.ready,
                "mode": self.mode,
                "records": len(self.records),
                "seconds": round(time.time() - started_at, 2),
            },
            flush=True,
        )

    def _can_use_encoder(self) -> bool:
        return all(
            [
                self.model_path is not None,
                self.model_path.exists(),
                torch is not None,
                ContrastiveLearningModel is not None,
                SMILESToGraph is not None,
            ]
        )

    def _load_records(self) -> list[dict[str, Any]]:
        if self.dataset_path.suffix.lower() == ".smi":
            return self._load_smi_records()

        deduped: dict[str, dict[str, Any]] = {}
        with self.dataset_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_smiles = (row.get(self.smiles_column) or "").strip()
                canonical = self._canonicalize_smiles(raw_smiles)
                if not canonical:
                    self.stats["invalid_smiles"] += 1
                    continue
                if canonical in deduped:
                    continue

                metadata = self._clean_metadata(row)
                deduped[canonical] = {
                    "smiles": canonical,
                    "raw_smiles": raw_smiles,
                    "metadata": metadata,
                }
        return list(deduped.values())

    def _load_smi_records(self) -> list[dict[str, Any]]:
        deduped: dict[str, dict[str, Any]] = {}
        with self.dataset_path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                raw_smiles = parts[0].strip()
                canonical = self._canonicalize_smiles(raw_smiles)
                if not canonical:
                    self.stats["invalid_smiles"] += 1
                    continue
                if canonical in deduped:
                    continue

                metadata: dict[str, Any] = {}
                if len(parts) > 1:
                    metadata["chembl_id"] = parts[1].strip()

                deduped[canonical] = {
                    "smiles": canonical,
                    "raw_smiles": raw_smiles,
                    "metadata": metadata,
                }
        return list(deduped.values())

    def _clean_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in metadata.items()
            if key not in {self.smiles_column, "mol", "logP"}
            and value not in ("", None)
        }

    def _cache_stem(self) -> str:
        dataset_stat = self.dataset_path.stat()
        digest = hashlib.sha256(
            (
                f"{self.dataset_path.name}::{dataset_stat.st_size}::"
                f"{self.mode}::{self.CACHE_VERSION}"
            ).encode("utf-8")
        ).hexdigest()[:12]
        return str(self.cache_dir / f"library_{digest}")

    def _load_or_build_index(self) -> None:
        matrix_path = Path(f"{self._cache_stem()}.npy")
        records_path = Path(f"{self._cache_stem()}.json")
        fingerprints_path = Path(f"{self._cache_stem()}_fps.pkl")
        print(
            "[engine] cache lookup",
            {
                "stem": self._cache_stem(),
                "records_exists": records_path.exists(),
                "matrix_exists": matrix_path.exists(),
                "fingerprints_exists": fingerprints_path.exists(),
            },
            flush=True,
        )

        if self.mode in {"fingerprint", "hybrid"} and records_path.exists() and fingerprints_path.exists():
            print("[engine] cache hit", {"mode": self.mode, "kind": "fingerprints"}, flush=True)
            self.records = json.loads(records_path.read_text())
            for record in self.records:
                record["metadata"] = self._clean_metadata(record.get("metadata", {}))
            with fingerprints_path.open("rb") as handle:
                fp_bytes = pickle.load(handle)
            self.fingerprints = [DataStructs.CreateFromBinaryText(fp) for fp in fp_bytes]
            if self.mode == "hybrid":
                self._load_encoder()
            return

        if self.mode == "encoder" and matrix_path.exists() and records_path.exists():
            print("[engine] cache hit", {"mode": self.mode, "kind": "matrix"}, flush=True)
            self.matrix = np.load(matrix_path)
            self.records = json.loads(records_path.read_text())
            for record in self.records:
                record["metadata"] = self._clean_metadata(record.get("metadata", {}))
            self._load_encoder()
            return

        print("[engine] cache miss", {"mode": self.mode}, flush=True)
        if self.mode == "encoder":
            self._load_encoder()
            matrix = [self._embed_smiles(record["smiles"]) for record in self.records]
        else:
            matrix = [self._fingerprint_array(record["smiles"]) for record in self.records]

        valid_rows: list[np.ndarray] = []
        valid_records: list[dict[str, Any]] = []
        for record, vector in zip(self.records, matrix):
            if vector is None:
                continue
            valid_rows.append(vector)
            valid_records.append(record)

        self.records = valid_records
        if self.mode == "encoder":
            self.matrix = np.vstack(valid_rows) if valid_rows else None
        else:
            self.fingerprints = valid_rows

        if self.mode == "encoder" and self.matrix is not None:
            np.save(matrix_path, self.matrix)
            records_path.write_text(json.dumps(self.records))
        elif self.mode in {"fingerprint", "hybrid"} and self.fingerprints:
            fp_bytes = [DataStructs.BitVectToBinaryText(fp) for fp in self.fingerprints]
            with fingerprints_path.open("wb") as handle:
                pickle.dump(fp_bytes, handle, protocol=pickle.HIGHEST_PROTOCOL)
            records_path.write_text(json.dumps(self.records))
            if self.mode == "hybrid":
                self._load_encoder()

    def _load_encoder(self) -> None:
        if self.model is not None:
            return
        print("[engine] loading encoder", {"model_path": str(self.model_path)}, flush=True)
        self.graph_converter = SMILESToGraph()
        self.model = ContrastiveLearningModel()
        state_dict = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("[engine] encoder loaded", flush=True)

    def _canonicalize_smiles(self, smiles: str) -> str | None:
        if not smiles or Chem is None:
            return None
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        return Chem.MolToSmiles(molecule)

    def _fingerprint_array(self, smiles: str) -> Any | None:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or self.fp_generator is None:
            return None
        return self.fp_generator.GetFingerprint(molecule)

    def _embed_smiles(self, smiles: str) -> np.ndarray | None:
        graph = self.graph_converter.smiles_to_graph(smiles)
        if graph is None:
            return None
        graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
        with torch.no_grad():
            embedding = self.model.encoder(graph)
        return embedding.numpy().flatten().astype(np.float32)

    def search(self, query_smiles: str, top_k: int = 10) -> dict[str, Any]:
        if not self.ready:
            return {
                "mode": self.mode,
                "query_smiles": query_smiles,
                "canonical_smiles": None,
                "results": [],
                "error": "Search index is not ready. Install the demo dependencies first.",
            }

        canonical = self._canonicalize_smiles(query_smiles)
        if canonical is None:
            return {
                "mode": self.mode,
                "query_smiles": query_smiles,
                "canonical_smiles": None,
                "results": [],
                "error": "That SMILES string could not be parsed.",
            }

        if self.mode == "encoder":
            query_vector = self._embed_smiles(canonical)
        else:
            query_vector = self._fingerprint_array(canonical)
        if query_vector is None:
            return {
                "mode": self.mode,
                "query_smiles": query_smiles,
                "canonical_smiles": canonical,
                "results": [],
                "error": "The query molecule could not be vectorized for similarity search.",
            }

        results: list[SearchResult] = []
        ranked_results = self._rank_candidates(canonical, query_vector, top_k)
        for idx, score in ranked_results:
            candidate = self.records[int(idx)]
            if candidate["smiles"] == canonical:
                continue
            results.append(
                SearchResult(
                    rank=len(results) + 1,
                    smiles=candidate["smiles"],
                    score=float(score),
                    metadata=candidate["metadata"],
                )
            )
            if len(results) >= top_k:
                break

        return {
            "mode": self.mode,
            "query_smiles": query_smiles,
            "canonical_smiles": canonical,
            "library_size": len(self.records),
            "results": [result.__dict__ for result in results],
        }

    def _cosine_similarity(self, query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        query_norm = np.linalg.norm(query) + 1e-8
        matrix_norm = np.linalg.norm(matrix, axis=1) + 1e-8
        return (matrix @ query) / (matrix_norm * query_norm)

    def _rerank_with_encoder(
        self,
        canonical_query: str,
        candidate_indices: np.ndarray,
    ) -> list[tuple[int, float]]:
        query_embedding = self._embed_smiles(canonical_query)
        if query_embedding is None:
            return []

        reranked: list[tuple[int, float]] = []
        for idx in candidate_indices:
            candidate_smiles = self.records[int(idx)]["smiles"]
            candidate_embedding = self._embed_smiles(candidate_smiles)
            if candidate_embedding is None:
                continue
            score = float(
                np.dot(query_embedding, candidate_embedding)
                / ((np.linalg.norm(query_embedding) + 1e-8) * (np.linalg.norm(candidate_embedding) + 1e-8))
            )
            reranked.append((int(idx), score))

        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked

    def _rank_candidates(
        self, canonical: str, query_vector: Any, top_k: int
    ) -> list[tuple[int, float]]:
        if self.mode == "encoder":
            similarities = self._cosine_similarity(query_vector, self.matrix)
            ranked_indices = np.argsort(similarities)[::-1]
            return [(int(idx), float(similarities[int(idx)])) for idx in ranked_indices]

        similarities = np.array(
            DataStructs.BulkTanimotoSimilarity(query_vector, self.fingerprints),
            dtype=np.float32,
        )
        candidate_count = max(top_k, self.rerank_k if self.mode == "hybrid" else top_k)
        ranked_indices = np.argsort(similarities)[::-1][:candidate_count]
        ranked_pairs = [(int(idx), float(similarities[int(idx)])) for idx in ranked_indices]

        if self.mode != "hybrid":
            return ranked_pairs

        reranked = self._rerank_with_encoder(canonical, ranked_indices)
        return reranked if reranked else ranked_pairs

    def render_svg(self, smiles: str, width: int = 320, height: int = 180) -> str | None:
        if Chem is None or rdMolDraw2D is None:
            return None
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        rdMolDraw2D.PrepareMolForDrawing(molecule)
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        options = drawer.drawOptions()
        options.setBackgroundColour((0.97, 0.98, 0.99))
        drawer.DrawMolecule(molecule)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
