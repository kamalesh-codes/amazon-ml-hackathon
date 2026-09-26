from dataclasses import dataclass, field
from pathlib import Path
import json

@dataclass
class PipelineConfig:
    source1_path: str = "data/source1.tsv"
    source2_path: str = "data/source2.tsv"
    source3_path: str = "data/source3.tsv"
    ground_truth_path: str = "data/ground_truth.tsv"
    index_dir: str = "indexes"
    candidates_path: str = "outputs/candidates.tsv"
    evaluation_path: str = "outputs/evaluation.json"
    chunk_size: int = 4096
    embedding_batch_size: int = 64
    top_k: int = 50
    model_name: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    gpu_ids: list[int] = field(default_factory=lambda: [0, 1])
    nlist: int = 4096
    pq_m: int = 64
    pq_nbits: int = 8
    nprobe: int = 32
    exact_threshold: int = 50000
    fallback_mode: str = "all_countries"  # all_countries | unknown_only | none
    temp_dir: str = "tmp"
    workers: int = 1
    log_level: str = "INFO"
    backend: str = "sentence_transformers"  # hash is for tests only
    show_progress: bool = True

    @classmethod
    def from_mapping(cls, values):
        """Create a config from a plain mapping or an OmegaConf DictConfig."""
        data = dict(values)
        data.pop("hydra", None)
        return cls(**data)

    @classmethod
    def from_json(cls, path: str):
        with open(path, encoding="utf-8") as f:
            return cls.from_mapping(json.load(f))

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2, ensure_ascii=False)

    def index_path(self):
        return Path(self.index_dir)
