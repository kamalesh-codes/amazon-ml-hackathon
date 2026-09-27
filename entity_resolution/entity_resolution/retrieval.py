from pathlib import Path

from .country import route_country
from .embedding import make_embedder
from .indexing import VectorIndex
from .preprocessing import normalize_entity


class Retriever:
    def __init__(self, index_dir, config, device=None):
        self.config = config
        self.index_dir = index_dir
        self.embedder = make_embedder(config, device)
        self.countries = []
        self.indexes = {}
        root = Path(index_dir)
        if not root.exists():
            return
        shard_roots = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("shard-gpu"))
        if not shard_roots:
            shard_roots = [root]
        for shard_root in shard_roots:
            for country_dir in sorted(shard_root.iterdir()):
                if not country_dir.is_dir() or not ((country_dir / "name.config.json").exists() or (country_dir / "combined.config.json").exists()):
                    continue
                country = country_dir.name
                if (country_dir / "combined.config.json").exists():
                    shard = {"combined": VectorIndex.load(country_dir, "combined")}
                else:
                    shard = {"name": VectorIndex.load(country_dir, "name"), "address": VectorIndex.load(country_dir, "address")}
                self.indexes.setdefault(country, []).append(shard)
        self.countries = sorted(self.indexes)
        self.shard_count = sum(len(v) for v in self.indexes.values())
        if not self.countries:
            raise FileNotFoundError(
                f"No country indexes found under {index_dir}. Build indexes first and use the same index_dir."
            )

    def retrieve_batch(self, rows):
        """Embed and search a whole query batch, vectorizing FAISS calls too."""
        normalized = [normalize_entity(row) for row in rows]
        candidates = [[] for _ in normalized]
        if self.config.retrieval_mode == "combined":
            names = addresses = None
            combined = self.embedder.encode(
                [f"{row['business_name_norm']} [SEP] {row['business_address_norm']}" for row in normalized],
                self.config.embedding_batch_size,
            )
        else:
            combined = None
            names = self.embedder.encode([row["business_name_norm"] for row in normalized], self.config.embedding_batch_size)
            addresses = self.embedder.encode([row["business_address_norm"] for row in normalized], self.config.embedding_batch_size)
        country_to_rows = {}
        for i, row in enumerate(normalized):
            for country in route_country(row.get("country"), self.countries, self.config.fallback_mode):
                country_to_rows.setdefault(country, []).append(i)
        for country, row_indices in country_to_rows.items():
            for shard in self.indexes[country]:
                if self.config.retrieval_mode == "combined":
                    queries_by_field = (("combined", combined[row_indices]),)
                else:
                    queries_by_field = (("name", names[row_indices]), ("address", addresses[row_indices]))
                for field, queries in queries_by_field:
                    _, internal_ids = shard[field].search(queries, self.config.top_k)
                    for local_i, result_ids in enumerate(internal_ids):
                        global_i = row_indices[local_i]
                        for internal_id in result_ids:
                            if internal_id >= 0:
                                candidates[global_i].append(shard[field].metadata[int(internal_id)]["entity_id"])
        return [list(dict.fromkeys(ids)) for ids in candidates]

    def retrieve(self, row):
        return self.retrieve_batch([row])[0]
