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
                if not country_dir.is_dir() or not (country_dir / "name.config.json").exists():
                    continue
                country = country_dir.name
                self.indexes.setdefault(country, []).append(
                    {"name": VectorIndex.load(country_dir, "name"), "address": VectorIndex.load(country_dir, "address")}
                )
        self.countries = sorted(self.indexes)

    def retrieve_batch(self, rows):
        """Embed and search a whole query batch, vectorizing FAISS calls too."""
        normalized = [normalize_entity(row) for row in rows]
        names = self.embedder.encode([row["business_name_norm"] for row in normalized], self.config.embedding_batch_size)
        addresses = self.embedder.encode([row["business_address_norm"] for row in normalized], self.config.embedding_batch_size)
        candidates = [[] for _ in normalized]
        country_to_rows = {}
        for i, row in enumerate(normalized):
            for country in route_country(row.get("country"), self.countries, self.config.fallback_mode):
                country_to_rows.setdefault(country, []).append(i)
        for country, row_indices in country_to_rows.items():
            q_names = names[row_indices]
            q_addresses = addresses[row_indices]
            for shard in self.indexes[country]:
                for field, queries in (("name", q_names), ("address", q_addresses)):
                    _, internal_ids = shard[field].search(queries, self.config.top_k)
                    for local_i, result_ids in enumerate(internal_ids):
                        global_i = row_indices[local_i]
                        for internal_id in result_ids:
                            if internal_id >= 0:
                                candidates[global_i].append(shard[field].metadata[int(internal_id)]["entity_id"])
        return [list(dict.fromkeys(ids)) for ids in candidates]

    def retrieve(self, row):
        return self.retrieve_batch([row])[0]
