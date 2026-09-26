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
        self.indexes = {}  # country -> list of {name,address}, one entry per GPU shard
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
        """Embed names/addresses in batches, then retrieve each query."""
        normalized = [normalize_entity(row) for row in rows]
        names = self.embedder.encode([row["business_name_norm"] for row in normalized], self.config.embedding_batch_size)
        addresses = self.embedder.encode([row["business_address_norm"] for row in normalized], self.config.embedding_batch_size)
        results = []
        for row, name, address in zip(normalized, names, addresses):
            countries = route_country(row.get("country"), self.countries, self.config.fallback_mode)
            ranked = []
            for country in countries:
                for shard in self.indexes[country]:
                    for field, query in (("name", name), ("address", address)):
                        _, internal_ids = shard[field].search(query[None, :], self.config.top_k)
                        for internal_id in internal_ids[0]:
                            if internal_id >= 0:
                                entity_id = shard[field].metadata[int(internal_id)]["entity_id"]
                                ranked.append((entity_id, int(internal_id)))
            # Search results are already rank ordered within each shard/field;
            # deterministic union is sufficient for candidate generation.
            results.append(list(dict.fromkeys(entity_id for entity_id, _ in ranked)))
        return results

    def retrieve(self, row):
        return self.retrieve_batch([row])[0]
