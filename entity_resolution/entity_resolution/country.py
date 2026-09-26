import re

ALIASES = {
    "india":"IN", "in":"IN", "ind":"IN", "us":"US", "usa":"US",
    "united states":"US", "united states of america":"US", "gb":"GB", "uk":"GB",
    "united kingdom":"GB", "great britain":"GB", "de":"DE", "germany":"DE",
    "fr":"FR", "france":"FR", "ca":"CA", "canada":"CA", "au":"AU", "australia":"AU",
}

def normalize_country(value) -> str:
    s = re.sub(r"\s+", " ", str(value or "").strip().casefold())
    if s in ALIASES: return ALIASES[s]
    if len(s) == 2 and s.isalpha(): return s.upper()
    if len(s) == 3 and s.isalpha(): return s.upper()  # retain unknown ISO3 deterministically
    return ""

def route_country(query_country, available, fallback_mode="all_countries") -> list[str]:
    c = normalize_country(query_country)
    available = sorted(set(available))
    if c and c in available: return [c]
    if fallback_mode == "none": return []
    if fallback_mode == "unknown_only": return ["__UNKNOWN__"] if "__UNKNOWN__" in available else []
    return available
