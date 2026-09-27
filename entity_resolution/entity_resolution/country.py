import re

ALIASES = {
    "india":"IN", "in":"IN", "ind":"IN", "us":"US", "usa":"US",
    "united states":"US", "united states of america":"US", "gb":"GB", "uk":"GB",
    "united kingdom":"GB", "great britain":"GB", "de":"DE", "germany":"DE",
    "fr":"FR", "france":"FR", "ca":"CA", "canada":"CA", "au":"AU", "australia":"AU",
    "cn":"CN", "china":"CN", "chn":"CN", "jp":"JP", "japan":"JP", "jpn":"JP",
    "sg":"SG", "singapore":"SG", "sgp":"SG", "ae":"AE", "uae":"AE", "are":"AE",
    "br":"BR", "brazil":"BR", "bra":"BR", "mx":"MX", "mexico":"MX", "mex":"MX",
    "it":"IT", "italy":"IT", "ita":"IT", "es":"ES", "spain":"ES", "esp":"ES",
    "nl":"NL", "netherlands":"NL", "nld":"NL", "za":"ZA", "south africa":"ZA", "zaf":"ZA",
}

def normalize_country(value) -> str:
    s = re.sub(r"[._-]+", " ", str(value or "").strip().casefold())
    s = re.sub(r"\s+", " ", s).strip()
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
