import re
import unicodedata

_WS = re.compile(r"\s+", re.UNICODE)
_PUNCT = re.compile(r"[\u0000-\u001f\u007f]", re.UNICODE)

def normalize_text(value) -> str:
    if value is None:
        return ""
    s = unicodedata.normalize("NFKC", str(value)).casefold()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s

def normalize_name(value) -> str:
    return normalize_text(value)

def normalize_address(value) -> str:
    # Keep digits and Unicode letters; only normalize separators/spacing.
    return normalize_text(value)

def normalize_entity(row: dict) -> dict:
    return {**row, "business_name_norm": normalize_name(row.get("business_name")),
            "business_address_norm": normalize_address(row.get("business_address"))}
