from __future__ import annotations

import re
import unicodedata
from typing import Any, List, Tuple

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    RAPIDFUZZ_AVAILABLE = False


def norm_text(x: Any) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return ""
    s = str(x).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("&", " and ").replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9\s\-_]", " ", s)
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fuzzy_match(query: str, choices: List[str], threshold: float = 60.0) -> List[Tuple[str, float]]:
    q = norm_text(query)
    if not q or not choices:
        return []
    if RAPIDFUZZ_AVAILABLE:
        hits = process.extract(q, [norm_text(c) for c in choices], scorer=fuzz.WRatio, limit=10)
        out = [(choices[idx], float(score)) for (_, score, idx) in hits if score >= threshold]
        return out[:10]
    # fallback: token Jaccard
    qt = set(q.split())
    scored = []
    for c in choices:
        ct = set(norm_text(c).split())
        inter = len(qt & ct)
        union = len(qt | ct)
        score = 100.0 * (inter / union) if union else 0.0
        if score >= threshold:
            scored.append((c, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:10]
