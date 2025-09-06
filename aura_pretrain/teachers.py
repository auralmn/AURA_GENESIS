from __future__ import annotations
from typing import Dict, Any

def emotion_label(record: Dict[str, Any]) -> str:
    pl = record.get("plutchik") or {}
    p = pl.get("primary", "neutral")
    if isinstance(p, list):
        p = p[0] if p else "neutral"
    return str(p)

def router_teacher(record: Dict[str, Any]) -> str:
    # Use realm, domain, SVC verbs & keywords to pick a specialist
    text = (record.get("text") or "").lower()
    domain = str((record.get("metadata") or {}).get("domain","")).lower()
    realm  = str(record.get("realm","")).lower()
    svc = ((record.get("metadata") or {}).get("svc")) or {}
    verb = str(svc.get("verb","")).lower()

    hist_kw = ["ancient","medieval","dynasty","empire","revolution","world war","napoleon","rome","egypt","pharaoh","renaissance"]
    emo_kw  = ["feel","afraid","scared","angry","excited","love","hate","sad","joy","anxious"]
    ana_kw  = ["compare","explain","why","because","effect","cause","analyze","tradeoff"]

    if realm.startswith("history") or any(k in text for k in hist_kw):
        return "historical_specialist"
    if any(k in text for k in emo_kw):
        return "amygdala_specialist"
    if any(k in text for k in ana_kw):
        return "analytical_specialist"
    if verb in {"define","explain","compare"}:
        return "analytical_specialist"
    # Could route memory/recall if dataset marks cross-turn refs; default to general
    return "general_chat"

def hippocampus_salience(record: Dict[str, Any]) -> float:
    # Simple salience: mix difficulty, length, and emotion intensity (if present)
    text = record.get("text","") or ""
    diff = float((record.get("metadata") or {}).get("difficulty", 0.5))
    pl = record.get("plutchik") or {}
    inten = float(pl.get("intensity", 0.5))
    L = len(text)
    score = 0.4*diff + 0.4*inten + 0.2*float(L > 160)
    if "!" in text:
        score += 0.1
    return float(max(0.0, min(1.0, score)))
