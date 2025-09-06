from __future__ import annotations
import hashlib
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # optional

# -------- SBERT loader (optional) ----------
def maybe_load_sbert(model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer(model_name, device=device)
    except Exception:
        return None

# -------- helpers ----------
def _hash_to_index(s: str, dim: int, salt: str = "") -> int:
    h = hashlib.sha1((salt + s).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % dim

def _onehot_bow(values: List[str], dim: int, salt: str) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for x in values:
        if not x:
            continue
        v[_hash_to_index(str(x).lower(), dim, salt)] += 1.0
    n = float(v.sum())
    if n > 0:
        v /= n
    return v

def _svc_bucket(record: Dict[str, Any], dim: int = 64) -> np.ndarray:
    # Hash Subject/Verb/Complement into a small bucket
    svc = ((record.get("metadata") or {}).get("svc")) or {}
    sub = str(svc.get("subject", ""))
    verb = str(svc.get("verb", ""))
    comp = str(svc.get("complement", ""))
    return _onehot_bow([sub, verb, comp], dim, salt="svc")

def _linguistic_scalar_feats(record: Dict[str, Any]) -> np.ndarray:
    text = record.get("text", "") or ""
    toks = (record.get("linguistic_features") or {}).get("tokens", [])
    n_tok = len(toks) if isinstance(toks, list) else len(text.split())
    n_punct = sum(text.count(p) for p in [".", ",", "!", "?", ";", ":"])
    avg_tok = (sum(len(t.get("text", "")) for t in toks) / max(1, n_tok)) if toks else (len(text)/max(1, n_tok))
    diff = float((record.get("metadata") or {}).get("difficulty", 0.5))
    return np.array([len(text)/200.0, n_tok/100.0, n_punct/20.0, avg_tok/10.0, diff], dtype=np.float32)

def _domain_realm_bucket(record: Dict[str, Any], dim: int = 32) -> np.ndarray:
    domain = str((record.get("metadata") or {}).get("domain", "")).lower()
    realm  = str(record.get("realm","")).lower()
    parts = [p for p in [domain, realm] if p]
    return _onehot_bow(parts, dim, salt="dr")

# -------- main feature builders ----------
def build_router_features(
    record: Dict[str, Any],
    sbert_model=None,
    alpha: float = 0.7,
    proj_seed: int = 1337,
    out_dim: int = 384
) -> np.ndarray:
    """
    384-D features for Thalamus/Router/Hippocampus:
      f = alpha * SBERT + (1-alpha) * RP(extras)
    extras includes SVC hash, domain/realm hash, and simple scalars.
    """
    # SBERT (384) if available else zeros
    if sbert_model is not None:
        sbert_vec = sbert_model.encode([record.get("text", "") or ""], normalize_embeddings=True)[0]
        sbert_vec = np.asarray(sbert_vec, dtype=np.float32)
    else:
        sbert_vec = np.zeros(out_dim, dtype=np.float32)

    # extras -> random projection -> 384
    extras = np.concatenate([
        _svc_bucket(record, 64),
        _domain_realm_bucket(record, 32),
        _linguistic_scalar_feats(record),
    ]).astype(np.float32)

    rng = np.random.default_rng(proj_seed)
    R = rng.standard_normal((extras.shape[0], out_dim)).astype(np.float32) / np.sqrt(extras.shape[0])
    proj = extras @ R  # (384,)

    f = alpha * sbert_vec + (1.0 - alpha) * proj
    # L2 normalize for stability
    n = float(np.linalg.norm(f) + 1e-8)
    return (f / n).astype(np.float32)

def build_emotion_sine_embedding(
    record: Dict[str, Any],
    length: int = 192,
    params: Optional[Dict[str, Dict[str, float]]] = None
) -> np.ndarray:
    """
    Sine-only embedding to stay compatible with AmygdalaRelay (EmotionClassifier).
    Uses primary/secondary from record['plutchik'] if present, otherwise neutral.
    """
    pl = record.get("plutchik") or {}
    primary = pl.get("primary", "neutral")
    if isinstance(primary, list): primary = primary[0] if primary else "neutral"
    secondary = pl.get("secondary", None)
    if isinstance(secondary, list): secondary = secondary[0] if secondary else None
    intensity = float(pl.get("intensity", 1.0))

    # default params
    if params is None:
        params = {}
    def getp(lbl, idx):
        if lbl not in params:
            params[lbl] = {"freq": 1.5 + 0.3*idx, "amp": 0.7, "phase": 0.5 + 0.4*idx}
        return params[lbl]

    p1 = getp(str(primary), 0)
    t = np.linspace(0, 2*np.pi, length, dtype=np.float32)
    emb = (p1["amp"]*intensity*np.sin(p1["freq"]*t + p1["phase"])).astype(np.float32)

    if secondary and secondary != primary:
        p2 = getp(str(secondary), 1)
        emb += (0.5 * p2["amp"] * intensity * np.sin(p2["freq"]*t + p2["phase"])).astype(np.float32)
    return emb.astype(np.float32)
