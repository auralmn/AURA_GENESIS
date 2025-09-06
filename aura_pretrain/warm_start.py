from __future__ import annotations
import json, numpy as np, asyncio, trio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .features import maybe_load_sbert, build_router_features, build_emotion_sine_embedding
from .teachers import emotion_label, router_teacher, hippocampus_salience

# Boot + weights I/O (support both your structured package and local dev)
try:
    from aura.system.bootloader import boot_aura_genesis, AuraBootConfig
except Exception:
    try:
        from aura.boot.bootloader import boot_aura_genesis, AuraBootConfig
    except Exception:
        from bootloader import boot_aura_genesis, AuraBootConfig  # fallback

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

# ---- tiny linear softmax for AmygdalaRelay compatibility ------------
def _train_linear_softmax(X: np.ndarray, y_idx: np.ndarray, lr=5e-3, epochs=15, wd=0.0) -> Tuple[np.ndarray, np.ndarray]:
    # W in R^{D x C}, b in R^{C}
    D = X.shape[1]; C = int(y_idx.max())+1
    rng = np.random.default_rng(7)
    W = (rng.standard_normal((D, C))/np.sqrt(D)).astype(np.float32)
    b = np.zeros((C,), dtype=np.float32)

    for _ in range(epochs):
        # simple full-batch gradient descent
        Z = X @ W + b  # [N,C]
        Z -= Z.max(axis=1, keepdims=True)
        expz = np.exp(Z); P = expz / np.clip(expz.sum(axis=1, keepdims=True), 1e-8, None)
        # CE grad
        Y = np.zeros_like(P); Y[np.arange(X.shape[0]), y_idx] = 1.0
        G = (P - Y) / X.shape[0] + wd * (W / X.shape[0])
        dW = X.T @ G
        db = G.sum(axis=0)
        W -= lr * dW.astype(np.float32)
        b -= lr * db.astype(np.float32)
    return W.astype(np.float32), b.astype(np.float32)

async def warm_start_from_dataset(
    dataset_path: str,
    *,
    out_dir: str = "svc_nlms_weights",
    sbert_device: Optional[str] = None,
    use_attention: bool = True,
    emotion_dim: int = 192,
    router_alpha: float = 0.7,
    moe_topk: int = 2,
    hippocampus_sample: int = 512
) -> Dict[str, Any]:
    rows = _load_jsonl(dataset_path)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 0) SBERT (optional)
    sbert = maybe_load_sbert(device=sbert_device)

    # 1) Build features
    router_feats = np.stack([
        build_router_features(r, sbert_model=sbert, alpha=router_alpha, out_dim=384) for r in rows
    ], axis=0)

    # 2) Prepare emotion labels + sine embeddings for amygdala pretrain
    labels = sorted({emotion_label(r) for r in rows})
    L2I = {e:i for i,e in enumerate(labels)}
    X_em = np.stack([build_emotion_sine_embedding(r, length=emotion_dim) for r in rows], axis=0)
    y_em = np.array([L2I[emotion_label(r)] for r in rows], dtype=np.int64)

    # 3) Boot minimal system (safe boot already in your Bootloader)
    cfg = AuraBootConfig()
    cfg.offline_mode = True
    boot = await boot_aura_genesis(cfg)
    net = boot.system_components["network"]
    router = net._thalamic_router
    hip = net._hippocampus

    # sanity: make sure router MoE exists + attention is enabled if desired
    try:
        if router.moe and moe_topk:
            router.moe.top_k = int(moe_topk)
    except Exception:
        pass

    # 4) Amygdala linear head (Export for AmygdalaRelay)
    W, b = _train_linear_softmax(X_em, y_em, lr=5e-3, epochs=15)
    np.save(Path(out_dir) / "emotion_classifier_W.npy", W)
    np.save(Path(out_dir) / "emotion_classifier_b.npy", b)
    json.dump({
        "LABEL_TO_IDX": L2I,
        "IDX_TO_LABEL": {i:k for k,i in L2I.items()}
    }, open(Path(out_dir) / "emotion_classifier_labels.json", "w"))

    # 5) Router pretrain (Liquid-MoE + multi-channel attention)
    for r, x in zip(rows, router_feats):
        txt = r.get("text","")
        # MoE-based analysis (uses attention internally to compute gating gain)
        decision = router.analyze_conversation_intent(txt, x)
        plan = router.route_conversation(decision, txt, x)
        # Teacher says where we *should* have gone
        teacher = router_teacher(r)
        ok = (teacher == plan["primary_specialist"])
        outcome = {
            "user_satisfaction": 1.0 if ok else 0.2,
            "response_quality":  1.0 if ok else 0.3
        }
        # This will update MoE (gating), generate attention telemetry, and NLMS adapters if present
        await router.adaptive_routing_update_with_attention(plan, outcome, x, txt)

    # 6) Hippocampus salience warm-start (subset for speed)
    idx = np.arange(router_feats.shape[0])
    if hippocampus_sample and hippocampus_sample < len(idx):
        idx = np.random.default_rng(7).choice(idx, size=hippocampus_sample, replace=False)
    for i in idx:
        s = float(hippocampus_salience(rows[i]))
        x = router_feats[i]
        # light warm-up across a slice of neurons
        for n in hip.neurons[:1024]:  # cap for speed in huge hippo
            await n.update_nlms(x, s)

    # 7) Save network (try your tools.weights_io, else fall back to Network.save_weights)
    saved = {}
    try:
        from tools.weights_io import save_network_weights
        saved = save_network_weights(net, out_dir)
    except Exception:
        try:
            net.save_weights(out_dir)
            saved = {"fallback": True}
        except Exception:
            saved = {"saved": False}

    # Return a compact summary
    try:
        attn_summary = router.get_attention_summary()
        moe_stats = router.get_moe_stats()
    except Exception:
        attn_summary, moe_stats = {}, {}

    return {
        "out_dir": out_dir,
        "labels": labels,
        "emotion_dim": emotion_dim,
        "saved": saved,
        "attention": attn_summary,
        "moe": moe_stats,
    }
