#!/usr/bin/env python3
"""
AURA Pretraining on JSONL Chat Data (human/bot pairs)

Goal
----
Warm up routing (Liquid-MoE + NLMS WTA groups), thalamus/hippocampus touchpoints,
and attention telemetry using existing AURA core.

Input
-----
JSONL where each line has at least a `text` field and contains
`<human>:` / `<bot>:` style turns (single or multi-turn in one line is OK).

Example line:
{"text": "<human>: Hello!\n<bot>: Hi there."}

Usage
-----
python aura_pretrain_chat_jsonl.py \
  --files data/chat/*.jsonl \
  --aura-root . \
  --epochs 1 \
  --max-rows 0 \
  --features-mode sbert \
  --enable-attention \
  --save-dir weights_pretrain

Notes
-----
* Uses SBERT if available (Network.features_mode='sbert'); otherwise falls back to zeros.
* For each pair, we:
    1) Extract features from the HUMAN text
    2) Ask router to analyze + route
    3) Update router via MoE (with attention gain) + classic NLMS family update
    4) Touch hippocampus (encode) to build some memory traces
* Conversation outcome is a weak heuristic from the BOT response (length/sanity),
  just to prime routing; replace with proper reward when available.
"""

from __future__ import annotations
import argparse, sys, os, json, math, glob, re, time
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Dict, Any, Tuple, Optional

import numpy as np
import trio

# ------------------------------ Utils ------------------------------

def info(msg: str) -> None:
    print(f"[AURA-PRETRAIN] {msg}")

TURN_RE = re.compile(r"\s*<(human|bot)>\s*:\s*", re.IGNORECASE)

@dataclass
class Pair:
    human: str
    bot: str
    meta: Dict[str, Any]

# Robust split that tolerates extra whitespace and multi-turn in a single string
# Returns a list of Pair(human, bot) extracted in order.

def extract_pairs(text: str) -> List[Pair]:
    if not text:
        return []
    # Normalize common alternatives seen in datasets
    t = text.replace("<Human>", "<human>").replace("<Bot>", "<bot>")
    t = re.sub(r"\[/?INST\]", "", t, flags=re.IGNORECASE)
    t = t.replace("Assistant:", "<bot>:").replace("Human:", "<human>:")

    pieces: List[Tuple[str, str]] = []  # (role, content)
    idx = 0
    for m in TURN_RE.finditer(t):
        role = m.group(1).lower()
        start = m.end()
        if idx < start:
            # previous segment content is from idx..m.start(), but we only keep after a tag
            pass
        # find next tag to close this segment
        nxt = TURN_RE.search(t, pos=start)
        seg = t[start: nxt.start()] if nxt else t[start:]
        pieces.append((role, seg.strip()))
        if not nxt:
            break
        idx = nxt.start()

    pairs: List[Pair] = []
    cur_h: Optional[str] = None
    for role, seg in pieces:
        if role == 'human':
            # if stray human without closing previous, reset
            if cur_h is not None and seg:
                # drop unfinished
                pass
            cur_h = seg
        else:  # bot
            if cur_h is not None:
                pairs.append(Pair(cur_h, seg, meta={}))
                cur_h = None
            else:
                # stray bot; ignore
                pass
    return pairs


def iter_jsonl(paths: List[str]) -> Iterator[Dict[str, Any]]:
    for p in paths:
        for path in glob.glob(p):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue


def proxy_outcome(bot: str) -> Dict[str, float]:
    # Quick-and-dirty success proxy
    if not bot:
        return {"user_satisfaction": 0.2, "response_quality": 0.2}
    L = len(bot)
    words = len(bot.split())
    bad = any(x in bot.lower() for x in ["i can't", "i cannot", "sorry", "unable", "not allowed"])
    quality = 0.2 + 0.8 * (min(1.0, words / 40.0))
    if bad:
        quality *= 0.6
    satisfaction = 0.3 + 0.7 * (min(1.0, L / 200.0))
    if bad:
        satisfaction *= 0.7
    return {
        "user_satisfaction": float(np.clip(satisfaction, 0.0, 1.0)),
        "response_quality": float(np.clip(quality, 0.0, 1.0)),
    }

# ------------------------------ Trainer ------------------------------

async def build_network(features: int, features_mode: str, enable_attention: bool) -> Any:
    # Lazy import with flexible root
    from aura.core.network import Network
    from aura.core.thalamic_router import ThalamicConversationRouter

    net = Network(
        neuron_count=1000,           # light footprint for pretraining; your boot can be larger
        features=features,
        enable_span=False,
        offline=False,               # allow SBERT if installed
        features_mode=features_mode,
    )

    # Swap router to enable attention + fresh MoE if requested
    if enable_attention:
        net._thalamic_router = ThalamicConversationRouter(
            neuron_count=60,
            features=features,
            input_dim=features,
            routing_confidence_threshold=0.6,
            enable_attention=True,
        )

    # Init weights for submodules
    await net.init_weights()
    return net

@dataclass
class Stats:
    seen_pairs: int = 0
    updates: int = 0
    avg_quality: float = 0.0
    avg_conf: float = 0.0

async def train_epoch(net: Any, records: Iterator[Dict[str, Any]], limit: int = 0) -> Stats:
    st = Stats()
    router = net._thalamic_router
    hip = net._hippocampus

    tick = time.time()
    for rec in records:
        text = rec.get('text') or rec.get('conversation') or ''
        if not text:
            continue
        pairs = extract_pairs(text)
        if not pairs:
            continue

        for i, p in enumerate(pairs):
            st.seen_pairs += 1
            # Features from HUMAN side
            x = net.get_features(p.human)
            # Analyze + plan
            rd = router.analyze_conversation_intent(p.human, x)
            plan = router.route_conversation(rd, p.human, x)

            # Heuristic outcome from BOT
            outcome = proxy_outcome(p.bot)
            st.avg_quality += outcome['response_quality']
            st.avg_conf += float(rd['routing_confidence'])

            # Update MoE + NLMS groups
            await router.adaptive_routing_update_with_attention(plan, outcome, x, p.human)
            await router.adaptive_routing_update(plan, outcome, x)
            st.updates += 1

            # Touch hippocampus memory trace
            try:
                hip.encode_memory(x)
            except Exception:
                pass

            if limit and st.seen_pairs >= limit:
                break
        if limit and st.seen_pairs >= limit:
            break

        # Light heartbeat
        if st.seen_pairs % 1000 == 0:
            dt = time.time() - tick
            q = st.avg_quality / max(1, st.updates)
            c = st.avg_conf / max(1, st.updates)
            info(f"{st.seen_pairs} pairs | avg_quality={q:.3f} avg_conf={c:.3f} | {st.updates} updates | {dt:.1f}s")
            tick = time.time()

    # finalize moving averages
    if st.updates:
        st.avg_quality /= st.updates
        st.avg_conf   /= st.updates
    return st

# ------------------------------ CLI ------------------------------

async def amain(args) -> None:
    # Wire imports path
    if args.aura_root:
        sys.path.insert(0, os.path.abspath(args.aura_root))

    # Late imports after path surgery
    try:
        from aura.core.thalamic_router import ThalamicConversationRouter  # noqa: F401
    except Exception as e:
        info("Could not import AURA core. Use --aura-root to point to your repo root.")
        raise

    # Build the network
    net = await build_network(features=args.features, features_mode=args.features_mode, enable_attention=args.enable_attention)

    # Training loop
    files = args.files or []
    if not files:
        raise SystemExit("--files is required (one or more JSONL globs)")

    total_stats = Stats()
    for epoch in range(1, args.epochs + 1):
        info(f"Epoch {epoch}/{args.epochs} — reading JSONL…")
        rec_it = iter_jsonl(files)
        stats = await train_epoch(net, rec_it, limit=args.max_rows)
        info(f"Epoch {epoch} done: pairs={stats.seen_pairs} updates={stats.updates} avg_quality={stats.avg_quality:.3f} avg_conf={stats.avg_conf:.3f}")

        # Save weights each epoch
        try:
            os.makedirs(args.save_dir, exist_ok=True)
            net.save_weights(args.save_dir)
            info(f"Weights saved to {args.save_dir}")
        except Exception as e:
            info(f"Save failed: {e}")

        # MoE + Attention telemetry snapshot
        try:
            moe = net._thalamic_router.get_moe_stats()
            attn = net._thalamic_router.get_attention_summary()
            info(f"MoE stats: {json.dumps(moe)}")
            info(f"Attention summary: {json.dumps(attn)}")
        except Exception:
            pass

    info("Pretraining complete.")


def main() -> None:
    ap = argparse.ArgumentParser(description="AURA pretraining on JSONL chat data")
    ap.add_argument('--files', nargs='+', required=True, help='One or more JSONL paths or globs')
    ap.add_argument('--aura-root', default='.', help='Path to AURA repo root (so imports like core.* resolve)')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--max-rows', type=int, default=0, help='Limit total pairs processed (0 = all)')
    ap.add_argument('--features', type=int, default=384)
    ap.add_argument('--features-mode', choices=['sbert','phasor','combined'], default='sbert')
    ap.add_argument('--enable-attention', action='store_true')
    ap.add_argument('--save-dir', default='weights_pretrain')
    args = ap.parse_args()

    trio.run(amain, args)


if __name__ == '__main__':
    main()
