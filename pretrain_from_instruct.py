# pretrain_aura_from_instruct.py
#!/usr/bin/env python3
import argparse, json, os, math
import numpy as np
import trio
import faulthandler, signal, sys, time, os, torch
from numpy import DataLoader
faulthandler.register(signal.SIGUSR1)              # kill -USR1 <pid> -> dump stacks
faulthandler.dump_traceback_later(60, repeat=True) # dump every 60s if no progress

os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
os.environ.setdefault("PYTHONUNBUFFERED","1")
if torch.cuda.is_available():
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING","1")  # surface kernel errors

torch.device('cpu')
# Import your AURA modules
from aura.core.network import Network

def difficulty_from_lengths(prompt: str, response: str) -> float:
    # quick scalar: length + punctuation density
    p = prompt.strip()
    r = response.strip()
    L = len(p.split()) + 0.5*len(r.split())
    dens = (p.count("?")+p.count(":")+p.count(",")) / max(1,len(p))
    x = 0.02*L + 5*dens
    return float(max(0.0, min(1.0, 1- math.exp(-x))))

async def pretrain_one(network: Network, prompt: str, response: str):
    # features for routing/regions come from the prompt
    feats = network.get_features(prompt).astype(np.float64)

    # === Thalamic router (Liquid-MoE + multi-channel spike attention) ===
    router = network._thalamic_router
    decision = router.analyze_conversation_intent(prompt, feats)
    plan = router.route_conversation(decision, prompt, feats)
    outcome = {  # warm-up assumes good pairs
        "user_satisfaction": 0.95,
        "response_quality": 0.95,
    }
    await router.adaptive_routing_update_with_attention(plan, outcome, feats, prompt)

    # === Optional: Amygdala warm-up ===
    # Treat non-toxic, non-angry text as mildly positive exposure
    try:
        amyg = network._amygdala
        amyg.process_emotional_salience(feats, event_data={"title":"instruct_pair"})
        await amyg.fear_conditioning(feats, outcome='positive')
    except Exception:
        pass

    # === Optional: Hippocampus encoding ===
    try:
        hip = network._hippocampus
        hip.encode(feats)
        hip.encode_memory(feats)
    except Exception:
        pass

async def pretrain_stream(jsonl_path: str,
                          save_dir: str,
                          limit: int | None = None,
                          save_every: int = 5000,
                          offline: bool = False):
    # Build network (SBERT enabled if offline=False)
    net = Network(
        neuron_count=1000,      # keep smaller for warm-up, your big configs will still work
        features=384,
        enable_span=False,
        offline=offline,        # set False to use SBERT if available
        features_mode='sbert',  # 'sbert'|'phasor'|'combined'
        weights_dir=save_dir,
    )
    await net.init_weights()

    # training loop (streaming)
    seen = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        async with trio.open_nursery() as nursery:
            for line in f:
                if limit and seen >= limit:
                    break
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                txt = str(row.get("text","")).strip()
                # reuse parser from the inspector (inline here)
                # expects "<human>:" ... "<bot>:"
                h_idx = txt.lower().find("<human>:")
                b_idx = txt.lower().find("<bot>:")
                if h_idx < 0 or b_idx < 0: 
                    continue
                prompt = txt[h_idx+8:b_idx].strip()
                response = txt[b_idx+6:].strip()
                if not prompt or not response:
                    continue

                # (Optional) attach pseudo labels to feed Network.train_on_data later
                data = {
                    "text": prompt,
                    "domain": "general",
                    "realm": "conversational",
                    "difficulty": difficulty_from_lengths(prompt, response)
                }
                # You can also call: await net.train_on_data([data]) if you want specialist heads warm-up.

                # Router/regions warm-up
                nursery.start_soon(pretrain_one, net, prompt, response)
                seen += 1

                if seen % save_every == 0:
                    try:
                        net.save_weights(save_dir)
                        print(f"💾 saved weights @ {save_dir} (seen={seen})")
                    except Exception:
                        pass
            # let remaining tasks finish
    # final save
    try:
        net.save_weights(save_dir)
        print(f"✅ final save -> {save_dir} (seen={seen})")
    except Exception:
        pass

def main():
    ap = argparse.ArgumentParser("pretrain_aura_from_instruct")
    ap.add_argument("in_jsonl", help="clean or raw instruct JSONL with 'text' field")
    ap.add_argument("--save-dir", default="svc_nlms_weights_pretrain")
    ap.add_argument("--limit", type=int, help="max examples")
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--offline", action="store_true", help="disable SBERT & stay offline")
    args = ap.parse_args()
    trio.run(pretrain_stream, args.in_jsonl, args.save_dir, args.limit, args.save_every, args.offline)

if __name__ == "__main__":
    main()
