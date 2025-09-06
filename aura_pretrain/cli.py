from __future__ import annotations
import argparse, asyncio, trio
from .warm_start import warm_start_from_dataset

def main():
    p = argparse.ArgumentParser("AURA offline warm-start")
    p.add_argument("--data", required=True, help="Path to JSONL dataset")
    p.add_argument("--out", default="svc_nlms_weights", help="Output directory for weights/heads")
    p.add_argument("--sbert-device", default=None, help="e.g., cpu | cuda | mps")
    p.add_argument("--emotion-dim", type=int, default=192, help="Sine embedding length for AmygdalaRelay")
    p.add_argument("--router-alpha", type=float, default=0.7, help="Blend SBERT vs extras projection")
    p.add_argument("--moe-topk", type=int, default=2, help="Top-K experts for Liquid-MoE")
    p.add_argument("--hippo-sample", type=int, default=512, help="How many examples to warm-start hippocampus")
    args = p.parse_args()

    async def run():
        res = await warm_start_from_dataset(
            args.data,
            out_dir=args.out,
            sbert_device=args.sbert_device,
            emotion_dim=args.emotion_dim,
            router_alpha=args.router_alpha,
            moe_topk=args.moe_topk,
            hippocampus_sample=args.hippo_sample
        )
        print("\n=== Warm-start summary ===")
        for k,v in res.items():
            print(f"{k}: {v if not isinstance(v, dict) else '[dict]'}")

    trio.run(run)

if __name__ == "__main__":
    main()
