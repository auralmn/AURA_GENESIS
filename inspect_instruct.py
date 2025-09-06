# inspect_instruct_jsonl.py
#!/usr/bin/env python3
import argparse, json, re, sys
from collections import Counter
import torch
torch.device('mps')

PAIR_RE = re.compile(
    r'(?s)\s*<human>:\s*(.*?)\s*<bot>:\s*(.*)\s*$'
)

def parse_text_field(s: str):
    m = PAIR_RE.match(s.strip())
    if not m:
        return None, None
    prompt = m.group(1).strip()
    response = m.group(2).strip()
    return prompt, response

def main():
    ap = argparse.ArgumentParser("inspect_instruct_jsonl")
    ap.add_argument("in_jsonl", help="input JSONL with a 'text' field")
    ap.add_argument("--out", help="optional cleaned JSONL {prompt,response,text}")
    ap.add_argument("--sample", type=int, default=5)
    args = ap.parse_args()

    n, ok, bad = 0, 0, 0
    lengths = []
    out_f = open(args.out, "w", encoding="utf-8") if args.out else None

    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            n += 1
            try:
                row = json.loads(line)
            except Exception:
                bad += 1
                continue
            text = row.get("text", "")
            prompt, response = parse_text_field(text)
            if prompt is None:
                bad += 1
                continue
            ok += 1
            lengths.append((len(prompt.split()), len(response.split())))
            if out_f:
                out_f.write(json.dumps({
                    "prompt": prompt, "response": response, "text": text
                }, ensure_ascii=False) + "\n")
            if ok <= args.sample:
                print(f"\n--- sample #{ok} ---")
                print("PROMPT:\n", prompt[:500])
                print("RESPONSE:\n", response[:500])

    if out_f: out_f.close()

    if lengths:
        import numpy as np
        pw = np.array([p for p,_ in lengths])
        rw = np.array([r for _,r in lengths])
        def stats(x): 
            return f"min={int(x.min())} p50={int(np.median(x))} p90={int(np.percentile(x,90))} max={int(x.max())}"
        print("\n== Stats ==")
        print(f"total lines: {n} | parsed pairs: {ok} | failed: {bad}")
        print(f"prompt words:  {stats(pw)}")
        print(f"response words:{stats(rw)}")
    else:
        print("\nNo valid pairs found.", file=sys.stderr)

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    import numpy as np, json

    m = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'mps')
    B = 512
    buf, feats = [], []
    with open('datasets/instruction_tuning/instruct_55k.jsonl','r') as f:
        for line in f:
            text = json.loads(line)['text']
            buf.append(text)
            if len(buf) == B:
                feats.append(m.encode(buf, batch_size=B, convert_to_numpy=True, show_progress_bar=False))
                buf.clear()
    if buf: feats.append(m.encode(buf, batch_size=len(buf), convert_to_numpy=True, show_progress_bar=False))
    X = np.vstack(feats).astype('float32')
    np.save('instruct_55k_sbert.npy', X)  # -> load in Dataloader, no encoder in the loop

