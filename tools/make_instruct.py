#!/usr/bin/env python3
"""
AURA Instruct Dataset Generator
Auto-builds a tiny instruct set from existing datasets for fine-tuning
"""

import os
import json
import random
from glob import glob
from typing import Dict, Any, List

TEMPLATES = {
    "historical": (
        "Answer as a helpful historian.", 
        "Explain the significance of: {text}", 
        "A concise, sourced explanation."
    ),
    "socratic": (
        "Be a Socratic tutor.",
        "Ask one probing question about: {text}",
        "One question, no answer."
    ),
    "empathy": (
        "Be supportive and empathetic.",
        "Respond to: {text}",
        "A short, validating reply."
    ),
    "inventions": (
        "Think like an analyst.",
        "What are the trade-offs in: {text}?",
        "Bulleted pros/cons."
    ),
    "conversations": (
        "Be a helpful conversational AI.",
        "Continue this conversation: {text}",
        "A natural, engaging response."
    ),
    "movie_annotated": (
        "Be an empathetic emotional intelligence system.",
        "Analyze the emotional content: {text}",
        "A brief emotional analysis with empathy."
    )
}

def bucket_from_path(p: str) -> str:
    """Extract bucket name from file path"""
    for k in TEMPLATES.keys():
        if f"/{k}/" in p.replace("\\", "/"):
            return k
    return None

def create_instruct_sample(rec: Dict[str, Any], bucket: str) -> Dict[str, Any]:
    """Create an instruct sample from a record"""
    sys_prompt, instruction_template, output_hint = TEMPLATES[bucket]
    text = rec.get("text", "")
    
    if not text:
        return None
    
    # Create instruction
    instruction = instruction_template.format(text=text)
    
    # Add metadata
    meta = {
        "bucket": bucket,
        "id": rec.get("id"),
        "original_text": text[:100] + "..." if len(text) > 100 else text
    }
    
    # Add domain/realm info if available
    if "metadata" in rec:
        meta["domain"] = rec["metadata"].get("domain")
        meta["realm"] = rec["metadata"].get("realm")
        meta["difficulty"] = rec["metadata"].get("difficulty")
    
    # Add emotional info if available
    if "plutchik" in rec:
        meta["emotion"] = rec["plutchik"].get("primary")
        meta["intensity"] = rec["plutchik"].get("intensity")
    
    return {
        "system": sys_prompt,
        "instruction": instruction,
        "input": "",
        "output_hint": output_hint,
        "meta": meta
    }

def main(root: str = "datasets", out: str = "datasets/instruct/instruct_seed.jsonl", 
         max_per_bucket: int = 2000, min_text_length: int = 10):
    """Generate instruct dataset from existing datasets"""
    
    print("🤖 AURA Instruct Dataset Generator")
    print("=" * 50)
    print(f"📁 Dataset root: {root}")
    print(f"📝 Output file: {out}")
    print(f"📊 Max per bucket: {max_per_bucket}")
    print(f"📏 Min text length: {min_text_length}")
    
    # Create output directory
    os.makedirs(os.path.dirname(out), exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(1337)
    
    # Collect all samples
    items = []
    bucket_counts = {}
    
    print(f"\n📚 Scanning datasets...")
    
    for path in glob(os.path.join(root, "*/*.jsonl")):
        bucket = bucket_from_path(path)
        if not bucket:
            continue
            
        print(f"  📂 Processing {bucket} from {os.path.basename(path)}")
        bucket_counts[bucket] = 0
        
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    rec = json.loads(line)
                except Exception as e:
                    print(f"    ⚠️  JSON parse error at line {line_num}: {e}")
                    continue
                
                text = rec.get("text", "")
                if not text or len(text) < min_text_length:
                    continue
                
                # Create instruct sample
                sample = create_instruct_sample(rec, bucket)
                if sample:
                    items.append(sample)
                    bucket_counts[bucket] += 1
                
                # Progress indicator
                if line_num % 1000 == 0:
                    print(f"    📊 Processed {line_num} lines, {bucket_counts[bucket]} samples")
    
    print(f"\n📊 Raw collection complete:")
    for bucket, count in bucket_counts.items():
        print(f"  {bucket}: {count} samples")
    
    # Shuffle and limit samples per bucket
    print(f"\n🎲 Shuffling and limiting samples...")
    random.shuffle(items)
    
    by_bucket = {}
    for item in items:
        bucket = item["meta"]["bucket"]
        by_bucket.setdefault(bucket, []).append(item)
    
    # Limit samples per bucket
    kept = []
    for bucket, samples in by_bucket.items():
        limited = samples[:max_per_bucket]
        kept.extend(limited)
        print(f"  {bucket}: {len(limited)} samples (from {len(samples)})")
    
    # Shuffle final dataset
    random.shuffle(kept)
    
    # Write output file
    print(f"\n💾 Writing instruct dataset...")
    with open(out, "w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ Wrote {len(kept)} instruct samples → {out}")
    
    # Print sample statistics
    print(f"\n📈 Final dataset statistics:")
    print(f"  Total samples: {len(kept)}")
    print(f"  Average text length: {sum(len(item['meta']['original_text']) for item in kept) / len(kept):.1f}")
    
    # Show sample distribution
    final_buckets = {}
    for item in kept:
        bucket = item["meta"]["bucket"]
        final_buckets[bucket] = final_buckets.get(bucket, 0) + 1
    
    print(f"  Sample distribution:")
    for bucket, count in sorted(final_buckets.items()):
        print(f"    {bucket}: {count} samples")
    
    # Show example samples
    print(f"\n📝 Example samples:")
    for i, item in enumerate(kept[:3]):
        print(f"  Sample {i+1} ({item['meta']['bucket']}):")
        print(f"    System: {item['system']}")
        print(f"    Instruction: {item['instruction'][:100]}...")
        print(f"    Output hint: {item['output_hint']}")
        print()
    
    return len(kept)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate AURA instruct dataset")
    parser.add_argument("--root", default="datasets", 
                       help="Root directory containing dataset folders")
    parser.add_argument("--out", default="datasets/instruct/instruct_seed.jsonl",
                       help="Output file path")
    parser.add_argument("--max-per-bucket", type=int, default=2000,
                       help="Maximum samples per bucket")
    parser.add_argument("--min-text-length", type=int, default=10,
                       help="Minimum text length for inclusion")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.root):
        print(f"❌ Dataset root not found: {args.root}")
        print(f"   Expected structure:")
        print(f"   {args.root}/")
        print(f"   ├── conversations/")
        print(f"   ├── empathy/")
        print(f"   ├── historical/")
        print(f"   ├── historical_teacher/")
        print(f"   ├── inventions/")
        print(f"   ├── movie_annotated/")
        print(f"   └── socratic/")
        exit(1)
    
    count = main(args.root, args.out, args.max_per_bucket, args.min_text_length)
    print(f"\n🎉 Successfully generated {count} instruct samples!")
