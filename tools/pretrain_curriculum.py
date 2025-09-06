#!/usr/bin/env python3
"""
AURA Curriculum Pretraining System
Warm-starts brain regions with domain-specific knowledge from existing datasets
"""

import os
import json
import asyncio
import numpy as np
import trio
from glob import glob
from typing import Dict, Any, Iterable, List, Optional, Tuple
import sys

# Add project root to path
sys.path.insert(0, '/Volumes/Others2/AURA_GENESIS')

# Import AURA bootloader
from aura.system.bootloader import boot_aura_genesis, AuraBootConfig

# ---------- Dataset Plumbing ----------

def iter_jsonl_dir(root: str) -> Iterable[Dict[str, Any]]:
    """Iterate through all JSONL files in directory tree"""
    for path in sorted(glob(os.path.join(root, "**/*.jsonl"), recursive=True)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue

# Map folder name -> routing target + defaults
ROUTING_MAP = {
    "conversations": ("general_chat", {}),
    "empathy": ("amygdala_specialist", {}),
    "historical": ("historical_specialist", {"realm": "history"}),
    "historical_teacher": ("historical_specialist", {"realm": "history/teacher"}),
    "inventions": ("analytical_specialist", {"domain": "Science/Tech"}),
    "movie_annotated": ("amygdala_specialist", {}),
    "socratic": ("analytical_specialist", {"realm": "socratic"}),
}

def guess_bucket(filepath: str) -> Optional[str]:
    """Guess dataset bucket from file path"""
    for k in ROUTING_MAP:
        if f"/{k}/" in filepath.replace("\\", "/"):
            return k
    return None

def record_iter(dataset_root: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Yield (bucket, record) tuples from all datasets"""
    for path in sorted(glob(os.path.join(dataset_root, "*/*.jsonl"))):
        bucket = guess_bucket(path)
        if not bucket:
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: 
                    continue
                try:
                    rec = json.loads(line)
                    yield bucket, rec
                except Exception:
                    continue

# ---------- Adapters into your Network ----------

async def update_specialists(net, rec: Dict[str, Any], feats: np.ndarray):
    """Update domain, realm, and difficulty specialists"""
    meta = rec.get("metadata", {})
    domain = meta.get("domain")
    realm = rec.get("realm") or (meta.get("realm") if isinstance(meta.get("realm"), str) else None)
    diff = meta.get("difficulty")
    
    # Update domain classifier
    if domain and hasattr(net, 'specialists') and 'domain_classifier' in net.specialists:
        sp = net.specialists['domain_classifier']
        y = np.zeros(len(net.domains), dtype=np.float64)
        if domain in net.domains: 
            y[net.domains.index(domain)] = 1.0
        sp.nlms_head.update(feats.reshape(1, -1), y.reshape(1, -1))
    
    # Update realm classifier
    if realm and hasattr(net, 'specialists') and 'realm_classifier' in net.specialists:
        sp = net.specialists['realm_classifier']
        y = np.zeros(len(net.realms), dtype=np.float64)
        if realm in net.realms: 
            y[net.realms.index(realm)] = 1.0
        sp.nlms_head.update(feats.reshape(1, -1), y.reshape(1, -1))
    
    # Update difficulty regressor
    if isinstance(diff, (int, float)) and hasattr(net, 'specialists') and 'difficulty_regressor' in net.specialists:
        sp = net.specialists['difficulty_regressor']
        y = np.array([float(diff)], dtype=np.float64)
        sp.nlms_head.update(feats.reshape(1, -1), y.reshape(1, -1))

async def router_warmup(net, target_key: str, feats: np.ndarray, text: str = ""):
    """Family-target update (positive for chosen family, light negatives for siblings)"""
    router = net._thalamic_router
    fam = router.routing_neurons.get(target_key, [])
    neg_fams = [k for k in router.routing_neurons.keys() if k != target_key]
    
    # Positive push for target family
    for n in fam:
        if hasattr(n, "enable_attention") and n.enable_attention:
            await n.update_nlms_with_attention(feats, 1.0, text)
        else:
            await n.update_nlms(feats, 1.0)
    
    # Light negatives for sibling families
    for k in neg_fams:
        for n in router.routing_neurons[k]:
            await n.update_nlms(feats, 0.0)

async def amygdala_warmup(net, rec: Dict[str, Any], feats: np.ndarray):
    """Warm up amygdala with emotional conditioning"""
    am = net._amygdala
    pl = (rec.get("plutchik") or rec.get("metadata", {}).get("plutchik"))
    if not isinstance(pl, dict):
        return
    
    prim = pl.get("primary")
    outcome = "positive"
    if isinstance(prim, str):
        neg = {"anger", "fear", "disgust", "sadness"}
        outcome = "threatening" if prim.lower() in neg else "positive"
    
    # Use fear conditioning if available
    if hasattr(am, 'fear_conditioning'):
        await am.fear_conditioning(feats, outcome, event_data=rec.get("metadata"))
    else:
        # Fallback: direct NLMS update
        y = 1.0 if outcome == "positive" else 0.0
        for neuron in am.neurons[:10]:  # Update first 10 neurons
            await neuron.update_nlms(feats, y)

async def hippocampus_warmup(net, rec: Dict[str, Any], feats: np.ndarray):
    """Warm up hippocampus with memory encoding"""
    hip = net._hippocampus
    
    # Calculate salience based on content
    text = rec.get("text", "")
    salience = 0.5  # Default salience
    
    # Boost salience for important content
    if any(word in text.lower() for word in ["important", "significant", "key", "critical"]):
        salience = 0.8
    elif any(word in text.lower() for word in ["remember", "note", "recall"]):
        salience = 0.9
    
    # Encode memory if method available
    if hasattr(hip, 'encode_memory'):
        try:
            hip.encode_memory(feats, salience=salience)
        except Exception:
            pass
    else:
        # Fallback: direct neuron updates
        for neuron in hip.neurons[:20]:  # Update first 20 neurons
            await neuron.update_nlms(feats, salience)

# ---------- Main Pretraining Pass ----------

async def run_pretrain(dataset_root: str, passes: int = 1):
    """Run curriculum pretraining on all datasets"""
    print("🧠 AURA Curriculum Pretraining System")
    print("=" * 60)
    
    # Boot safely with existing sequence
    cfg = AuraBootConfig()
    cfg.offline_mode = True  # SBERT optional; Network handles offline
    boot = await boot_aura_genesis(cfg)
    net = boot.system_components['network']
    
    print(f"📚 Dataset root: {dataset_root}")
    print(f"🔄 Training passes: {passes}")
    print(f"🎯 Routing targets: {list(ROUTING_MAP.keys())}")
    
    # Track statistics
    stats = {
        'total_samples': 0,
        'by_bucket': {},
        'routing_updates': 0,
        'specialist_updates': 0,
        'amygdala_updates': 0,
        'hippocampus_updates': 0
    }
    
    for epoch in range(passes):
        print(f"\n🔄 Epoch {epoch + 1}/{passes}")
        seen = 0
        epoch_stats = {'by_bucket': {}}
        
        for bucket, rec in record_iter(dataset_root):
            text = rec.get("text", "")
            if not text:
                continue
                
            # Get features
            try:
                feats = net.get_features(text)
            except Exception as e:
                print(f"⚠️  Feature extraction failed for bucket {bucket}: {e}")
                continue
            
            # Update specialists
            try:
                await update_specialists(net, rec, feats)
                stats['specialist_updates'] += 1
            except Exception as e:
                print(f"⚠️  Specialist update failed: {e}")
            
            # Router warmup
            try:
                target, defaults = ROUTING_MAP[bucket]
                await router_warmup(net, target, feats, text)
                stats['routing_updates'] += 1
            except Exception as e:
                print(f"⚠️  Router warmup failed: {e}")
            
            # Amygdala warmup for emotional content
            if bucket in ("empathy", "movie_annotated"):
                try:
                    await amygdala_warmup(net, rec, feats)
                    stats['amygdala_updates'] += 1
                except Exception as e:
                    print(f"⚠️  Amygdala warmup failed: {e}")
            
            # Hippocampus warmup for historical content
            if bucket in ("historical", "historical_teacher"):
                try:
                    await hippocampus_warmup(net, rec, feats)
                    stats['hippocampus_updates'] += 1
                except Exception as e:
                    print(f"⚠️  Hippocampus warmup failed: {e}")
            
            # Track statistics
            seen += 1
            stats['total_samples'] += 1
            epoch_stats['by_bucket'][bucket] = epoch_stats['by_bucket'].get(bucket, 0) + 1
            stats['by_bucket'][bucket] = stats['by_bucket'].get(bucket, 0) + 1
            
            if seen % 1000 == 0:
                print(f"  📊 Processed {seen} samples...")
        
        print(f"✅ Epoch {epoch + 1} complete - {seen} samples processed")
        
        # Print epoch statistics
        print(f"  📈 Epoch {epoch + 1} bucket distribution:")
        for bucket, count in epoch_stats['by_bucket'].items():
            print(f"    {bucket}: {count} samples")
    
    # Print final statistics
    print(f"\n📊 Final Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Routing updates: {stats['routing_updates']}")
    print(f"  Specialist updates: {stats['specialist_updates']}")
    print(f"  Amygdala updates: {stats['amygdala_updates']}")
    print(f"  Hippocampus updates: {stats['hippocampus_updates']}")
    
    print(f"\n📈 Bucket distribution:")
    for bucket, count in stats['by_bucket'].items():
        print(f"  {bucket}: {count} samples")
    
    # Save weights snapshot
    try:
        print(f"\n💾 Saving weights snapshot...")
        await boot.shutdown_system()  # Graceful path persists via weights_io
        print(f"✅ Weights saved successfully!")
    except Exception as e:
        print(f"⚠️  Weight saving failed: {e}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AURA Curriculum Pretraining")
    parser.add_argument("--dataset-root", default="datasets", 
                       help="Root directory containing dataset folders")
    parser.add_argument("--passes", type=int, default=1,
                       help="Number of training passes")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_root):
        print(f"❌ Dataset root not found: {args.dataset_root}")
        print(f"   Expected structure:")
        print(f"   {args.dataset_root}/")
        print(f"   ├── conversations/")
        print(f"   ├── empathy/")
        print(f"   ├── historical/")
        print(f"   ├── historical_teacher/")
        print(f"   ├── inventions/")
        print(f"   ├── movie_annotated/")
        print(f"   └── socratic/")
        return 1
    
    trio.run(run_pretrain, args.dataset_root, args.passes)
    return 0

if __name__ == "__main__":
    sys.exit(main())
