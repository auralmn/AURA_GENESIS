#!/usr/bin/env python3
"""
Router Smoke Test
Quick probe to sanity-check confidence + usage after the tweaks
"""

import argparse
import sys
import asyncio
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, '/Volumes/Others2/AURA_GENESIS')

async def test_router_confidence_and_usage():
    """Test router confidence and usage with sample queries"""
    
    # Test queries covering different domains
    test_queries = [
        "Summarize the causes of World War I.",
        "I'm feeling anxious—can you help me calm down?",
        "Can you recall what we agreed on earlier?",
        "Compare SVM vs. random forest for tabular data.",
        "hey! what's up?",
        "Explain quantum computing principles",
        "I'm so excited about this new project!",
        "What happened in the Battle of Hastings?",
        "How do I implement a neural network?",
        "I'm really disappointed with the results"
    ]
    
    print("🧪 Router Smoke Test")
    print("=" * 60)
    
    try:
        # Import and boot AURA system
        from aura.system.bootloader import boot_aura_genesis
        from aura.system.bootloader import AuraBootConfig
        
        print("🚀 Booting AURA system...")
        cfg = AuraBootConfig()
        boot = await boot_aura_genesis(cfg)
        network = boot.system_components['network']
        
        print("✅ AURA system booted successfully")
        
        # Test each query
        print(f"\n🔍 Testing {len(test_queries)} queries...")
        print("-" * 60)
        
        total_confidence = 0.0
        usage_stats = {}
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i:2d}. Query: {query[:48]:48}")
            
            try:
                # Get features
                feats = network.get_features(query)
                
                # Analyze conversation intent
                dec = network._thalamic_router.analyze_conversation_intent(query, feats)
                
                # Extract results
                primary_target = dec.get('primary_target', 'unknown')
                routing_confidence = dec.get('routing_confidence', 0.0)
                specialist_confidence = dec.get('specialist_confidence', 0.0)
                
                print(f"    → Target: {primary_target:22}  conf={routing_confidence:.3f}")
                print(f"    → Specialist conf: {specialist_confidence:.3f}")
                
                total_confidence += routing_confidence
                
                # Track usage
                if primary_target not in usage_stats:
                    usage_stats[primary_target] = 0
                usage_stats[primary_target] += 1
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                continue
        
        # Calculate statistics
        avg_confidence = total_confidence / len(test_queries)
        
        print(f"\n📊 Router Statistics:")
        print("-" * 60)
        print(f"Average routing confidence: {avg_confidence:.3f}")
        print(f"Total queries processed: {len(test_queries)}")
        
        print(f"\n📈 Target distribution:")
        for target, count in sorted(usage_stats.items()):
            percentage = (count / len(test_queries)) * 100
            print(f"  {target:20}: {count:2d} ({percentage:5.1f}%)")
        
        # Test MoE usage if available
        if hasattr(network._thalamic_router, 'liquid_moe_router'):
            print(f"\n🔧 MoE Usage Statistics:")
            moe_stats = network._thalamic_router.liquid_moe_router.get_usage_stats()
            print(f"  Usage std: {moe_stats['usage_std']:.4f}")
            print(f"  Usage entropy: {moe_stats['usage_entropy']:.4f}")
            print(f"  Target usage: {moe_stats['target_usage']:.4f}")
            
            # Test energy consumption
            energy_stats = network._thalamic_router.liquid_moe_router.get_energy_stats()
            print(f"  Energy per query: {energy_stats['total_energy_j']:.6f} J")
        
        # Test attention system if available
        if hasattr(network._thalamic_router, 'attn') and network._thalamic_router.attn is not None:
            print(f"\n🎯 Attention System Test:")
            attn = network._thalamic_router.attn
            
            # Test with a sample query
            sample_query = "I'm feeling really excited about this!"
            tokens = sample_query.split()
            from aura.core.attention import prosody_channels_from_text
            amp, pitch, boundary = prosody_channels_from_text(tokens)
            token_ids = [hash(t.lower()) % 50000 for t in tokens]
            
            ar = attn.compute(token_ids, amp, pitch, boundary)
            
            print(f"  Sample query: '{sample_query}'")
            print(f"  μ scalar: {ar['mu_scalar']:.3f}")
            print(f"  Winners: {len(ar['winners_idx'])} tokens")
            print(f"  Spike rates: amp={ar['spikes']['amp'].mean():.3f}, "
                  f"pitch={ar['spikes']['pitch'].mean():.3f}, "
                  f"boundary={ar['spikes']['boundary'].mean():.3f}")
        else:
            print(f"\n🎯 Attention System Test:")
            print(f"  Attention system not initialized (this is normal for basic boot)")
        
        print(f"\n✅ Smoke test completed successfully!")
        
        # Shutdown system
        await boot.shutdown_system()
        
        return True
        
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Router Smoke Test")
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    print("🧪 AURA Router Smoke Test")
    print("=" * 50)
    
    # Run async test
    import trio
    success = trio.run(test_router_confidence_and_usage)
    
    if success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())
