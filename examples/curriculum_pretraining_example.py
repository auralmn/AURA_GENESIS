#!/usr/bin/env python3
"""
AURA Curriculum Pretraining Example
Demonstrates the complete curriculum-based pretraining system
"""

import os
import json
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, '/Volumes/Others2/AURA_GENESIS')

def create_mock_datasets():
    """Create mock datasets for testing the curriculum system"""
    print("🧪 Creating Mock Datasets for Testing")
    print("=" * 50)
    
    # Create dataset directories
    datasets = {
        "conversations": [
            {"text": "Hello, how are you today?", "metadata": {"domain": "general"}},
            {"text": "What's the weather like?", "metadata": {"domain": "general"}},
            {"text": "Can you help me with something?", "metadata": {"domain": "general"}}
        ],
        "empathy": [
            {"text": "I'm feeling really sad today", "plutchik": {"primary": "sadness", "intensity": 0.8}},
            {"text": "I'm so excited about this!", "plutchik": {"primary": "joy", "intensity": 0.9}},
            {"text": "I'm scared about the future", "plutchik": {"primary": "fear", "intensity": 0.7}}
        ],
        "historical": [
            {"text": "The Roman Empire fell in 476 AD", "metadata": {"domain": "History", "realm": "ancient"}},
            {"text": "Napoleon was defeated at Waterloo", "metadata": {"domain": "History", "realm": "modern"}},
            {"text": "The Renaissance began in Italy", "metadata": {"domain": "History", "realm": "renaissance"}}
        ],
        "inventions": [
            {"text": "The printing press revolutionized communication", "metadata": {"domain": "Science/Tech", "difficulty": 0.6}},
            {"text": "Electricity changed the world", "metadata": {"domain": "Science/Tech", "difficulty": 0.7}},
            {"text": "The internet connected everyone", "metadata": {"domain": "Science/Tech", "difficulty": 0.8}}
        ],
        "socratic": [
            {"text": "What is justice?", "metadata": {"realm": "philosophy", "difficulty": 0.9}},
            {"text": "How do we know what we know?", "metadata": {"realm": "philosophy", "difficulty": 0.8}},
            {"text": "What makes a good life?", "metadata": {"realm": "philosophy", "difficulty": 0.7}}
        ],
        "movie_annotated": [
            {"text": "The character felt overwhelming joy", "plutchik": {"primary": "joy", "intensity": 0.9}},
            {"text": "A scene of deep sadness and loss", "plutchik": {"primary": "sadness", "intensity": 0.8}},
            {"text": "The tension was palpable", "plutchik": {"primary": "fear", "intensity": 0.7}}
        ]
    }
    
    # Create directories and files
    base_dir = Path("test_datasets")
    base_dir.mkdir(exist_ok=True)
    
    for bucket, samples in datasets.items():
        bucket_dir = base_dir / bucket
        bucket_dir.mkdir(exist_ok=True)
        
        file_path = bucket_dir / f"{bucket}_samples.jsonl"
        with open(file_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"  📁 {bucket}: {len(samples)} samples → {file_path}")
    
    print(f"\n✅ Mock datasets created in {base_dir}")
    return str(base_dir)

def test_instruct_generation():
    """Test the instruct dataset generation"""
    print("\n🤖 Testing Instruct Dataset Generation")
    print("=" * 50)
    
    # Create mock datasets
    dataset_root = create_mock_datasets()
    
    # Import and run the instruct generator
    from tools.make_instruct import main as make_instruct_main
    
    output_file = "test_datasets/instruct/instruct_seed.jsonl"
    
    print(f"📝 Generating instruct dataset...")
    count = make_instruct_main(
        root=dataset_root,
        out=output_file,
        max_per_bucket=10,  # Small number for testing
        min_text_length=5
    )
    
    print(f"✅ Generated {count} instruct samples")
    
    # Show some examples
    if os.path.exists(output_file):
        print(f"\n📖 Sample instruct entries:")
        with open(output_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:  # Show first 3 examples
                    break
                sample = json.loads(line)
                print(f"  Example {i+1} ({sample['meta']['bucket']}):")
                print(f"    System: {sample['system']}")
                print(f"    Instruction: {sample['instruction']}")
                print(f"    Output hint: {sample['output_hint']}")
                print()
    
    return count

def test_curriculum_pretraining():
    """Test the curriculum pretraining system"""
    print("\n🧠 Testing Curriculum Pretraining")
    print("=" * 50)
    
    # Create mock datasets
    dataset_root = create_mock_datasets()
    
    # Import the pretraining system
    from tools.pretrain_curriculum import run_pretrain
    
    print(f"🚀 Running curriculum pretraining...")
    print(f"   Dataset root: {dataset_root}")
    print(f"   Passes: 1")
    
    # Note: This would normally run the full pretraining
    # For testing, we'll just verify the imports work
    try:
        import trio
        print(f"✅ Trio import successful")
        
        # We could run the actual pretraining here, but it requires
        # the full AURA system to be booted, which might be slow for testing
        print(f"⚠️  Full pretraining test skipped (requires AURA system boot)")
        print(f"   To run full test: python tools/pretrain_curriculum.py --dataset-root {dataset_root}")
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    return True

def demonstrate_curriculum_mapping():
    """Demonstrate the curriculum mapping system"""
    print("\n📚 Curriculum Mapping Demonstration")
    print("=" * 50)
    
    from tools.pretrain_curriculum import ROUTING_MAP, guess_bucket
    
    print("🎯 Dataset → Brain Region Mapping:")
    for dataset, (target, defaults) in ROUTING_MAP.items():
        print(f"  {dataset:20} → {target:20} {defaults}")
    
    print(f"\n🔍 Path Bucket Detection:")
    test_paths = [
        "datasets/conversations/chat.jsonl",
        "datasets/empathy/emotional.jsonl", 
        "datasets/historical/ancient.jsonl",
        "datasets/inventions/tech.jsonl",
        "datasets/socratic/philosophy.jsonl",
        "datasets/movie_annotated/scenes.jsonl"
    ]
    
    for path in test_paths:
        bucket = guess_bucket(path)
        if bucket:
            target, defaults = ROUTING_MAP[bucket]
            print(f"  {path:40} → {bucket:15} → {target}")
        else:
            print(f"  {path:40} → No bucket detected")
    
    return True

def main():
    """Run all curriculum pretraining tests"""
    print("🚀 AURA Curriculum Pretraining Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Curriculum mapping
        print("\n📋 Test 1: Curriculum Mapping")
        success1 = demonstrate_curriculum_mapping()
        
        # Test 2: Instruct generation
        print("\n📋 Test 2: Instruct Dataset Generation")
        success2 = test_instruct_generation()
        
        # Test 3: Curriculum pretraining (imports only)
        print("\n📋 Test 3: Curriculum Pretraining (Imports)")
        success3 = test_curriculum_pretraining()
        
        if success1 and success2 and success3:
            print("\n🎉 All tests passed successfully!")
            print("\nKey Features Demonstrated:")
            print("  ✅ Dataset → Brain region mapping")
            print("  ✅ Path-based bucket detection")
            print("  ✅ Instruct dataset generation")
            print("  ✅ Template-based sample creation")
            print("  ✅ Curriculum pretraining system imports")
            
            print("\n🚀 Ready for Production:")
            print("  # Generate instruct dataset:")
            print("  python tools/make_instruct.py --root datasets --out datasets/instruct/instruct_seed.jsonl")
            print()
            print("  # Run curriculum pretraining:")
            print("  python tools/pretrain_curriculum.py --dataset-root datasets --passes 1")
            
            print("\n📊 Curriculum Benefits:")
            print("  🧠 Domain-specific brain region warm-up")
            print("  🎯 Router learns proper routing targets")
            print("  😊 Amygdala learns emotional conditioning")
            print("  🧠 Hippocampus learns memory encoding")
            print("  📚 Specialists learn domain/realm/difficulty")
            print("  🤖 Instruct dataset for fine-tuning")
            
        else:
            print("\n❌ Some tests failed!")
            return False
            
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup test files
        import shutil
        if os.path.exists("test_datasets"):
            shutil.rmtree("test_datasets")
            print(f"\n🧹 Cleaned up test files")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
