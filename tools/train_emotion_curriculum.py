#!/usr/bin/env python3
"""
AURA Emotion Curriculum Training
Combines GoEmotions training with curriculum pretraining for comprehensive emotion intelligence
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to Python path
sys.path.insert(0, '/Volumes/Others2/AURA_GENESIS')

def run_go_emotions_training(go_emotions_files: List[str], models_dir: str, 
                           max_items: int = None) -> bool:
    """Run GoEmotions emotion classifier training"""
    print("😊 Step 1: GoEmotions Emotion Classifier Training")
    print("=" * 60)
    
    # Import the GoEmotions trainer
    from train_go_emotions import main as train_go_emotions_main
    
    # Build command line arguments
    sys.argv = [
        'train_go_emotions.py',
        '--files'] + go_emotions_files + [
        '--models-dir', models_dir
    ]
    
    if max_items:
        sys.argv.extend(['--max-items', str(max_items)])
    
    try:
        result = train_go_emotions_main()
        return result == 0
    except Exception as e:
        print(f"❌ GoEmotions training failed: {e}")
        return False

def run_curriculum_pretraining(dataset_root: str, passes: int = 1) -> bool:
    """Run curriculum pretraining on all datasets"""
    print("\n🧠 Step 2: Curriculum Pretraining")
    print("=" * 60)
    
    # Import the curriculum pretraining system
    from pretrain_curriculum import main as pretrain_curriculum_main
    
    # Build command line arguments
    sys.argv = [
        'pretrain_curriculum.py',
        '--dataset-root', dataset_root,
        '--passes', str(passes)
    ]
    
    try:
        result = pretrain_curriculum_main()
        return result == 0
    except Exception as e:
        print(f"❌ Curriculum pretraining failed: {e}")
        return False

def run_instruct_generation(dataset_root: str, output_file: str, max_per_bucket: int = 2000) -> bool:
    """Generate instruct dataset from existing datasets"""
    print("\n🤖 Step 3: Instruct Dataset Generation")
    print("=" * 60)
    
    # Import the instruct generator
    from make_instruct import main as make_instruct_main
    
    try:
        count = make_instruct_main(
            root=dataset_root,
            out=output_file,
            max_per_bucket=max_per_bucket,
            min_text_length=10
        )
        print(f"✅ Generated {count} instruct samples")
        return True
    except Exception as e:
        print(f"❌ Instruct generation failed: {e}")
        return False

def check_go_emotions_availability() -> List[str]:
    """Check for available GoEmotions datasets"""
    go_emotions_paths = []
    
    # Check common GoEmotions locations
    potential_paths = [
        "datasets/go_emotions/train.jsonl",
        "datasets/go_emotions/validation.jsonl", 
        "datasets/go_emotions/test.jsonl",
        "datasets/pretrain/go_emotions.jsonl"
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            go_emotions_paths.append(path)
    
    return go_emotions_paths

def main():
    parser = argparse.ArgumentParser(description="AURA Emotion Curriculum Training Pipeline")
    parser.add_argument('--dataset-root', default='datasets', 
                       help='Root directory containing dataset folders')
    parser.add_argument('--models-dir', default='models',
                       help='Output directory for trained models')
    parser.add_argument('--go-emotions-files', nargs='+', default=None,
                       help='GoEmotions JSONL files (auto-detected if not specified)')
    parser.add_argument('--curriculum-passes', type=int, default=1,
                       help='Passes for curriculum pretraining')
    parser.add_argument('--max-emotion-items', type=int, default=None,
                       help='Maximum GoEmotions items to process')
    parser.add_argument('--skip-emotion', action='store_true',
                       help='Skip GoEmotions emotion classifier training')
    parser.add_argument('--skip-curriculum', action='store_true',
                       help='Skip curriculum pretraining')
    parser.add_argument('--skip-instruct', action='store_true',
                       help='Skip instruct dataset generation')
    parser.add_argument('--instruct-output', default='datasets/instruct/instruct_seed.jsonl',
                       help='Output file for instruct dataset')
    parser.add_argument('--max-instruct-per-bucket', type=int, default=2000,
                       help='Maximum instruct samples per bucket')
    
    args = parser.parse_args()
    
    print("🎭 AURA Emotion Curriculum Training Pipeline")
    print("=" * 70)
    print(f"📁 Dataset root: {args.dataset_root}")
    print(f"💾 Models directory: {args.models_dir}")
    
    success_count = 0
    total_steps = 0
    
    # Step 1: GoEmotions Emotion Classifier Training
    if not args.skip_emotion:
        total_steps += 1
        
        # Auto-detect GoEmotions files if not specified
        go_emotions_files = args.go_emotions_files
        if not go_emotions_files:
            go_emotions_files = check_go_emotions_availability()
        
        if not go_emotions_files:
            print("⚠️  No GoEmotions files found. Skipping emotion classifier training.")
            print("   Expected locations:")
            print("   - datasets/go_emotions/train.jsonl")
            print("   - datasets/go_emotions/validation.jsonl")
            print("   - datasets/pretrain/go_emotions.jsonl")
        else:
            print(f"📚 Found GoEmotions files: {go_emotions_files}")
            if run_go_emotions_training(
                go_emotions_files, 
                args.models_dir, 
                args.max_emotion_items
            ):
                success_count += 1
                print("✅ GoEmotions training completed successfully")
            else:
                print("❌ GoEmotions training failed")
    else:
        print("⏭️  Skipping GoEmotions emotion classifier training")
    
    # Step 2: Curriculum Pretraining
    if not args.skip_curriculum:
        total_steps += 1
        if run_curriculum_pretraining(args.dataset_root, args.curriculum_passes):
            success_count += 1
            print("✅ Curriculum pretraining completed successfully")
        else:
            print("❌ Curriculum pretraining failed")
    else:
        print("⏭️  Skipping curriculum pretraining")
    
    # Step 3: Instruct Dataset Generation
    if not args.skip_instruct:
        total_steps += 1
        if run_instruct_generation(
            args.dataset_root, 
            args.instruct_output,
            args.max_instruct_per_bucket
        ):
            success_count += 1
            print("✅ Instruct dataset generation completed successfully")
        else:
            print("❌ Instruct dataset generation failed")
    else:
        print("⏭️  Skipping instruct dataset generation")
    
    # Summary
    print(f"\n🎉 Training Pipeline Complete!")
    print("=" * 70)
    print(f"📊 Results: {success_count}/{total_steps} steps completed successfully")
    
    if success_count == total_steps:
        print("✅ All training steps completed successfully!")
        print("\n🚀 Next steps:")
        print("   1. Start the AURA system: python -m aura.system.bootloader")
        print("   2. The system will automatically load the trained emotion classifier")
        print("   3. Use the curriculum-pretrained brain regions for intelligent processing")
        print("   4. Fine-tune with the generated instruct dataset if needed")
        return 0
    else:
        print("⚠️  Some training steps failed. Check the logs above for details.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
