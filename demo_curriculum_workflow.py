#!/usr/bin/env python3
"""
AURA Curriculum Workflow Demonstration
Shows the complete curriculum pretraining and instruct generation workflow
"""

import os
import sys
import json
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, '/Volumes/Others2/AURA_GENESIS')

def create_demo_datasets():
    """Create realistic demo datasets for the workflow"""
    print("🎬 Creating Demo Datasets")
    print("=" * 40)
    
    # Create realistic datasets
    datasets = {
        "conversations": [
            {
                "text": "Hello! How can I help you today?",
                "metadata": {"domain": "general", "difficulty": 0.2}
            },
            {
                "text": "What's the weather like in your area?",
                "metadata": {"domain": "general", "difficulty": 0.3}
            },
            {
                "text": "I'm looking for a good restaurant recommendation.",
                "metadata": {"domain": "general", "difficulty": 0.4}
            }
        ],
        "empathy": [
            {
                "text": "I'm feeling really overwhelmed with work lately",
                "plutchik": {"primary": "sadness", "intensity": 0.7, "secondary": "fear"},
                "metadata": {"domain": "Psychology", "difficulty": 0.6}
            },
            {
                "text": "I'm so excited about my upcoming vacation!",
                "plutchik": {"primary": "joy", "intensity": 0.9, "secondary": "anticipation"},
                "metadata": {"domain": "Psychology", "difficulty": 0.3}
            },
            {
                "text": "I'm really angry about what happened at work today",
                "plutchik": {"primary": "anger", "intensity": 0.8, "secondary": "disgust"},
                "metadata": {"domain": "Psychology", "difficulty": 0.7}
            }
        ],
        "historical": [
            {
                "text": "The fall of the Roman Empire in 476 AD marked the end of ancient history",
                "metadata": {"domain": "History", "realm": "ancient", "difficulty": 0.8}
            },
            {
                "text": "Napoleon's defeat at Waterloo in 1815 ended the Napoleonic Wars",
                "metadata": {"domain": "History", "realm": "modern", "difficulty": 0.7}
            },
            {
                "text": "The Renaissance began in Italy in the 14th century",
                "metadata": {"domain": "History", "realm": "renaissance", "difficulty": 0.6}
            }
        ],
        "inventions": [
            {
                "text": "The printing press revolutionized communication and education",
                "metadata": {"domain": "Science/Tech", "realm": "invention", "difficulty": 0.7}
            },
            {
                "text": "Electricity transformed daily life and industry",
                "metadata": {"domain": "Science/Tech", "realm": "invention", "difficulty": 0.6}
            },
            {
                "text": "The internet connected the world and changed everything",
                "metadata": {"domain": "Science/Tech", "realm": "invention", "difficulty": 0.8}
            }
        ],
        "socratic": [
            {
                "text": "What is the nature of justice in society?",
                "metadata": {"realm": "philosophy", "difficulty": 0.9}
            },
            {
                "text": "How do we distinguish between knowledge and belief?",
                "metadata": {"realm": "philosophy", "difficulty": 0.8}
            },
            {
                "text": "What constitutes a good life?",
                "metadata": {"realm": "philosophy", "difficulty": 0.7}
            }
        ],
        "movie_annotated": [
            {
                "text": "The character's face showed overwhelming joy as they reunited",
                "plutchik": {"primary": "joy", "intensity": 0.9},
                "metadata": {"domain": "Entertainment", "difficulty": 0.4}
            },
            {
                "text": "A scene of deep sadness and loss brought tears to the audience",
                "plutchik": {"primary": "sadness", "intensity": 0.8},
                "metadata": {"domain": "Entertainment", "difficulty": 0.5}
            },
            {
                "text": "The tension was palpable as the hero faced the final challenge",
                "plutchik": {"primary": "fear", "intensity": 0.7},
                "metadata": {"domain": "Entertainment", "difficulty": 0.6}
            }
        ]
    }
    
    # Create directories and files
    base_dir = Path("demo_datasets")
    base_dir.mkdir(exist_ok=True)
    
    total_samples = 0
    for bucket, samples in datasets.items():
        bucket_dir = base_dir / bucket
        bucket_dir.mkdir(exist_ok=True)
        
        file_path = bucket_dir / f"{bucket}_demo.jsonl"
        with open(file_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"  📁 {bucket:15} → {len(samples):2} samples → {file_path}")
        total_samples += len(samples)
    
    print(f"\n✅ Created {total_samples} demo samples in {base_dir}")
    return str(base_dir)

def run_instruct_generation(dataset_root):
    """Run the instruct dataset generation"""
    print(f"\n🤖 Generating Instruct Dataset")
    print("=" * 40)
    
    from tools.make_instruct import main as make_instruct_main
    
    output_file = f"{dataset_root}/instruct/instruct_seed.jsonl"
    
    print(f"📝 Processing datasets from: {dataset_root}")
    print(f"📄 Output file: {output_file}")
    
    count = make_instruct_main(
        root=dataset_root,
        out=output_file,
        max_per_bucket=5,  # Small number for demo
        min_text_length=5
    )
    
    print(f"✅ Generated {count} instruct samples")
    
    # Show some examples
    if os.path.exists(output_file):
        print(f"\n📖 Sample instruct entries:")
        with open(output_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 2:  # Show first 2 examples
                    break
                sample = json.loads(line)
                print(f"  Example {i+1} ({sample['meta']['bucket']}):")
                print(f"    System: {sample['system']}")
                print(f"    Instruction: {sample['instruction']}")
                print(f"    Output hint: {sample['output_hint']}")
                print()
    
    return count

def demonstrate_curriculum_mapping():
    """Show the curriculum mapping system"""
    print(f"\n📚 Curriculum Mapping System")
    print("=" * 40)
    
    from tools.pretrain_curriculum import ROUTING_MAP
    
    print("🎯 Dataset → Brain Region Mapping:")
    for dataset, (target, defaults) in ROUTING_MAP.items():
        print(f"  {dataset:20} → {target:20} {defaults}")
    
    print(f"\n🧠 Brain Region Training:")
    print(f"  🎯 Router: Learns to route conversations to correct specialists")
    print(f"  😊 Amygdala: Learns emotional conditioning from Plutchik data")
    print(f"  🧠 Hippocampus: Learns memory encoding from historical content")
    print(f"  📚 Specialists: Learn domain, realm, and difficulty classification")
    
    return True

def show_workflow_benefits():
    """Show the benefits of the curriculum workflow"""
    print(f"\n🚀 Curriculum Workflow Benefits")
    print("=" * 40)
    
    benefits = [
        "🧠 Domain-Specific Warm-up: Each brain region learns its specialty",
        "🎯 Smart Routing: Router learns to send conversations to the right place",
        "😊 Emotional Intelligence: Amygdala learns from real emotional data",
        "🧠 Memory Systems: Hippocampus learns to encode important information",
        "📚 Specialist Training: Domain, realm, and difficulty classifiers get real data",
        "🤖 Instruct Dataset: Ready-to-use training data for fine-tuning",
        "⚡ Fast Boot: System starts with knowledge instead of random weights",
        "🎭 Multi-Modal: Handles text, emotions, domains, and difficulty levels",
        "🔄 Scalable: Easy to add new datasets and brain regions",
        "📊 Monitored: Full telemetry and health monitoring during training"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")
    
    return True

def show_production_usage():
    """Show how to use the system in production"""
    print(f"\n🏭 Production Usage")
    print("=" * 40)
    
    commands = [
        "# 1. Generate instruct dataset from your data",
        "python tools/make_instruct.py --root datasets --out datasets/instruct/instruct_seed.jsonl",
        "",
        "# 2. Run curriculum pretraining",
        "python tools/pretrain_curriculum.py --dataset-root datasets --passes 1",
        "",
        "# 3. Start online system",
        "python -m aura.system.bootloader",
        "",
        "# 4. Monitor and control (in REPL)",
        "status                    # Check system health",
        "router.stats             # View routing statistics", 
        "attn.tail 10             # Show recent attention events",
        "shutdown                 # Graceful shutdown"
    ]
    
    for cmd in commands:
        print(f"  {cmd}")
    
    return True

def main():
    """Run the complete curriculum workflow demonstration"""
    print("🎬 AURA Curriculum Workflow Demonstration")
    print("=" * 60)
    
    try:
        # Step 1: Create demo datasets
        dataset_root = create_demo_datasets()
        
        # Step 2: Show curriculum mapping
        demonstrate_curriculum_mapping()
        
        # Step 3: Generate instruct dataset
        instruct_count = run_instruct_generation(dataset_root)
        
        # Step 4: Show workflow benefits
        show_workflow_benefits()
        
        # Step 5: Show production usage
        show_production_usage()
        
        print(f"\n🎉 Curriculum Workflow Demonstration Complete!")
        print(f"=" * 60)
        print(f"📊 Summary:")
        print(f"  • Created {len(os.listdir(dataset_root))} dataset buckets")
        print(f"  • Generated {instruct_count} instruct samples")
        print(f"  • Demonstrated curriculum mapping system")
        print(f"  • Showed production usage patterns")
        
        print(f"\n🚀 Next Steps:")
        print(f"  1. Replace demo datasets with your real data")
        print(f"  2. Run: python tools/make_instruct.py --root your_datasets")
        print(f"  3. Run: python tools/pretrain_curriculum.py --dataset-root your_datasets")
        print(f"  4. Run: python -m aura.system.bootloader")
        print(f"  5. Enjoy your intelligent AURA system! 🧠✨")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup demo files
        import shutil
        if os.path.exists("demo_datasets"):
            shutil.rmtree("demo_datasets")
            print(f"\n🧹 Cleaned up demo files")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
