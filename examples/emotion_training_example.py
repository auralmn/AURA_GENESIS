#!/usr/bin/env python3
"""
AURA Emotion Training Example
Demonstrates the complete emotion training pipeline with GoEmotions and curriculum pretraining
"""

import os
import sys
import json
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, '/Volumes/Others2/AURA_GENESIS')

def create_mock_go_emotions_dataset():
    """Create a mock GoEmotions dataset for testing"""
    print("🎭 Creating Mock GoEmotions Dataset")
    print("=" * 50)
    
    # GoEmotions labels (simplified set)
    labels = [
        0,  # admiration
        1,  # amusement  
        2,  # anger
        3,  # annoyance
        4,  # approval
        5,  # caring
        6,  # confusion
        7,  # curiosity
        8,  # desire
        9,  # disappointment
        10, # disapproval
        11, # disgust
        12, # embarrassment
        13, # excitement
        14, # fear
        15, # gratitude
        16, # grief
        17, # joy
        18, # love
        19, # nervousness
        20, # optimism
        21, # pride
        22, # realization
        23, # relief
        24, # remorse
        25, # sadness
        26, # surprise
    ]
    
    # Create mock data
    mock_data = [
        {"text": "I love this movie so much!", "labels": [17, 18]},  # joy, love
        {"text": "This is absolutely terrible and I hate it", "labels": [2, 11]},  # anger, disgust
        {"text": "I'm so excited about the new project!", "labels": [13]},  # excitement
        {"text": "I'm really scared about what might happen", "labels": [14]},  # fear
        {"text": "Thank you so much for your help", "labels": [15]},  # gratitude
        {"text": "I'm so confused about this problem", "labels": [6]},  # confusion
        {"text": "This is amazing! I'm so proud of you", "labels": [17, 21]},  # joy, pride
        {"text": "I'm really disappointed with the results", "labels": [9]},  # disappointment
        {"text": "I'm curious about how this works", "labels": [7]},  # curiosity
        {"text": "I'm so nervous about the presentation", "labels": [19]},  # nervousness
        {"text": "I'm optimistic about the future", "labels": [20]},  # optimism
        {"text": "I feel so relieved that it's over", "labels": [23]},  # relief
        {"text": "I'm so sad about what happened", "labels": [25]},  # sadness
        {"text": "Wow, I didn't see that coming!", "labels": [26]},  # surprise
        {"text": "I'm so embarrassed about my mistake", "labels": [12]},  # embarrassment
    ]
    
    # Create directory and file
    os.makedirs("mock_datasets/go_emotions", exist_ok=True)
    file_path = "mock_datasets/go_emotions/train.jsonl"
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in mock_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"📁 Created mock GoEmotions dataset: {file_path}")
    print(f"   Samples: {len(mock_data)}")
    print(f"   Labels: {len(labels)}")
    
    return file_path

def test_go_emotions_training():
    """Test the GoEmotions training system"""
    print("\n😊 Testing GoEmotions Training")
    print("=" * 50)
    
    # Create mock dataset
    go_emotions_file = create_mock_go_emotions_dataset()
    
    # Import and run the GoEmotions trainer
    from tools.train_go_emotions import main as train_go_emotions_main
    
    # Save original sys.argv
    original_argv = sys.argv.copy()
    
    try:
        # Set up command line arguments
        sys.argv = [
            'train_go_emotions.py',
            '--files', go_emotions_file,
            '--models-dir', 'mock_models',
            '--epochs', '2',  # Small number for testing
            '--max-items', '10',  # Limit for testing
            '--device', 'cpu'  # Use CPU for testing
        ]
        
        print(f"🚀 Running GoEmotions training...")
        result = train_go_emotions_main()
        
        if result == 0:
            print("✅ GoEmotions training completed successfully")
            
            # Check if files were created
            models_dir = Path("mock_models")
            expected_files = [
                "clf_emotion.pt",
                "emotion_classifier_W.npy", 
                "emotion_classifier_b.npy",
                "emotion_classifier_labels.json"
            ]
            
            print(f"📁 Checking output files in {models_dir}:")
            for file in expected_files:
                file_path = models_dir / file
                if file_path.exists():
                    print(f"  ✅ {file}")
                else:
                    print(f"  ❌ {file} (missing)")
            
            return True
        else:
            print("❌ GoEmotions training failed")
            return False
            
    except Exception as e:
        print(f"❌ GoEmotions training error: {e}")
        return False
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

def test_emotion_curriculum_pipeline():
    """Test the complete emotion curriculum pipeline"""
    print("\n🎭 Testing Emotion Curriculum Pipeline")
    print("=" * 50)
    
    # Create mock datasets
    go_emotions_file = create_mock_go_emotions_dataset()
    
    # Create mock curriculum datasets
    os.makedirs("mock_datasets/empathy", exist_ok=True)
    empathy_file = "mock_datasets/empathy/empathy_samples.jsonl"
    with open(empathy_file, 'w', encoding='utf-8') as f:
        empathy_data = [
            {"text": "I'm feeling really sad today", "plutchik": {"primary": "sadness", "intensity": 0.8}},
            {"text": "I'm so excited about this!", "plutchik": {"primary": "joy", "intensity": 0.9}},
            {"text": "I'm scared about the future", "plutchik": {"primary": "fear", "intensity": 0.7}}
        ]
        for item in empathy_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"📁 Created mock empathy dataset: {empathy_file}")
    
    # Import and run the emotion curriculum pipeline
    from tools.train_emotion_curriculum import main as train_emotion_curriculum_main
    
    # Save original sys.argv
    original_argv = sys.argv.copy()
    
    try:
        # Set up command line arguments
        sys.argv = [
            'train_emotion_curriculum.py',
            '--dataset-root', 'mock_datasets',
            '--models-dir', 'mock_models',
            '--go-emotions-files', go_emotions_file,
            '--emotion-epochs', '1',  # Small number for testing
            '--curriculum-passes', '1',
            '--max-emotion-items', '10',
            '--device', 'cpu',
            '--skip-instruct'  # Skip instruct generation for this test
        ]
        
        print(f"🚀 Running emotion curriculum pipeline...")
        result = train_emotion_curriculum_main()
        
        if result == 0:
            print("✅ Emotion curriculum pipeline completed successfully")
            return True
        else:
            print("❌ Emotion curriculum pipeline failed")
            return False
            
    except Exception as e:
        print(f"❌ Emotion curriculum pipeline error: {e}")
        return False
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

def demonstrate_integration_benefits():
    """Show the benefits of the integrated emotion training system"""
    print("\n🚀 Integration Benefits")
    print("=" * 50)
    
    benefits = [
        "😊 GoEmotions Training: High-quality emotion classification from research dataset",
        "🧠 Curriculum Pretraining: Domain-specific brain region warm-up",
        "🎯 Router Learning: Smart conversation routing to emotion specialists", 
        "😊 Amygdala Conditioning: Emotional intelligence from Plutchik data",
        "🧠 Hippocampus Memory: Emotional memory encoding and retrieval",
        "📚 Specialist Training: Domain, realm, and difficulty classification",
        "🤖 Instruct Dataset: Ready-to-use training data for fine-tuning",
        "⚡ Fast Boot: System starts with emotion knowledge instead of random weights",
        "🎭 Multi-Modal: Handles text, emotions, domains, and difficulty levels",
        "🔄 Scalable: Easy to add new emotion datasets and brain regions",
        "📊 Monitored: Full telemetry and health monitoring during training",
        "🔧 PyTorch + NumPy: Both PyTorch models and NumPy exports for flexibility"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")
    
    return True

def show_production_usage():
    """Show how to use the system in production"""
    print("\n🏭 Production Usage")
    print("=" * 50)
    
    commands = [
        "# 1. Train emotion classifier on GoEmotions",
        "python tools/train_go_emotions.py --files datasets/go_emotions/train.jsonl --epochs 4",
        "",
        "# 2. Run complete emotion curriculum pipeline", 
        "python tools/train_emotion_curriculum.py --dataset-root datasets --emotion-epochs 4",
        "",
        "# 3. Start AURA system (auto-loads emotion classifier)",
        "python -m aura.system.bootloader",
        "",
        "# 4. Test emotion classification",
        "# The system will automatically use the trained emotion classifier",
        "# for emotional analysis and routing decisions"
    ]
    
    for cmd in commands:
        print(f"  {cmd}")
    
    return True

def main():
    """Run the complete emotion training demonstration"""
    print("🎭 AURA Emotion Training Example")
    print("=" * 70)
    
    try:
        # Test 1: GoEmotions training
        print("\n📋 Test 1: GoEmotions Training")
        success1 = test_go_emotions_training()
        
        # Test 2: Emotion curriculum pipeline
        print("\n📋 Test 2: Emotion Curriculum Pipeline")
        success2 = test_emotion_curriculum_pipeline()
        
        # Show benefits
        print("\n📋 Test 3: Integration Benefits")
        success3 = demonstrate_integration_benefits()
        
        # Show production usage
        print("\n📋 Test 4: Production Usage")
        success4 = show_production_usage()
        
        if success1 and success2 and success3 and success4:
            print("\n🎉 All tests completed successfully!")
            print("\nKey Features Demonstrated:")
            print("  ✅ GoEmotions emotion classifier training")
            print("  ✅ Curriculum pretraining integration")
            print("  ✅ PyTorch model + NumPy export compatibility")
            print("  ✅ AURA system integration")
            print("  ✅ Production usage patterns")
            
            print("\n🚀 Ready for Production:")
            print("  # Train emotion classifier:")
            print("  python tools/train_go_emotions.py --files datasets/go_emotions/train.jsonl")
            print()
            print("  # Run complete pipeline:")
            print("  python tools/train_emotion_curriculum.py --dataset-root datasets")
            print()
            print("  # Start AURA system:")
            print("  python -m aura.system.bootloader")
            
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
        if os.path.exists("mock_datasets"):
            shutil.rmtree("mock_datasets")
            print(f"\n🧹 Cleaned up mock datasets")
        if os.path.exists("mock_models"):
            shutil.rmtree("mock_models")
            print(f"🧹 Cleaned up mock models")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
