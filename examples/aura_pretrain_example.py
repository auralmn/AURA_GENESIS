#!/usr/bin/env python3
"""
AURA Pretrain Package Example
Demonstrates warm-starting the entire AURA system with enriched SVC/linguistic data
"""

import json
import numpy as np
from pathlib import Path
from aura_pretrain.features import build_router_features, build_emotion_sine_embedding, maybe_load_sbert
from aura_pretrain.teachers import emotion_label, router_teacher, hippocampus_salience

def create_mock_svc_dataset(n_samples: int = 100) -> list:
    """Create a mock enriched SVC/linguistic dataset for testing"""
    
    # Sample data with SVC structure, linguistic features, and metadata
    sample_texts = [
        "I love learning about ancient civilizations and their architectural marvels!",
        "Can you explain the time complexity of this sorting algorithm?",
        "I'm feeling anxious about the presentation tomorrow.",
        "The Renaissance period was a time of great artistic and scientific advancement.",
        "What are the tradeoffs between different machine learning approaches?",
        "I'm excited about this new project we're starting!",
        "The French Revolution changed the course of European history forever.",
        "I need help understanding this complex mathematical concept.",
        "This is absolutely amazing! I can't believe how well it works!",
        "The ancient Egyptians built incredible pyramids using advanced engineering techniques."
    ]
    
    emotions = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation", "neutral"]
    domains = ["history", "technology", "mathematics", "science", "art", "general"]
    realms = ["historical", "technical", "emotional", "analytical", "general"]
    difficulties = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    dataset = []
    for i in range(n_samples):
        text = np.random.choice(sample_texts)
        
        # SVC structure
        svc = {
            "subject": np.random.choice(["I", "The system", "This algorithm", "The user", "The data"]),
            "verb": np.random.choice(["love", "explain", "analyze", "compare", "understand", "build", "create"]),
            "complement": np.random.choice(["concepts", "algorithms", "data", "systems", "models", "solutions"])
        }
        
        # Linguistic features
        tokens = text.split()
        linguistic_features = {
            "tokens": [{"text": tok, "pos": "NN" if i % 3 == 0 else "VB" if i % 3 == 1 else "JJ"} for i, tok in enumerate(tokens)],
            "sentiment": np.random.choice(["positive", "negative", "neutral"]),
            "complexity": np.random.choice(["simple", "moderate", "complex"])
        }
        
        # Metadata
        metadata = {
            "svc": svc,
            "domain": np.random.choice(domains),
            "difficulty": np.random.choice(difficulties),
            "source": "mock_data",
            "timestamp": f"2024-01-{i%30+1:02d}"
        }
        
        # Plutchik emotions
        plutchik = {
            "primary": np.random.choice(emotions),
            "intensity": np.random.uniform(0.3, 1.0),
            "secondary": np.random.choice(emotions + [None])
        }
        
        record = {
            "text": text,
            "metadata": metadata,
            "linguistic_features": linguistic_features,
            "plutchik": plutchik,
            "realm": np.random.choice(realms),
            "intent": np.random.choice(["question", "statement", "request", "exclamation"]),
            "tone": np.random.choice(["euphoric", "tense", "somber", "peaceful", "amazed", "neutral"])
        }
        
        dataset.append(record)
    
    return dataset

def test_feature_building():
    """Test the feature building functions"""
    print("🧪 Testing Feature Building...")
    
    # Create mock dataset
    dataset = create_mock_svc_dataset(10)
    
    # Test SBERT loading
    print("  Testing SBERT loading...")
    sbert = maybe_load_sbert()
    if sbert is not None:
        print("  ✅ SBERT loaded successfully")
    else:
        print("  ⚠️  SBERT not available, using zeros")
    
    # Test router features
    print("  Testing router features...")
    router_feats = []
    for record in dataset:
        feat = build_router_features(record, sbert_model=sbert, alpha=0.7, out_dim=384)
        router_feats.append(feat)
    
    router_feats = np.stack(router_feats)
    print(f"  Router features shape: {router_feats.shape}")
    assert router_feats.shape == (10, 384), f"Expected (10, 384), got {router_feats.shape}"
    
    # Test emotion sine embeddings
    print("  Testing emotion sine embeddings...")
    emotion_feats = []
    for record in dataset:
        feat = build_emotion_sine_embedding(record, length=192)
        emotion_feats.append(feat)
    
    emotion_feats = np.stack(emotion_feats)
    print(f"  Emotion features shape: {emotion_feats.shape}")
    assert emotion_feats.shape == (10, 192), f"Expected (10, 192), got {emotion_feats.shape}"
    
    print("  ✅ Feature building test passed!")

def test_teachers():
    """Test the teacher functions"""
    print("\n🧪 Testing Teacher Functions...")
    
    dataset = create_mock_svc_dataset(20)
    
    # Test emotion labeling
    print("  Testing emotion labeling...")
    emotion_labels = [emotion_label(record) for record in dataset]
    print(f"  Emotion labels: {set(emotion_labels)}")
    
    # Test router teaching
    print("  Testing router teaching...")
    router_teachers = [router_teacher(record) for record in dataset]
    print(f"  Router teachers: {set(router_teachers)}")
    
    # Test hippocampus salience
    print("  Testing hippocampus salience...")
    salience_scores = [hippocampus_salience(record) for record in dataset]
    print(f"  Salience range: {min(salience_scores):.3f} - {max(salience_scores):.3f}")
    
    print("  ✅ Teacher functions test passed!")

def test_linear_softmax():
    """Test the linear softmax training"""
    print("\n🧪 Testing Linear Softmax Training...")
    
    from aura_pretrain.warm_start import _train_linear_softmax
    
    # Create mock data
    X = np.random.randn(100, 192).astype(np.float32)
    y = np.random.randint(0, 5, 100)
    
    # Train linear model
    W, b = _train_linear_softmax(X, y, lr=1e-2, epochs=5)
    
    print(f"  Weight matrix shape: {W.shape}")
    print(f"  Bias vector shape: {b.shape}")
    print(f"  Weight matrix range: {W.min():.3f} - {W.max():.3f}")
    print(f"  Bias vector range: {b.min():.3f} - {b.max():.3f}")
    
    # Test prediction
    logits = X @ W + b
    predictions = np.argmax(logits, axis=1)
    accuracy = (predictions == y).mean()
    print(f"  Training accuracy: {accuracy:.3f}")
    
    print("  ✅ Linear softmax training test passed!")

def test_jsonl_loading():
    """Test JSONL loading functionality"""
    print("\n🧪 Testing JSONL Loading...")
    
    from aura_pretrain.warm_start import _load_jsonl
    
    # Create temporary JSONL file
    dataset = create_mock_svc_dataset(5)
    temp_file = "temp_test.jsonl"
    
    with open(temp_file, 'w') as f:
        for record in dataset:
            f.write(json.dumps(record) + '\n')
    
    # Test loading
    loaded_data = _load_jsonl(temp_file)
    print(f"  Loaded {len(loaded_data)} records")
    print(f"  First record keys: {list(loaded_data[0].keys())}")
    
    # Cleanup
    Path(temp_file).unlink()
    
    print("  ✅ JSONL loading test passed!")

def demonstrate_pretraining_workflow():
    """Demonstrate the complete pretraining workflow"""
    print("\n🚀 Demonstrating Pretraining Workflow...")
    
    # Create mock dataset
    dataset = create_mock_svc_dataset(50)
    
    # Save to temporary JSONL file
    temp_file = "temp_pretrain.jsonl"
    with open(temp_file, 'w') as f:
        for record in dataset:
            f.write(json.dumps(record) + '\n')
    
    print(f"  Created mock dataset: {temp_file}")
    print(f"  Dataset size: {len(dataset)} records")
    
    # Analyze dataset characteristics
    print("\n  Dataset Analysis:")
    emotions = [emotion_label(r) for r in dataset]
    router_targets = [router_teacher(r) for r in dataset]
    salience_scores = [hippocampus_salience(r) for r in dataset]
    
    print(f"    Emotions: {len(set(emotions))} unique ({set(emotions)})")
    print(f"    Router targets: {len(set(router_targets))} unique ({set(router_targets)})")
    print(f"    Salience range: {min(salience_scores):.3f} - {max(salience_scores):.3f}")
    
    # Test feature building
    print("\n  Feature Building:")
    sbert = maybe_load_sbert()
    
    router_feats = np.stack([build_router_features(r, sbert_model=sbert) for r in dataset])
    emotion_feats = np.stack([build_emotion_sine_embedding(r) for r in dataset])
    
    print(f"    Router features: {router_feats.shape} (384D)")
    print(f"    Emotion features: {emotion_feats.shape} (192D)")
    print(f"    Feature norms - Router: {np.linalg.norm(router_feats, axis=1).mean():.3f}")
    print(f"    Feature norms - Emotion: {np.linalg.norm(emotion_feats, axis=1).mean():.3f}")
    
    # Test linear model training
    print("\n  Linear Model Training:")
    from aura_pretrain.warm_start import _train_linear_softmax
    
    emotion_labels = [emotion_label(r) for r in dataset]
    unique_labels = sorted(set(emotion_labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y_emotion = np.array([label_to_idx[label] for label in emotion_labels])
    
    W, b = _train_linear_softmax(emotion_feats, y_emotion, lr=1e-2, epochs=10)
    
    # Test predictions
    logits = emotion_feats @ W + b
    predictions = np.argmax(logits, axis=1)
    accuracy = (predictions == y_emotion).mean()
    
    print(f"    Emotion classifier: {W.shape[0]}D -> {W.shape[1]} classes")
    print(f"    Training accuracy: {accuracy:.3f}")
    print(f"    Weight matrix stats: mean={W.mean():.3f}, std={W.std():.3f}")
    
    # Cleanup
    Path(temp_file).unlink()
    
    print("  ✅ Pretraining workflow demonstration completed!")

def main():
    """Run all tests and demonstrations"""
    print("🧠 AURA Pretrain Package Test Suite")
    print("=" * 60)
    
    try:
        test_feature_building()
        test_teachers()
        test_linear_softmax()
        test_jsonl_loading()
        demonstrate_pretraining_workflow()
        
        print("\n🎉 All tests passed successfully!")
        print("\nKey Features Demonstrated:")
        print("  ✅ SBERT integration with fallback to zeros")
        print("  ✅ Router features: SBERT + SVC + domain + linguistic")
        print("  ✅ Emotion features: Pure sine wave embeddings")
        print("  ✅ Teacher functions: Emotion, router, salience")
        print("  ✅ Linear softmax training for AmygdalaRelay")
        print("  ✅ JSONL dataset loading and processing")
        print("  ✅ Complete pretraining workflow")
        
        print("\n🚀 Ready for Production:")
        print("  python -m aura_pretrain.cli --data datasets/train.jsonl --out svc_nlms_weights")
        print("  --sbert-device mps --emotion-dim 192 --router-alpha 0.7 --moe-topk 2")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
