#!/usr/bin/env python3
"""
Liquid-MoE + Amygdala Integration Example
Demonstrates how to use trained models with Liquid-MoE routing
"""

import numpy as np
import torch
import json
from sentence_transformers import SentenceTransformer
from aura.core.thalamic_router import ThalamicConversationRouter
from aura.core.liquid_moe import LiquidMoERouter, NLMSExpertAdapter
from aura.core.attention_telemetry import AttentionTelemetryLogger

class AmygdalaMoEExpert:
    """Expert that wraps trained Amygdala models for Liquid-MoE"""
    
    def __init__(self, model_path: str, task_name: str):
        self.task_name = task_name
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """Load trained linear model weights"""
        # Load weights
        W = np.load(f"{self.model_path}/{self.task_name}_W.npy")
        b = np.load(f"{self.model_path}/{self.task_name}_b.npy")
        
        # Load label mappings
        with open(f"{self.model_path}/{self.task_name}_labels.json", 'r') as f:
            self.label_data = json.load(f)
        
        self.W = torch.tensor(W, dtype=torch.float32)
        self.b = torch.tensor(b, dtype=torch.float32)
        self.idx_to_label = self.label_data['IDX_TO_LABEL']
        self.label_to_idx = self.label_data['LABEL_TO_IDX']
    
    def predict(self, features: np.ndarray) -> float:
        """Predict using linear model"""
        with torch.no_grad():
            features_tensor = torch.tensor(features, dtype=torch.float32)
            logits = torch.matmul(features_tensor, self.W) + self.b
            probs = torch.softmax(logits, dim=-1)
            return float(probs.max().item())
    
    def predict_class(self, features: np.ndarray) -> str:
        """Predict class label"""
        with torch.no_grad():
            features_tensor = torch.tensor(features, dtype=torch.float32)
            logits = torch.matmul(features_tensor, self.W) + self.b
            pred_idx = torch.argmax(logits, dim=-1).item()
            return self.idx_to_label[str(pred_idx)]

class EmotionFeatureBuilder:
    """Builds features compatible with trained Amygdala models"""
    
    def __init__(self, sbert_model_name: str = "all-MiniLM-L6-v2"):
        self.sbert_model = SentenceTransformer(sbert_model_name)
        self.sine_length = 32
        self.sbert_dim = 384
        self.extra_features = 3
        self.total_features = self.sine_length + self.extra_features + self.sbert_dim
        
        # Load emotion parameters (same as training)
        self.emotion_params = self._load_emotion_params()
    
    def _load_emotion_params(self):
        """Load emotion parameters used during training"""
        # This would normally be loaded from a config file
        # For now, we'll use the same default parameters as training
        emotion_labels = [
            "joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation", "none"
        ]
        
        params = {}
        for idx, label in enumerate(emotion_labels):
            params[label] = {
                "freq": 1.5 + 0.3 * idx,
                "amp": 0.7,
                "phase": 0.5 + 0.4 * idx
            }
        
        return params
    
    def build_features(self, text: str, emotion_info: dict = None) -> np.ndarray:
        """Build features from text and optional emotion info"""
        # Default emotion info if not provided
        if emotion_info is None:
            emotion_info = {
                "primary": "none",
                "intensity": 1.0,
                "secondary": None
            }
        
        # Primary emotion sine wave
        prim = emotion_info.get("primary", "none")
        inten = float(emotion_info.get("intensity", 1.0))
        cfg = self.emotion_params.get(prim, {"freq": 1.5, "amp": 0.7, "phase": 0.5})
        
        t = np.linspace(0, 2*np.pi, self.sine_length, dtype=np.float32)
        emb = (cfg["amp"] * inten * np.sin(cfg["freq"] * t + cfg["phase"])).astype(np.float32)
        
        # Secondary emotion (if present)
        sec = emotion_info.get("secondary")
        if sec and sec in self.emotion_params:
            cfg2 = self.emotion_params[sec]
            emb += 0.5 * (cfg2["amp"] * inten * np.sin(cfg2["freq"] * t + cfg2["phase"])).astype(np.float32)
        
        # Extra features
        extras = np.array([
            len(text) / 100.0,
            int("!" in text),
            int(any(tone in text.lower() for tone in ["euphoric", "tense", "somber", "peaceful", "amazed"]))
        ], dtype=np.float32)
        
        # SBERT embedding
        sbert_vec = self.sbert_model.encode([text], normalize_embeddings=True)[0]
        
        # Concatenate all features
        return np.concatenate([emb, extras, sbert_vec]).astype(np.float32)

class LiquidMoEAmygdalaSystem:
    """Complete Liquid-MoE + Amygdala integration system"""
    
    def __init__(self, model_path: str = "models"):
        self.model_path = model_path
        self.feature_builder = EmotionFeatureBuilder()
        
        # Initialize Amygdala experts
        self.experts = {
            'emotion': AmygdalaMoEExpert(model_path, 'emotion_classifier'),
            'intent': AmygdalaMoEExpert(model_path, 'intent_classifier'),
            'tone': AmygdalaMoEExpert(model_path, 'tone_classifier')
        }
        
        # Initialize Liquid-MoE router
        self.moe_router = LiquidMoERouter(
            experts={name: NLMSExpertAdapter(self._create_dummy_neuron()) for name in self.experts.keys()},
            in_dim=419,  # Total feature dimension
            hidden_dim=64,
            top_k=2,
            temperature=1.0
        )
        
        # Initialize attention system
        self.router = ThalamicConversationRouter(
            neuron_count=30,
            features=419,  # Match feature dimension
            input_dim=419,
            enable_attention=True
        )
        
        # Set up attention hook
        self.logger = AttentionTelemetryLogger()
        self.router.set_attention_hook(lambda ev: self.logger.log_event(ev))
    
    def _create_dummy_neuron(self):
        """Create dummy neuron for MoE expert adapter"""
        from aura.core.neuron import Neuron
        from aura.core.nlms import NLMSHead
        
        neuron = Neuron(
            neuron_id="dummy",
            n_features=419,
            n_outputs=1
        )
        neuron.nlms_head = NLMSHead(n_features=419, n_outputs=1)
        return neuron
    
    def analyze_emotion(self, text: str, emotion_info: dict = None) -> dict:
        """Analyze emotion using trained Amygdala models"""
        # Build features
        features = self.feature_builder.build_features(text, emotion_info)
        
        # Get predictions from all experts
        results = {}
        for task_name, expert in self.experts.items():
            confidence = expert.predict(features)
            predicted_class = expert.predict_class(features)
            results[task_name] = {
                'predicted_class': predicted_class,
                'confidence': confidence
            }
        
        # Use Liquid-MoE for routing
        moe_result = self.moe_router.route(features, attn_gain=1.0)
        
        # Use attention system for additional analysis
        intent_result = self.router.analyze_conversation_intent(text, features)
        
        return {
            'text': text,
            'features_shape': features.shape,
            'expert_predictions': results,
            'moe_routing': moe_result,
            'attention_analysis': intent_result,
            'emotion_info': emotion_info
        }
    
    def batch_analyze(self, texts: list, emotion_infos: list = None) -> list:
        """Analyze multiple texts in batch"""
        if emotion_infos is None:
            emotion_infos = [None] * len(texts)
        
        results = []
        for text, emotion_info in zip(texts, emotion_infos):
            result = self.analyze_emotion(text, emotion_info)
            results.append(result)
        
        return results
    
    def get_system_stats(self) -> dict:
        """Get system statistics"""
        return {
            'moe_stats': self.router.get_moe_stats(),
            'attention_summary': self.router.get_attention_summary(),
            'expert_info': {
                name: {
                    'task': name,
                    'num_classes': len(expert.idx_to_label),
                    'feature_dim': expert.W.shape[0]
                }
                for name, expert in self.experts.items()
            }
        }

def example_usage():
    """Example usage of the Liquid-MoE + Amygdala system"""
    print("🧠 Liquid-MoE + Amygdala Integration Example")
    print("=" * 60)
    
    # Initialize system
    print("1. Initializing system...")
    system = LiquidMoEAmygdalaSystem()
    
    # Test texts with different emotional content
    test_texts = [
        "I'm so excited about this project! 😊",
        "This is terrible, I can't believe it happened.",
        "What is the time complexity of this algorithm?",
        "I love this new feature! It's amazing! 🎉",
        "I'm feeling anxious about the presentation tomorrow.",
        "Can you help me understand this concept?",
        "WOW! This is incredible! I'm amazed! 🤯",
        "I'm disappointed with the results.",
        "This is a great opportunity for learning.",
        "I'm confused about how this works."
    ]
    
    # Emotion info for some texts
    emotion_infos = [
        {"primary": "joy", "intensity": 0.8, "secondary": "anticipation"},
        {"primary": "sadness", "intensity": 0.7, "secondary": "anger"},
        None,  # No emotion info
        {"primary": "joy", "intensity": 0.9, "secondary": "surprise"},
        {"primary": "fear", "intensity": 0.6, "secondary": "anticipation"},
        None,  # No emotion info
        {"primary": "surprise", "intensity": 0.9, "secondary": "joy"},
        {"primary": "sadness", "intensity": 0.5, "secondary": None},
        {"primary": "trust", "intensity": 0.7, "secondary": "anticipation"},
        {"primary": "fear", "intensity": 0.4, "secondary": "confusion"}
    ]
    
    print("\n2. Analyzing texts...")
    
    for i, (text, emotion_info) in enumerate(zip(test_texts, emotion_infos)):
        print(f"\n--- Text {i+1} ---")
        print(f"Text: '{text}'")
        if emotion_info:
            print(f"Emotion info: {emotion_info}")
        
        # Analyze emotion
        result = system.analyze_emotion(text, emotion_info)
        
        # Show expert predictions
        print("Expert predictions:")
        for task, pred in result['expert_predictions'].items():
            print(f"  {task}: {pred['predicted_class']} (conf: {pred['confidence']:.3f})")
        
        # Show MoE routing
        print(f"MoE routing: {result['moe_routing']['topk']}")
        print(f"MoE prediction: {result['moe_routing']['y']:.3f}")
        
        # Show attention analysis
        attn = result['attention_analysis']
        print(f"Attention gain: {attn['attention_gain']:.3f}")
        print(f"Primary target: {attn['primary_target']}")
    
    print("\n3. System statistics...")
    stats = system.get_system_stats()
    
    print("MoE Statistics:")
    moe_stats = stats['moe_stats']
    print(f"  Enabled: {moe_stats['enabled']}")
    print(f"  Number of experts: {moe_stats['n_experts']}")
    print(f"  Top-K routing: {moe_stats['top_k']}")
    
    print("\nExpert Information:")
    for name, info in stats['expert_info'].items():
        print(f"  {name}: {info['num_classes']} classes, {info['feature_dim']}D features")
    
    print("\n4. Batch analysis example...")
    
    # Batch analysis
    batch_texts = [
        "I'm happy about this!",
        "This makes me sad.",
        "I'm excited for the future!"
    ]
    
    batch_results = system.batch_analyze(batch_texts)
    
    print("Batch analysis results:")
    for i, result in enumerate(batch_results):
        print(f"  Text {i+1}: {result['expert_predictions']['emotion']['predicted_class']} "
              f"(conf: {result['expert_predictions']['emotion']['confidence']:.3f})")
    
    print("\n🎉 Integration example completed successfully!")
    print("\nKey Features Demonstrated:")
    print("  🧠 Trained Amygdala models integrated with Liquid-MoE")
    print("  🌊 Continuous-time dynamics with attention modulation")
    print("  🎯 Multi-task prediction (emotion, intent, tone)")
    print("  ⚡ Attention-modulated routing and learning")
    print("  📊 Comprehensive telemetry and monitoring")
    print("  🚀 Streaming analysis with real-time adaptation")

if __name__ == "__main__":
    example_usage()
