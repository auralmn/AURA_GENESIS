#!/usr/bin/env python3
"""
Multi-Channel Spiking Attention Example
Demonstrates how to use the new attention system with real-world examples
"""

import numpy as np
import asyncio
import trio
from aura.core.multi_channel_attention import (
    MultiChannelSpikingAttention, 
    prosody_channels_from_text,
    build_token_to_feature_mapping,
    AttentionPresets
)
from aura.core.neuron import Neuron, MaturationStage, ActivityState

async def example_emotional_processing():
    """Example: Emotional content processing with multi-channel attention"""
    print("🎭 Emotional Content Processing Example")
    print("=" * 50)
    
    # Create emotional attention system
    emotional_attention = AttentionPresets.emotional()
    
    # Sample emotional text
    emotional_texts = [
        "I am SO excited about this! 😊",
        "This is absolutely terrible and awful!",
        "I love you so much! ❤️",
        "What a beautiful day! 🌟",
        "I hate this so much! 😡"
    ]
    
    # Create emotional processing neuron
    emotional_neuron = Neuron(
        neuron_id=1,
        specialization='emotional_processor',
        abilities={'emotional_salience': 0.9, 'valence_detection': 0.85},
        n_features=384,
        n_outputs=1,
        enable_attention=True
    )
    
    print("Processing emotional texts with multi-channel attention...")
    for i, text in enumerate(emotional_texts):
        # Extract features (simplified)
        x = np.random.randn(384)  # In real usage, this would be SBERT features
        
        # Process with multi-channel attention
        result = await emotional_neuron.update_nlms_with_multi_channel_attention(
            x, y_true=0.8, content_text=text, multi_channel_attention=emotional_attention
        )
        
        print(f"  Text {i+1}: '{text}'")
        print(f"    Attention result: {float(result):.3f}")
        print(f"    Attention stats: {emotional_neuron.attention_stats}")
        print()

async def example_analytical_processing():
    """Example: Analytical content processing with multi-channel attention"""
    print("🔬 Analytical Content Processing Example")
    print("=" * 50)
    
    # Create analytical attention system
    analytical_attention = AttentionPresets.analytical()
    
    # Sample analytical text
    analytical_texts = [
        "The algorithm complexity is O(n log n).",
        "This mathematical proof demonstrates the theorem.",
        "The data structure provides efficient access.",
        "The optimization reduces computational overhead.",
        "The analysis shows significant improvements."
    ]
    
    # Create analytical processing neuron
    analytical_neuron = Neuron(
        neuron_id=2,
        specialization='analytical_processor',
        abilities={'logical_reasoning': 0.9, 'pattern_recognition': 0.85},
        n_features=384,
        n_outputs=1,
        enable_attention=True
    )
    
    print("Processing analytical texts with multi-channel attention...")
    for i, text in enumerate(analytical_texts):
        x = np.random.randn(384)
        
        result = await analytical_neuron.update_nlms_with_multi_channel_attention(
            x, y_true=0.7, content_text=text, multi_channel_attention=analytical_attention
        )
        
        print(f"  Text {i+1}: '{text}'")
        print(f"    Attention result: {float(result):.3f}")
        print()

async def example_streaming_processing():
    """Example: Streaming content processing with stateful attention"""
    print("🌊 Streaming Content Processing Example")
    print("=" * 50)
    
    # Create streaming attention system
    streaming_attention = AttentionPresets.streaming()
    
    # Simulate streaming text chunks
    streaming_chunks = [
        "Hello everyone!",
        "Welcome to our presentation.",
        "Today we'll discuss AI.",
        "This is very exciting!",
        "Let's begin with the basics."
    ]
    
    # Create streaming processing neuron
    streaming_neuron = Neuron(
        neuron_id=3,
        specialization='streaming_processor',
        abilities={'real_time_processing': 0.9, 'context_maintenance': 0.85},
        n_features=384,
        n_outputs=1,
        enable_attention=True
    )
    
    print("Processing streaming chunks with stateful attention...")
    for i, chunk in enumerate(streaming_chunks):
        x = np.random.randn(384)
        
        result = await streaming_neuron.update_nlms_with_multi_channel_attention(
            x, y_true=0.6, content_text=chunk, multi_channel_attention=streaming_attention
        )
        
        print(f"  Chunk {i+1}: '{chunk}'")
        print(f"    Attention result: {float(result):.3f}")
        print(f"    LIF state: amp={streaming_attention.v_amp:.3f}, pitch={streaming_attention.v_pitch:.3f}, bound={streaming_attention.v_bound:.3f}")
        print()

def example_custom_attention_config():
    """Example: Custom attention configuration"""
    print("⚙️ Custom Attention Configuration Example")
    print("=" * 50)
    
    # Create custom attention system
    custom_attention = MultiChannelSpikingAttention(
        k_winners=4,
        w_amp=1.5,      # Emphasize amplitude
        w_pitch=0.8,    # De-emphasize pitch
        w_bound=1.2,    # Emphasize boundaries
        gain_up=2.0,    # High gain for winners
        gain_down=0.3,  # Low gain for non-winners
        smoothing=2,    # Light smoothing
        normalize_salience=True
    )
    
    # Test with sample text
    text = "WOW! This is AMAZING! I can't believe it! 😱"
    tokens = text.lower().split()
    token_ids = [hash(token) % 50000 for token in tokens]
    
    # Extract prosody channels
    amp, pitch, boundary = prosody_channels_from_text(tokens)
    
    # Compute attention
    result = custom_attention.compute(
        token_ids=token_ids,
        amp=amp,
        pitch=pitch,
        boundary=boundary
    )
    
    print(f"Text: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Amplitude: {amp}")
    print(f"Pitch: {pitch}")
    print(f"Boundary: {boundary}")
    print(f"Mu scalar: {result['mu_scalar']:.3f}")
    print(f"Winners: {result['winners_idx']}")
    print(f"Salience: {result['salience']}")
    print()

def example_per_feature_gains():
    """Example: Per-feature gain modulation"""
    print("🎯 Per-Feature Gain Modulation Example")
    print("=" * 50)
    
    # Create attention system
    attention = AttentionPresets.emotional()
    
    # Sample text
    text = "I love this amazing product! It's fantastic!"
    tokens = text.lower().split()
    token_ids = [hash(token) % 50000 for token in tokens]
    
    # Extract prosody channels
    amp, pitch, boundary = prosody_channels_from_text(tokens)
    
    # Build token-to-feature mapping
    feature_size = 10
    token_to_feature = build_token_to_feature_mapping(tokens, feature_size)
    
    # Compute attention with per-feature gains
    result = attention.compute(
        token_ids=token_ids,
        amp=amp,
        pitch=pitch,
        boundary=boundary,
        feature_size=feature_size,
        token_to_feature=token_to_feature
    )
    
    print(f"Text: '{text}'")
    print(f"Feature size: {feature_size}")
    print(f"Token-to-feature mapping: {token_to_feature}")
    print(f"Per-feature gains: {result['per_feature_gains']}")
    print(f"Mu scalar: {result['mu_scalar']:.3f}")
    print()

async def main():
    """Run all examples"""
    print("🧠 Multi-Channel Spiking Attention Examples")
    print("=" * 60)
    print()
    
    # Run examples
    await example_emotional_processing()
    await example_analytical_processing()
    await example_streaming_processing()
    example_custom_attention_config()
    example_per_feature_gains()
    
    print("🎉 All examples completed successfully!")
    print("\nKey Benefits of Multi-Channel Spiking Attention:")
    print("  🎯 Selective Focus - Winners get enhanced learning")
    print("  ⚡ Adaptive Learning - Dynamic learning rate modulation")
    print("  🧠 Biologically Plausible - Mimics neural attention mechanisms")
    print("  📊 k-WTA Competition - Winner-take-all dynamics")
    print("  🔄 Real-time Processing - Streaming support with stateful LIF")
    print("  🎭 Domain-Specific Tuning - Preset configurations for different use cases")
    print("  📈 Per-Feature Modulation - Granular control over learning rates")

if __name__ == "__main__":
    trio.run(main)
