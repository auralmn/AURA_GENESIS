#!/usr/bin/env python3
"""
ThalamicConversationRouter with Multi-Channel Attention Example
Demonstrates the attention upgrade with μ scaling
"""

import numpy as np
import asyncio
import trio
from aura.core.thalamic_router import ThalamicConversationRouter
from aura.core.attention import RouterAttentionPresets

async def example_conversational_routing():
    """Example: Conversational routing with attention"""
    print("💬 Conversational Routing with Attention Example")
    print("=" * 60)
    
    # Create router with attention enabled
    router = ThalamicConversationRouter(
        neuron_count=30,
        features=384,
        input_dim=384,
        enable_attention=True
    )
    
    # Sample conversations
    conversations = [
        {
            'query': "Hello! How are you today?",
            'features': np.random.randn(384),
            'outcome': {'user_satisfaction': 0.8, 'response_quality': 0.7}
        },
        {
            'query': "I'm so excited about this project! 😊",
            'features': np.random.randn(384),
            'outcome': {'user_satisfaction': 0.9, 'response_quality': 0.8}
        },
        {
            'query': "Can you help me with this technical problem?",
            'features': np.random.randn(384),
            'outcome': {'user_satisfaction': 0.7, 'response_quality': 0.9}
        }
    ]
    
    print("Processing conversations with attention...")
    for i, conv in enumerate(conversations):
        print(f"\n  Conversation {i+1}:")
        print(f"    Query: '{conv['query']}'")
        
        # Analyze intent with attention
        intent = router.analyze_conversation_intent(conv['query'], conv['features'])
        print(f"    Primary target: {intent['primary_target']}")
        print(f"    Attention gain: {intent['attention_gain']:.3f}")
        print(f"    Routing confidence: {intent['routing_confidence']:.3f}")
        
        # Get attention telemetry
        telemetry = router.get_attention_telemetry(conv['query'])
        print(f"    Winners: {telemetry['winners_count']} tokens")
        print(f"    Spike counts: {telemetry['spike_counts']}")
        
        # Update routing with attention
        routing_plan = {
            'primary_specialist': intent['primary_target'],
            'routing_strategy': intent['routing_strategy']
        }
        
        await router.adaptive_routing_update_with_attention(
            routing_plan, conv['outcome'], conv['features'], conv['query']
        )
        print("    ✅ Routing updated with attention modulation")

async def example_preset_comparison():
    """Example: Comparing different attention presets"""
    print("\n🎛️ Attention Preset Comparison Example")
    print("=" * 60)
    
    presets = {
        'conversational': RouterAttentionPresets.conversational(),
        'technical': RouterAttentionPresets.technical(),
        'emotional': RouterAttentionPresets.emotional(),
        'streaming': RouterAttentionPresets.streaming()
    }
    
    test_queries = [
        "Hello! How are you?",
        "What is the time complexity of this algorithm?",
        "I'm so excited about this! 😊🎉",
        "Can you analyze this data for me?"
    ]
    
    for query in test_queries:
        print(f"\n  Query: '{query}'")
        print("    Preset comparison:")
        
        for preset_name, preset in presets.items():
            router = ThalamicConversationRouter(
                neuron_count=20, features=384, input_dim=384, enable_attention=True
            )
            router.attn = preset
            
            intent = router.analyze_conversation_intent(query, np.random.randn(384))
            telemetry = router.get_attention_telemetry(query)
            
            print(f"      {preset_name:12}: gain={intent['attention_gain']:.3f}, "
                  f"winners={telemetry['winners_count']}, "
                  f"spikes={telemetry['spike_counts']}")

async def example_attention_telemetry():
    """Example: Detailed attention telemetry analysis"""
    print("\n📊 Attention Telemetry Analysis Example")
    print("=" * 60)
    
    router = ThalamicConversationRouter(
        neuron_count=30, features=384, input_dim=384, enable_attention=True
    )
    
    # Test different types of queries
    query_types = [
        ("Casual", "Hi there! How's it going?"),
        ("Technical", "What is the time complexity of quicksort?"),
        ("Emotional", "I'm so excited about this! 😊🎉"),
        ("Historical", "Tell me about the Roman Empire"),
        ("Analytical", "Can you analyze this data for me?"),
        ("Question", "What do you think about this?"),
        ("Exclamation", "WOW! This is AMAZING!"),
        ("Mixed", "I love this algorithm! It's so efficient! 🚀")
    ]
    
    for query_type, query in query_types:
        print(f"\n  {query_type} Query: '{query}'")
        
        # Get detailed telemetry
        telemetry = router.get_attention_telemetry(query)
        
        if telemetry['enabled']:
            print(f"    Mu scalar: {telemetry['mu_scalar']:.3f}")
            print(f"    Winners: {telemetry['winners_count']} out of {len(telemetry['tokens'])} tokens")
            print(f"    Winner indices: {telemetry['winners_idx']}")
            print(f"    Salience: avg={telemetry['avg_salience']:.3f}, max={telemetry['max_salience']:.3f}")
            print(f"    Spike counts: {telemetry['spike_counts']}")
            print(f"    Tokens: {telemetry['tokens']}")
            
            # Analyze prosody channels
            prosody = telemetry['prosody_channels']
            print(f"    Prosody analysis:")
            print(f"      Amplitude: {prosody['amplitude']}")
            print(f"      Pitch: {prosody['pitch']}")
            print(f"      Boundary: {prosody['boundary']}")

async def example_routing_confidence_impact():
    """Example: How attention affects routing confidence"""
    print("\n🎯 Routing Confidence Impact Example")
    print("=" * 60)
    
    # Create two routers: one with attention, one without
    router_no_attn = ThalamicConversationRouter(
        neuron_count=30, features=384, input_dim=384, enable_attention=False
    )
    
    router_with_attn = ThalamicConversationRouter(
        neuron_count=30, features=384, input_dim=384, enable_attention=True
    )
    
    test_queries = [
        "Hello! How are you?",
        "I'm so excited about this! 😊",
        "What is the complexity of this algorithm?",
        "Tell me about the Roman Empire",
        "Can you help me with this problem?"
    ]
    
    print("Comparing routing confidence with and without attention:")
    print(f"{'Query':<40} {'No Attn':<10} {'With Attn':<10} {'Gain':<8} {'Impact':<10}")
    print("-" * 80)
    
    for query in test_queries:
        features = np.random.randn(384)
        
        # Analyze without attention
        intent_no_attn = router_no_attn.analyze_conversation_intent(query, features)
        conf_no_attn = intent_no_attn['routing_confidence']
        
        # Analyze with attention
        intent_with_attn = router_with_attn.analyze_conversation_intent(query, features)
        conf_with_attn = intent_with_attn['routing_confidence']
        attn_gain = intent_with_attn['attention_gain']
        
        # Calculate impact
        impact = ((conf_with_attn - conf_no_attn) / conf_no_attn * 100) if conf_no_attn > 0 else 0
        
        print(f"{query:<40} {conf_no_attn:<10.3f} {conf_with_attn:<10.3f} {attn_gain:<8.3f} {impact:>+7.1f}%")

async def example_custom_attention_config():
    """Example: Custom attention configuration"""
    print("\n⚙️ Custom Attention Configuration Example")
    print("=" * 60)
    
    from aura.core.attention import MultiChannelSpikingAttention
    
    # Create custom attention configuration
    custom_attention = MultiChannelSpikingAttention(
        k_winners=3,
        w_amp=1.5,      # Emphasize amplitude
        w_pitch=0.8,    # De-emphasize pitch
        w_bound=1.2,    # Emphasize boundaries
        gain_up=2.0,    # High gain for winners
        gain_down=0.3,  # Low gain for non-winners
        smoothing=2,    # Light smoothing
        normalize_salience=True
    )
    
    # Create router with custom attention
    router = ThalamicConversationRouter(
        neuron_count=30, features=384, input_dim=384, enable_attention=True
    )
    router.attn = custom_attention
    
    test_query = "WOW! This is AMAZING! I can't believe it! 😱"
    print(f"Testing custom attention with query: '{test_query}'")
    
    # Get telemetry
    telemetry = router.get_attention_telemetry(test_query)
    print(f"  Custom config results:")
    print(f"    Mu scalar: {telemetry['mu_scalar']:.3f}")
    print(f"    Winners: {telemetry['winners_count']} tokens")
    print(f"    Winner indices: {telemetry['winners_idx']}")
    print(f"    Spike counts: {telemetry['spike_counts']}")
    print(f"    Tokens: {telemetry['tokens']}")

async def main():
    """Run all examples"""
    print("🧠 ThalamicConversationRouter with Multi-Channel Attention Examples")
    print("=" * 80)
    
    # Run examples
    await example_conversational_routing()
    await example_preset_comparison()
    await example_attention_telemetry()
    await example_routing_confidence_impact()
    await example_custom_attention_config()
    
    print("\n🎉 All examples completed successfully!")
    print("\nKey Benefits of Router Attention System:")
    print("  🎯 Multi-channel spike fusion (amplitude/pitch/boundary)")
    print("  ⚡ μ scaling for non-invasive attention injection")
    print("  🧠 Attention influence on routing confidence")
    print("  📊 Per-query attention telemetry")
    print("  🎛️ Preset configurations for different use cases")
    print("  🔒 Safe μ scaling with automatic restoration")
    print("  ⚡ O(T) complexity per message")
    print("  🎭 Domain-specific attention tuning")
    print("  📈 Enhanced routing accuracy through attention")

if __name__ == "__main__":
    trio.run(main)
