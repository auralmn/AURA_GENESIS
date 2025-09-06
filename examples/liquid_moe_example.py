#!/usr/bin/env python3
"""
Liquid-MoE Spike Router Example
Demonstrates continuous-time dynamics, Top-K sparse routing, and local learning
"""

import numpy as np
import asyncio
import trio
from aura.core.thalamic_router import ThalamicConversationRouter
from aura.core.liquid_moe import LiquidMoERouter, NLMSExpertAdapter, EnergyMeter
from aura.core.attention_telemetry import AttentionTelemetryLogger

async def example_liquid_moe_basic_usage():
    """Example: Basic Liquid-MoE usage"""
    print("🌊 Basic Liquid-MoE Usage Example")
    print("=" * 50)
    
    # Create router with attention enabled
    router = ThalamicConversationRouter(
        neuron_count=30,
        features=384,
        input_dim=384,
        enable_attention=True
    )
    
    print("1. Liquid-MoE Router Initialization")
    moe_stats = router.get_moe_stats()
    print(f"   MoE enabled: {moe_stats['enabled']}")
    print(f"   Number of experts: {moe_stats['n_experts']}")
    print(f"   Top-K routing: {moe_stats['top_k']}")
    print(f"   Temperature: {moe_stats['temperature']}")
    
    print("\n2. Expert Usage Balance")
    usage_balance = router.get_moe_usage_balance()
    print(f"   Target usage: {usage_balance['target_usage']:.3f}")
    print(f"   Usage std: {usage_balance['usage_std']:.3f}")
    print(f"   Usage entropy: {usage_balance['usage_entropy']:.3f}")
    
    print("\n3. Routing Different Query Types")
    
    test_queries = [
        "Hello! How are you today?",
        "What is the time complexity of this algorithm?",
        "I'm so excited about this! 😊",
        "Can you help me with this technical problem?",
        "WOW! This is AMAZING! 🎉"
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\n   Query {i+1}: '{query}'")
        
        # Analyze intent with MoE routing
        intent = router.analyze_conversation_intent(query, np.random.randn(384))
        
        print(f"   Primary target: {intent['primary_target']}")
        print(f"   Routing confidence: {intent['routing_confidence']:.3f}")
        print(f"   Attention gain: {intent['attention_gain']:.3f}")
        
        # Show MoE details
        moe_info = intent['moe']
        print(f"   MoE prediction: {moe_info['y']:.3f}")
        print(f"   Top-K experts: {moe_info['topk']}")
        print(f"   Energy consumed: {moe_info['energy_j']:.2e} J")

async def example_liquid_moe_learning():
    """Example: Liquid-MoE learning capabilities"""
    print("\n🧠 Liquid-MoE Learning Example")
    print("=" * 50)
    
    # Create router with attention enabled
    router = ThalamicConversationRouter(
        neuron_count=30,
        features=384,
        input_dim=384,
        enable_attention=True
    )
    
    print("1. Learning with Different Conversation Outcomes")
    
    conversations = [
        {
            'query': "Hello! How are you?",
            'features': np.random.randn(384),
            'outcome': {'user_satisfaction': 0.8, 'response_quality': 0.7}
        },
        {
            'query': "What is the time complexity?",
            'features': np.random.randn(384),
            'outcome': {'user_satisfaction': 0.9, 'response_quality': 0.9}
        },
        {
            'query': "I'm so excited! 😊",
            'features': np.random.randn(384),
            'outcome': {'user_satisfaction': 0.7, 'response_quality': 0.8}
        },
        {
            'query': "Can you help me?",
            'features': np.random.randn(384),
            'outcome': {'user_satisfaction': 0.6, 'response_quality': 0.9}
        }
    ]
    
    for i, conv in enumerate(conversations):
        print(f"\n   Learning Conversation {i+1}: '{conv['query']}'")
        
        # Analyze intent
        intent = router.analyze_conversation_intent(conv['query'], conv['features'])
        print(f"   Primary target: {intent['primary_target']}")
        print(f"   MoE prediction: {intent['moe']['y']:.3f}")
        
        # Update routing with learning
        routing_plan = {
            'primary_specialist': intent['primary_target'],
            'routing_strategy': intent['routing_strategy'],
            'confidence': intent['routing_confidence']
        }
        
        await router.adaptive_routing_update_with_attention(
            routing_plan, conv['outcome'], conv['features'], conv['query']
        )
        print("   ✅ Learning update completed")
        
        # Show updated usage balance
        usage_balance = router.get_moe_usage_balance()
        print(f"   Updated usage std: {usage_balance['usage_std']:.3f}")
        print(f"   Updated usage entropy: {usage_balance['usage_entropy']:.3f}")

async def example_liquid_moe_attention_integration():
    """Example: Liquid-MoE integration with attention system"""
    print("\n⚡ Liquid-MoE + Attention Integration Example")
    print("=" * 50)
    
    # Create router with attention enabled
    router = ThalamicConversationRouter(
        neuron_count=30,
        features=384,
        input_dim=384,
        enable_attention=True
    )
    
    # Set up attention hook for live logging
    logger = AttentionTelemetryLogger()
    router.set_attention_hook(lambda ev: logger.log_event(ev))
    
    print("1. Attention-Modulated Routing")
    
    attention_queries = [
        ("Casual", "Hi there! How's it going?"),
        ("Technical", "What is the time complexity of quicksort?"),
        ("Emotional", "I'm so excited about this! 😊🎉"),
        ("Question", "What do you think about this?"),
        ("Exclamation", "WOW! This is AMAZING!"),
        ("Mixed", "I love this algorithm! It's so efficient! 🚀")
    ]
    
    for query_type, query in attention_queries:
        print(f"\n   {query_type} Query: '{query}'")
        
        # Analyze intent
        intent = router.analyze_conversation_intent(query, np.random.randn(384))
        
        print(f"   Primary target: {intent['primary_target']}")
        print(f"   Attention gain: {intent['attention_gain']:.3f}")
        print(f"   MoE prediction: {intent['moe']['y']:.3f}")
        print(f"   Top-K experts: {intent['moe']['topk']}")
        
        # Show attention telemetry
        recent_events = router.recent_attention_events(1)
        if recent_events:
            event = recent_events[0]
            print(f"   Attention event: μ={event['mu_scalar']:.3f}, "
                  f"winners={len(event['winners_idx'])}, "
                  f"spikes=({event['spike_rate_amp']:.2f},{event['spike_rate_pitch']:.2f},{event['spike_rate_boundary']:.2f})")

async def example_liquid_moe_energy_tracking():
    """Example: Liquid-MoE energy tracking capabilities"""
    print("\n⚡ Liquid-MoE Energy Tracking Example")
    print("=" * 50)
    
    # Create router with attention enabled
    router = ThalamicConversationRouter(
        neuron_count=30,
        features=384,
        input_dim=384,
        enable_attention=True
    )
    
    print("1. Energy Consumption Tracking")
    
    # Process several queries and track energy
    queries = [
        "Hello! How are you?",
        "What is the time complexity?",
        "I'm so excited! 😊",
        "Can you help me?",
        "WOW! This is AMAZING!"
    ]
    
    total_energy = 0.0
    
    for i, query in enumerate(queries):
        print(f"\n   Query {i+1}: '{query}'")
        
        # Analyze intent
        intent = router.analyze_conversation_intent(query, np.random.randn(384))
        
        # Get energy stats
        moe_stats = router.get_moe_stats()
        energy_stats = moe_stats['energy_stats']
        
        print(f"   Total energy: {energy_stats['total_energy_j']:.2e} J")
        print(f"   Gating energy: {energy_stats['gating_energy_j']:.2e} J")
        print(f"   Expert energy: {energy_stats['expert_energy_j']:.2e} J")
        print(f"   Energy per MAC: {energy_stats['energy_per_mac_j']:.2e} J")
        
        total_energy += energy_stats['total_energy_j']
    
    print(f"\n   Total energy consumed: {total_energy:.2e} J")
    print(f"   Average energy per query: {total_energy / len(queries):.2e} J")

async def example_liquid_moe_load_balancing():
    """Example: Liquid-MoE load balancing capabilities"""
    print("\n⚖️ Liquid-MoE Load Balancing Example")
    print("=" * 50)
    
    # Create router with attention enabled
    router = ThalamicConversationRouter(
        neuron_count=30,
        features=384,
        input_dim=384,
        enable_attention=True
    )
    
    print("1. Load Balancing with Biased Queries")
    
    # Create biased queries that favor certain experts
    biased_queries = [
        ("Technical", "What is the time complexity of this algorithm?"),
        ("Technical", "How does this data structure work?"),
        ("Technical", "What is the space complexity?"),
        ("Technical", "Can you explain this algorithm?"),
        ("Technical", "What is the Big O notation?"),
        ("General", "Hello! How are you?"),
        ("General", "What's the weather like?"),
        ("General", "How are you doing?"),
        ("General", "Nice to meet you!"),
        ("General", "Good morning!")
    ]
    
    print("   Processing biased queries...")
    for query_type, query in biased_queries:
        intent = router.analyze_conversation_intent(query, np.random.randn(384))
        
        routing_plan = {
            'primary_specialist': intent['primary_target'],
            'routing_strategy': intent['routing_strategy'],
            'confidence': intent['routing_confidence']
        }
        
        outcome = {
            'user_satisfaction': np.random.uniform(0.6, 0.9),
            'response_quality': np.random.uniform(0.6, 0.9)
        }
        
        await router.adaptive_routing_update_with_attention(
            routing_plan, outcome, np.random.randn(384), query
        )
    
    print("\n2. Load Balancing Results")
    usage_balance = router.get_moe_usage_balance()
    
    print("   Expert usage distribution:")
    for expert, usage in usage_balance['usage_ma'].items():
        print(f"     {expert}: {usage:.3f}")
    
    print(f"\n   Load balancing metrics:")
    print(f"     Target usage: {usage_balance['target_usage']:.3f}")
    print(f"     Usage std: {usage_balance['usage_std']:.3f}")
    print(f"     Usage entropy: {usage_balance['usage_entropy']:.3f}")
    
    # Test load balancing effectiveness
    target = usage_balance['target_usage']
    max_deviation = max(abs(usage - target) for usage in usage_balance['usage_ma'].values())
    print(f"     Max deviation from target: {max_deviation:.3f}")
    
    if max_deviation < 0.1:
        print("   ✅ Load balancing is working well!")
    else:
        print("   ⚠️ Load balancing may need tuning")

async def example_liquid_moe_reset_and_recovery():
    """Example: Liquid-MoE reset and recovery capabilities"""
    print("\n🔄 Liquid-MoE Reset and Recovery Example")
    print("=" * 50)
    
    # Create router with attention enabled
    router = ThalamicConversationRouter(
        neuron_count=30,
        features=384,
        input_dim=384,
        enable_attention=True
    )
    
    print("1. State Reset")
    
    # Process some queries to build up state
    queries = [
        "Hello! How are you?",
        "What is the time complexity?",
        "I'm so excited! 😊"
    ]
    
    for query in queries:
        intent = router.analyze_conversation_intent(query, np.random.randn(384))
        print(f"   Processed: {query} -> {intent['primary_target']}")
    
    # Get state before reset
    usage_before = router.get_moe_usage_balance()
    print(f"   Usage before reset: {usage_before['usage_std']:.3f}")
    
    # Reset MoE state
    router.reset_moe()
    print("   ✅ MoE state reset")
    
    # Get state after reset
    usage_after = router.get_moe_usage_balance()
    print(f"   Usage after reset: {usage_after['usage_std']:.3f}")
    
    print("\n2. Recovery After Reset")
    
    # Process queries after reset
    for query in queries:
        intent = router.analyze_conversation_intent(query, np.random.randn(384))
        print(f"   Recovery: {query} -> {intent['primary_target']}")
    
    # Check final state
    usage_final = router.get_moe_usage_balance()
    print(f"   Usage after recovery: {usage_final['usage_std']:.3f}")

async def example_liquid_moe_custom_configuration():
    """Example: Custom Liquid-MoE configuration"""
    print("\n🎛️ Custom Liquid-MoE Configuration Example")
    print("=" * 50)
    
    # Create router with attention enabled
    router = ThalamicConversationRouter(
        neuron_count=30,
        features=384,
        input_dim=384,
        enable_attention=True
    )
    
    print("1. Current Configuration")
    moe_stats = router.get_moe_stats()
    print(f"   Number of experts: {moe_stats['n_experts']}")
    print(f"   Top-K routing: {moe_stats['top_k']}")
    print(f"   Temperature: {moe_stats['temperature']}")
    
    print("\n2. Testing Different Query Types")
    
    test_queries = [
        "Hello! How are you?",
        "What is the time complexity?",
        "I'm so excited! 😊",
        "Can you help me?",
        "WOW! This is AMAZING!"
    ]
    
    for query in test_queries:
        intent = router.analyze_conversation_intent(query, np.random.randn(384))
        print(f"   '{query}' -> {intent['primary_target']} (conf: {intent['routing_confidence']:.3f})")
    
    print("\n3. Usage Statistics")
    usage_balance = router.get_moe_usage_balance()
    print(f"   Usage std: {usage_balance['usage_std']:.3f}")
    print(f"   Usage entropy: {usage_balance['usage_entropy']:.3f}")
    
    print("\n4. Energy Statistics")
    energy_stats = moe_stats['energy_stats']
    print(f"   Total energy: {energy_stats['total_energy_j']:.2e} J")
    print(f"   Energy per MAC: {energy_stats['energy_per_mac_j']:.2e} J")

async def main():
    """Run all Liquid-MoE examples"""
    print("🌊 Liquid-MoE Spike Router Examples")
    print("=" * 80)
    
    # Run examples
    await example_liquid_moe_basic_usage()
    await example_liquid_moe_learning()
    await example_liquid_moe_attention_integration()
    await example_liquid_moe_energy_tracking()
    await example_liquid_moe_load_balancing()
    await example_liquid_moe_reset_and_recovery()
    await example_liquid_moe_custom_configuration()
    
    print("\n🎉 All Liquid-MoE examples completed successfully!")
    print("\nKey Features of Liquid-MoE System:")
    print("  🌊 Continuous-time liquid dynamics with input-dependent time constants")
    print("  🎯 Top-K sparse routing for efficient expert selection")
    print("  🧠 Local learning rules without backpropagation")
    print("  ⚡ Attention-modulated routing with spiking dynamics")
    print("  ⚖️ Load balancing with bias nudges")
    print("  📊 Energy tracking for Planck-style accounting")
    print("  🔄 State reset and recovery capabilities")
    print("  🎛️ Configurable temperature and routing parameters")
    print("  📈 Usage statistics and load monitoring")
    print("  🚀 Streaming learning with real-time adaptation")
    print("\nIntegration Benefits:")
    print("  🎯 Selective Expert Routing - Focus on most relevant specialists")
    print("  ⚡ Adaptive Learning - Higher learning rates for important concepts")
    print("  🧠 Neurobiologically Plausible - Mimics biological attention mechanisms")
    print("  📊 k-WTA Competition - Winner-take-all dynamics enhance learning efficiency")
    print("  🔄 Dynamic Modulation - Real-time learning rate adjustment based on content")
    print("  🎭 Domain-Specific Tuning - Different attention configs for each intelligence domain")
    print("  📈 Learning Efficiency - Reduced training time through selective focus")
    print("  🌟 Enhanced Generalization - Better pattern recognition through attention")

if __name__ == "__main__":
    trio.run(main)
