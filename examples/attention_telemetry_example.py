#!/usr/bin/env python3
"""
Attention Telemetry System Example
Demonstrates comprehensive attention monitoring and logging
"""

import numpy as np
import asyncio
import trio
from aura.core.thalamic_router import ThalamicConversationRouter
from aura.core.attention_telemetry import AttentionTelemetryLogger, print_attention_summary

async def example_live_attention_monitoring():
    """Example: Live attention monitoring with hooks"""
    print("🔴 Live Attention Monitoring Example")
    print("=" * 50)
    
    # Create router with attention enabled
    router = ThalamicConversationRouter(
        neuron_count=30,
        features=384,
        input_dim=384,
        enable_attention=True
    )
    
    # Set up live logging hook
    logger = AttentionTelemetryLogger()
    router.set_attention_hook(lambda ev: logger.log_event(ev))
    
    print("Processing conversations with live attention logging...")
    
    conversations = [
        {
            'query': "Hello! How are you today?",
            'features': np.random.randn(384),
            'outcome': {'user_satisfaction': 0.8, 'response_quality': 0.7}
        },
        {
            'query': "WOW! This is AMAZING! I can't believe it! 😊",
            'features': np.random.randn(384),
            'outcome': {'user_satisfaction': 0.9, 'response_quality': 0.8}
        },
        {
            'query': "What is the time complexity of this algorithm?",
            'features': np.random.randn(384),
            'outcome': {'user_satisfaction': 0.7, 'response_quality': 0.9}
        }
    ]
    
    for i, conv in enumerate(conversations):
        print(f"\n--- Conversation {i+1} ---")
        print(f"Query: '{conv['query']}'")
        
        # Analyze intent (generates telemetry)
        intent = router.analyze_conversation_intent(conv['query'], conv['features'])
        print(f"Primary target: {intent['primary_target']}")
        print(f"Attention gain: {intent['attention_gain']:.3f}")
        
        # Update routing (generates telemetry)
        routing_plan = {
            'primary_specialist': intent['primary_target'],
            'routing_strategy': intent['routing_strategy'],
            'confidence': intent['routing_confidence']
        }
        
        await router.adaptive_routing_update_with_attention(
            routing_plan, conv['outcome'], conv['features'], conv['query']
        )
        print("✅ Routing updated with attention modulation")

async def example_attention_analytics():
    """Example: Comprehensive attention analytics"""
    print("\n📊 Attention Analytics Example")
    print("=" * 50)
    
    # Create router and process some conversations
    router = ThalamicConversationRouter(
        neuron_count=30,
        features=384,
        input_dim=384,
        enable_attention=True
    )
    
    # Process various types of queries
    query_types = [
        ("Casual", "Hi there! How's it going?"),
        ("Technical", "What is the time complexity of quicksort?"),
        ("Emotional", "I'm so excited about this! 😊🎉"),
        ("Question", "What do you think about this?"),
        ("Exclamation", "WOW! This is AMAZING!"),
        ("Mixed", "I love this algorithm! It's so efficient! 🚀")
    ]
    
    for query_type, query in query_types:
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
    
    # Get comprehensive analytics
    print("Attention Summary Statistics:")
    print_attention_summary(router.attn_buf)
    
    print("\nRecent Events Analysis:")
    recent_events = router.recent_attention_events(5)
    for i, event in enumerate(recent_events):
        print(f"  Event {i+1}: μ={event['mu_scalar']:.3f}, "
              f"winners={len(event['winners_idx'])}, "
              f"salience_μ={event['salience_mean']:.3f}, "
              f"spikes=({event['spike_rate_amp']:.2f},{event['spike_rate_pitch']:.2f},{event['spike_rate_boundary']:.2f})")

def example_attention_filtering():
    """Example: Attention event filtering and analysis"""
    print("\n🔍 Attention Event Filtering Example")
    print("=" * 50)
    
    # Create router and process conversations
    router = ThalamicConversationRouter(
        neuron_count=30,
        features=384,
        input_dim=384,
        enable_attention=True
    )
    
    # Process various conversations
    conversations = [
        "Hello! How are you?",
        "WOW! This is AMAZING!",
        "What is the complexity?",
        "I'm so excited! 😊",
        "Can you help me?",
        "This is incredible! 🎉",
        "What do you think?",
        "I love this! ❤️"
    ]
    
    for query in conversations:
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
    
    # Filter events by different criteria
    print("Event Filtering Results:")
    
    # By note type
    update_events = router.attn_buf.get_events_by_note("adaptive_routing_update_with_attention")
    intent_events = router.attn_buf.get_events_by_note("intent_analysis_attention")
    print(f"  Update events: {len(update_events)}")
    print(f"  Intent analysis events: {len(intent_events)}")
    
    # High attention events
    high_attention_events = router.attn_buf.get_high_attention_events(threshold=1.5)
    print(f"  High attention events (μ > 1.5): {len(high_attention_events)}")
    
    # Show details of high attention events
    if high_attention_events:
        print("  High attention event details:")
        for i, event in enumerate(high_attention_events[:3]):
            print(f"    Event {i+1}: μ={event.mu_scalar:.3f}, "
                  f"winners={len(event.winners_idx)}, "
                  f"note={event.note}")

def example_spike_analysis():
    """Example: Spike pattern analysis"""
    print("\n⚡ Spike Pattern Analysis Example")
    print("=" * 50)
    
    # Create router and process conversations
    router = ThalamicConversationRouter(
        neuron_count=30,
        features=384,
        input_dim=384,
        enable_attention=True
    )
    
    # Process conversations with different emotional content
    emotional_queries = [
        "I'm so happy! 😊",
        "This is terrible! 😡",
        "WOW! Amazing! 🎉",
        "I love this! ❤️",
        "This is awful! 😭",
        "Fantastic! 🚀",
        "I hate this! 😤",
        "Incredible! 🌟"
    ]
    
    for query in emotional_queries:
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
    
    # Analyze spike patterns
    spike_analysis = router.attn_buf.get_spike_analysis()
    print("Spike Pattern Analysis:")
    print(f"  Dominant channel: {spike_analysis['dominant_channel']}")
    print(f"  Spike diversity: {spike_analysis['spike_diversity']:.3f}")
    print(f"  Channel correlations:")
    print(f"    Amplitude-Pitch: {spike_analysis['correlation_amp_pitch']:.3f}")
    print(f"    Amplitude-Boundary: {spike_analysis['correlation_amp_boundary']:.3f}")
    print(f"    Pitch-Boundary: {spike_analysis['correlation_pitch_boundary']:.3f}")

async def example_attention_presets_comparison():
    """Example: Comparing different attention presets"""
    print("\n🎛️ Attention Presets Comparison Example")
    print("=" * 50)
    
    from aura.core.attention import RouterAttentionPresets
    
    presets = {
        'conversational': RouterAttentionPresets.conversational(),
        'technical': RouterAttentionPresets.technical(),
        'emotional': RouterAttentionPresets.emotional(),
        'streaming': RouterAttentionPresets.streaming()
    }
    
    test_queries = [
        "Hello! How are you?",
        "What is the time complexity of this algorithm?",
        "I'm so excited about this! 😊",
        "Can you help me with this problem?"
    ]
    
    print("Comparing attention presets across different query types:")
    print(f"{'Query Type':<15} {'Conversational':<15} {'Technical':<15} {'Emotional':<15} {'Streaming':<15}")
    print("-" * 80)
    
    for query in test_queries:
        query_type = "Casual" if "Hello" in query else "Technical" if "algorithm" in query else "Emotional" if "excited" in query else "Question"
        
        results = []
        for preset_name, preset in presets.items():
            # Create router with this preset
            test_router = ThalamicConversationRouter(
                neuron_count=20, features=384, input_dim=384, enable_attention=True
            )
            test_router.attn = preset
            
            # Analyze intent
            intent = test_router.analyze_conversation_intent(query, np.random.randn(384))
            results.append(f"{intent['attention_gain']:.3f}")
        
        print(f"{query_type:<15} {results[0]:<15} {results[1]:<15} {results[2]:<15} {results[3]:<15}")

async def example_attention_telemetry_export():
    """Example: Exporting attention telemetry data"""
    print("\n📤 Attention Telemetry Export Example")
    print("=" * 50)
    
    # Create router and process conversations
    router = ThalamicConversationRouter(
        neuron_count=30,
        features=384,
        input_dim=384,
        enable_attention=True
    )
    
    # Process some conversations
    conversations = [
        "Hello! How are you?",
        "WOW! This is AMAZING!",
        "What is the complexity?",
        "I'm so excited! 😊"
    ]
    
    for query in conversations:
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
    
    # Export telemetry data
    print("Exporting attention telemetry data...")
    
    # Get summary statistics
    summary = router.get_attention_summary()
    print("Summary Statistics:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Get recent events as dictionaries
    recent_events = router.recent_attention_events(10)
    print(f"\nRecent Events (as dictionaries):")
    for i, event in enumerate(recent_events):
        print(f"  Event {i+1}: {event}")
    
    # Get events by type
    update_events = router.attn_buf.get_events_by_note("adaptive_routing_update_with_attention")
    intent_events = router.attn_buf.get_events_by_note("intent_analysis_attention")
    
    print(f"\nEvent Counts:")
    print(f"  Update events: {len(update_events)}")
    print(f"  Intent analysis events: {len(intent_events)}")
    print(f"  Total events: {len(router.attn_buf._events)}")

async def main():
    """Run all examples"""
    print("🧠 Attention Telemetry System Examples")
    print("=" * 80)
    
    # Run examples
    await example_live_attention_monitoring()
    await example_attention_analytics()
    await example_attention_filtering()
    await example_spike_analysis()
    await example_attention_presets_comparison()
    await example_attention_telemetry_export()
    
    print("\n🎉 All examples completed successfully!")
    print("\nKey Features of Attention Telemetry System:")
    print("  🔴 Live attention event logging with hooks")
    print("  📊 Comprehensive telemetry statistics")
    print("  🔍 Event filtering and analysis")
    print("  ⚡ Spike pattern analysis and correlations")
    print("  🎛️ Attention preset comparisons")
    print("  📤 Telemetry data export capabilities")
    print("  📈 Per-query attention monitoring")
    print("  🎯 Routing success correlation tracking")
    print("  🔄 Real-time attention modulation visibility")

if __name__ == "__main__":
    trio.run(main)
