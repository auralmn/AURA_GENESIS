#!/usr/bin/env python3
"""
Test script for AURA WebSocket Bridge
Verifies the WebSocket connection and data flow
"""

import asyncio
import trio
import sys
import os

# Add AURA to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from aura.system.websocket_bridge import AURAWebSocketBridge


async def test_websocket_bridge():
    """Test the WebSocket bridge functionality"""
    print("🧪 Testing AURA WebSocket Bridge...")
    print("=" * 50)
    
    # Create bridge instance
    bridge = AURAWebSocketBridge()
    
    try:
        # Test connection
        print("🔌 Testing WebSocket connection...")
        connected = await bridge.connect_to_server()
        
        if not connected:
            print("❌ Failed to connect to WebSocket server")
            print("💡 Make sure to start the WebSocket server first:")
            print("   cd dashboard/server && node websocket-server.js")
            return False
        
        print("✅ WebSocket connection successful")
        
        # Test AURA network initialization
        print("\n🧠 Testing AURA network initialization...")
        network_initialized = await bridge.initialize_aura_network()
        
        if not network_initialized:
            print("❌ Failed to initialize AURA network")
            return False
        
        print("✅ AURA network initialized successfully")
        
        # Test health monitoring
        print("\n🏥 Testing health monitoring...")
        if bridge.health_monitor:
            # Capture a snapshot
            snapshot = await bridge.health_monitor._capture_snapshot()
            print(f"✅ Health snapshot captured: {len(snapshot.get('neuron_statuses', []))} neurons")
            
            # Test firing event recording
            test_event = {
                'timestamp': trio.current_time(),
                'neuron_id': 'test_neuron_1',
                'region': 'thalamus',
                'firing_strength': 0.85,
                'trigger_source': 'test',
                'trigger_details': 'WebSocket bridge test',
                'context': 'Testing firing event recording'
            }
            
            bridge.health_monitor.record_firing_event(test_event)
            print("✅ Firing event recorded successfully")
        else:
            print("❌ Health monitor not available")
            return False
        
        print("\n🎉 All tests passed!")
        print("✅ WebSocket bridge is working correctly")
        print("🌐 You can now start the dashboard:")
        print("   cd dashboard && ./start-dashboard.sh")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        await bridge.stop()


async def main():
    """Main test function"""
    success = await test_websocket_bridge()
    
    if success:
        print("\n🎯 WebSocket bridge test completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 WebSocket bridge test failed!")
        sys.exit(1)


if __name__ == "__main__":
    trio.run(main)
