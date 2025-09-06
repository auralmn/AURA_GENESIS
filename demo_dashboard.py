#!/usr/bin/env python3
"""
AURA Dashboard Demo
Demonstrates the real-time health monitoring dashboard
"""

import asyncio
import trio
import sys
import os
import time
import numpy as np

# Add AURA to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from aura.system.websocket_bridge import AURAWebSocketBridge


async def demo_dashboard():
    """Run a demo of the AURA health dashboard"""
    print("🎬 AURA Health Dashboard Demo")
    print("=" * 40)
    print()
    print("This demo will:")
    print("1. Start the WebSocket bridge")
    print("2. Initialize AURA with health monitoring")
    print("3. Simulate neural activity")
    print("4. Stream data to the dashboard")
    print()
    print("🌐 Dashboard will be available at: http://localhost:3000")
    print("🔌 WebSocket server running on: http://localhost:3001")
    print()
    
    # Create bridge
    bridge = AURAWebSocketBridge()
    
    try:
        # Connect to WebSocket server
        print("🔌 Connecting to WebSocket server...")
        if not await bridge.connect_to_server():
            print("❌ Failed to connect to WebSocket server")
            print("💡 Please start the WebSocket server first:")
            print("   cd dashboard/server && node websocket-server.js")
            return
        
        # Initialize AURA
        print("🧠 Initializing AURA network...")
        if not await bridge.initialize_aura_network():
            print("❌ Failed to initialize AURA network")
            return
        
        print("✅ AURA network ready!")
        print()
        
        # Start monitoring
        print("🏥 Starting health monitoring...")
        print("📊 Dashboard will show real-time data")
        print()
        print("Press Ctrl+C to stop the demo")
        print()
        
        # Run the demo
        await bridge.start_health_monitoring()
        
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🛑 Shutting down demo...")
        await bridge.stop()


async def main():
    """Main demo function"""
    print("🚀 Starting AURA Dashboard Demo...")
    print()
    
    # Check if WebSocket server is running
    try:
        import socketio
        sio = socketio.AsyncClient()
        await sio.connect('http://localhost:3001')
        await sio.disconnect()
        print("✅ WebSocket server is running")
    except:
        print("❌ WebSocket server not running!")
        print("Please start it first:")
        print("  cd dashboard/server && node websocket-server.js")
        print()
        print("Or run the full dashboard:")
        print("  cd dashboard && ./start-dashboard.sh")
        return
    
    print()
    await demo_dashboard()


if __name__ == "__main__":
    trio.run(main)
