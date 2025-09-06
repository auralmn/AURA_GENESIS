#!/usr/bin/env python3
"""
Quick test script for the AURA_GENESIS bootloader
Tests the system with correct paths for weights and models
"""

import asyncio
import trio
import sys
import os

# Add the aura directory to the path
sys.path.insert(0, '/Volumes/Others2/AURA_GENESIS')

from aura.system.bootloader import AuraBootConfig, boot_aura_genesis


async def quick_test():
    """Quick test of the bootloader functionality"""
    print("AURA_GENESIS Quick Test")
    print("=" * 40)
    
    # Create configuration using YAML config file
    config = AuraBootConfig()
    
    try:
        print("Booting system...")
        bootloader = await boot_aura_genesis(config)
        
        print("System booted successfully!")
        
        # Test basic functionality
        print("\nTesting basic query processing...")
        result = await bootloader.process_query("Hello, how are you?")
        print(f"Query result: {result}")
        
        # Test SVC analysis
        print("\nTesting SVC analysis...")
        svc_result = await bootloader.analyze_svc_structure("The cat sat on the mat")
        print(f"SVC result: {svc_result}")
        
        # Test pre-trained models (if available)
        print("\nTesting pre-trained models...")
        try:
            emotion_result = await bootloader.classify_emotion("I'm so happy today!")
            print(f"Emotion classification: {emotion_result}")
        except Exception as e:
            print(f"Emotion classification not available: {e}")
        
        # Show system status
        print("\nSystem Status:")
        status = bootloader.get_system_status()
        print(f"Health: {status['health']['status']}")
        print(f"Components: {list(status['health']['component_status'].keys())}")
        
        # Graceful shutdown
        print("\nShutting down...")
        await bootloader.shutdown_system()
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Main test function"""
    print("Starting AURA_GENESIS bootloader test...")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")
    print()
    
    success = trio.run(quick_test)
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
