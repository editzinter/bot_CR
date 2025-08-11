#!/usr/bin/env python3
"""
Test script for the Clash Royale Gymnasium environment.
This script verifies that the environment can be created, reset, and stepped through.
"""

import gymnasium
import clash_royale
import numpy as np

def test_environment():
    """Test the basic functionality of the Clash Royale environment."""
    print("Testing Clash Royale Environment...")
    print("=" * 50)
    
    try:
        # Create the environment
        print("1. Creating environment...")
        env = gymnasium.make("clash-royale", render_mode="rgb_array")
        print(f"   ✓ Environment created successfully")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: {env.observation_space}")
        
        # Reset the environment
        print("\n2. Resetting environment...")
        obs, info = env.reset()
        print(f"   ✓ Environment reset successfully")
        print(f"   Observation type: {type(obs)}")
        if hasattr(obs, 'shape'):
            print(f"   Observation shape: {obs.shape}")
        else:
            print(f"   Observation: {obs}")
        print(f"   Info: {info}")
        
        # Sample and take an action
        print("\n3. Taking a random action...")
        action = env.action_space.sample()
        print(f"   Sampled action: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   ✓ Step completed successfully")
        print(f"   New observation type: {type(obs)}")
        if hasattr(obs, 'shape'):
            print(f"   New observation shape: {obs.shape}")
        else:
            print(f"   New observation: {obs}")
        print(f"   Reward: {reward}")
        print(f"   Terminated: {terminated}")
        print(f"   Truncated: {truncated}")
        print(f"   Info: {info}")
        
        # Test a few more steps
        print("\n4. Testing multiple steps...")
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   Step {i+1}: action={action}, reward={reward}, terminated={terminated}")
            
            if terminated or truncated:
                print(f"   Episode ended at step {i+1}")
                break
        
        # Close the environment
        print("\n5. Closing environment...")
        env.close()
        print("   ✓ Environment closed successfully")
        
        print("\n" + "=" * 50)
        print("✓ All tests passed! Environment is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_environment()
    exit(0 if success else 1)
