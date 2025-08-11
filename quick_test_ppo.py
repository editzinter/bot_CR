#!/usr/bin/env python3
"""
Quick PPO test for Clash Royale Environment
This script does a quick test with fewer timesteps to verify everything works.
"""

import os
import gymnasium
import clash_royale
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import torch

def quick_test():
    """Quick test of PPO training."""
    print("Quick PPO Test for Clash Royale")
    print("=" * 40)
    
    # Create directories
    os.makedirs("test_logs", exist_ok=True)
    os.makedirs("test_models", exist_ok=True)
    
    # Create environment
    print("1. Creating environment...")
    def make_env():
        env = gymnasium.make("clash-royale", render_mode="rgb_array")
        return Monitor(env, "test_logs/")
    
    vec_env = make_vec_env(make_env, n_envs=1)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"2. Using device: {device}")
    
    # Create PPO model
    print("3. Creating PPO model...")
    model = PPO(
        "CnnPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=128,  # Smaller for quick test
        batch_size=32,
        n_epochs=4,
        device=device,
    )
    
    print("4. Training for 1000 timesteps...")
    try:
        model.learn(total_timesteps=1000)
        print("   ✓ Training completed successfully!")
        
        # Save model
        model.save("test_models/quick_test_ppo")
        print("   ✓ Model saved")
        
        # Quick test
        print("5. Testing model...")
        env = gymnasium.make("clash-royale", render_mode="rgb_array")
        obs, _ = env.reset()
        
        for i in range(10):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   Step {i+1}: action={action}, reward={reward:.3f}")
            
            if terminated or truncated:
                break
        
        env.close()
        vec_env.close()
        
        print("   ✓ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        vec_env.close()
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n✓ Quick test passed! Ready for full training.")
    else:
        print("\n❌ Quick test failed. Check the error above.")
