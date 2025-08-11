#!/usr/bin/env python3
"""
PPO Training Script for Clash Royale Environment
This script trains a PPO agent on the Clash Royale environment.
"""

import os
import gymnasium
import clash_royale
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import torch

def create_environment():
    """Create and return the Clash Royale environment."""
    env = gymnasium.make("clash-royale", render_mode="rgb_array")
    return env

def train_ppo_agent():
    """Train a PPO agent on the Clash Royale environment."""
    print("Setting up PPO training for Clash Royale...")
    print("=" * 60)
    
    # Create directories for logs and models
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    
    # Create environment
    print("1. Creating environment...")
    env = create_environment()
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")
    
    # Wrap environment with Monitor for logging
    env = Monitor(env, "logs/")
    
    # Create vectorized environment (PPO works better with vectorized envs)
    print("2. Creating vectorized environment...")
    vec_env = make_vec_env(lambda: Monitor(create_environment(), "logs/"), n_envs=1)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"3. Using device: {device}")
    
    # Create PPO model with CNN policy (for image observations)
    print("4. Creating PPO model...")
    model = PPO(
        "CnnPolicy",  # CNN policy for image observations
        vec_env,
        verbose=1,
        tensorboard_log="tensorboard_logs/",
        device=device,
        # PPO hyperparameters
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
    
    print("5. Model created successfully!")
    print(f"   Policy: {model.policy}")
    print(f"   Total parameters: {sum(p.numel() for p in model.policy.parameters())}")
    
    # Setup evaluation callback
    eval_env = Monitor(create_environment(), "logs/eval/")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path="logs/eval/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Train the model
    print("6. Starting training...")
    print("   Training for 100,000 timesteps...")
    print("   You can monitor progress with: tensorboard --logdir tensorboard_logs/")
    
    try:
        model.learn(
            total_timesteps=100000,
            callback=eval_callback,
            tb_log_name="ppo_clash_royale"
        )
        
        print("7. Training completed successfully!")
        
        # Save the final model
        model.save("models/ppo_clash_royale_final")
        print("   Model saved to: models/ppo_clash_royale_final.zip")
        
        return model
        
    except KeyboardInterrupt:
        print("\n   Training interrupted by user")
        model.save("models/ppo_clash_royale_interrupted")
        print("   Model saved to: models/ppo_clash_royale_interrupted.zip")
        return model
    except Exception as e:
        print(f"\n   Error during training: {e}")
        return None
    finally:
        vec_env.close()
        eval_env.close()

def test_trained_model(model_path="models/ppo_clash_royale_final"):
    """Test a trained model in the environment."""
    print("\nTesting trained model...")
    print("=" * 40)
    
    try:
        # Load the model
        model = PPO.load(model_path)
        print(f"   Model loaded from: {model_path}.zip")
        
        # Create environment for testing
        env = create_environment()
        
        # Test the model
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        
        print("   Running test episode...")
        for step in range(1000):  # Max 1000 steps for testing
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if step % 100 == 0:
                print(f"   Step {step}: reward={reward:.3f}, total_reward={total_reward:.3f}")
            
            if terminated or truncated:
                break
        
        print(f"   Test completed!")
        print(f"   Total steps: {steps}")
        print(f"   Total reward: {total_reward:.3f}")
        print(f"   Average reward per step: {total_reward/steps:.3f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"   Error testing model: {e}")
        return False

if __name__ == "__main__":
    print("Clash Royale PPO Training")
    print("=" * 60)
    
    # Train the model
    model = train_ppo_agent()
    
    if model is not None:
        print("\n" + "=" * 60)
        # Test the trained model
        test_trained_model()
    
    print("\n" + "=" * 60)
    print("Training session completed!")
    print("Files created:")
    print("  - models/: Trained model files")
    print("  - logs/: Training logs")
    print("  - tensorboard_logs/: TensorBoard logs")
    print("\nTo view training progress:")
    print("  tensorboard --logdir tensorboard_logs/")
