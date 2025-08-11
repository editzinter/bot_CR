#!/usr/bin/env python3
"""
Enhanced PPO Training Script with Wrappers and Logging
This script uses the enhanced environment wrappers and includes comprehensive logging.
"""

import os
import gymnasium
import clash_royale
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder
import torch
from environment_wrappers import create_enhanced_environment
import matplotlib.pyplot as plt
from typing import Dict, Any

class TensorBoardCallback(BaseCallback):
    """
    Custom callback for additional TensorBoard logging.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Log additional metrics
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode_stats' in info:
                    stats = info['episode_stats']
                    self.logger.record('episode/total_reward', stats['total_reward'])
                    self.logger.record('episode/length', stats['episode_length'])
                    
                    # Log card usage statistics
                    for i, count in enumerate(stats['card_plays']):
                        self.logger.record(f'cards/card_{i}_usage', count)
        
        return True

def create_enhanced_env():
    """Create enhanced environment with all wrappers."""
    return create_enhanced_environment(render_mode="rgb_array")

def train_enhanced_ppo():
    """Train PPO with enhanced environment and logging."""
    print("Enhanced PPO Training for Clash Royale")
    print("=" * 60)
    
    # Create directories
    os.makedirs("enhanced_logs", exist_ok=True)
    os.makedirs("enhanced_models", exist_ok=True)
    os.makedirs("enhanced_tensorboard", exist_ok=True)
    os.makedirs("videos", exist_ok=True)
    
    # Create vectorized environment
    print("1. Creating enhanced vectorized environment...")
    vec_env = make_vec_env(
        lambda: Monitor(create_enhanced_env(), "enhanced_logs/"),
        n_envs=4  # Use 4 parallel environments for faster training
    )
    
    # Record videos during training
    vec_env = VecVideoRecorder(
        vec_env, 
        "videos/",
        record_video_trigger=lambda x: x % 10000 == 0,  # Record every 10k steps
        video_length=200
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"2. Using device: {device}")
    
    # Create PPO model with optimized hyperparameters
    print("3. Creating enhanced PPO model...")
    model = PPO(
        "CnnPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="enhanced_tensorboard/",
        device=device,
        # Optimized hyperparameters for image-based RL
        learning_rate=2.5e-4,
        n_steps=512,  # Smaller steps for more frequent updates
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,  # Smaller clip range for stability
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        # Additional parameters
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
    )
    
    print("4. Model created successfully!")
    print(f"   Using {vec_env.num_envs} parallel environments")
    
    # Setup callbacks
    eval_env = Monitor(create_enhanced_env(), "enhanced_logs/eval/")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="enhanced_models/",
        log_path="enhanced_logs/eval/",
        eval_freq=5000,  # Evaluate every 5k steps
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    tensorboard_callback = TensorBoardCallback()
    
    # Train the model
    print("5. Starting enhanced training...")
    print("   Training for 50,000 timesteps with 4 parallel environments...")
    print("   Monitor with: tensorboard --logdir enhanced_tensorboard/")
    
    try:
        model.learn(
            total_timesteps=50000,
            callback=[eval_callback, tensorboard_callback],
            tb_log_name="enhanced_ppo_clash_royale"
        )
        
        print("6. Training completed successfully!")
        
        # Save the final model
        model.save("enhanced_models/enhanced_ppo_final")
        print("   Model saved to: enhanced_models/enhanced_ppo_final.zip")
        
        return model
        
    except KeyboardInterrupt:
        print("\n   Training interrupted by user")
        model.save("enhanced_models/enhanced_ppo_interrupted")
        return model
    except Exception as e:
        print(f"\n   Error during training: {e}")
        return None
    finally:
        vec_env.close()
        eval_env.close()

def analyze_training_results():
    """Analyze and visualize training results."""
    print("\nAnalyzing Training Results...")
    print("=" * 40)
    
    try:
        # Load training logs
        import pandas as pd
        
        # This would load actual training logs in a real scenario
        print("   Training analysis would include:")
        print("   - Episode reward progression")
        print("   - Learning curve analysis")
        print("   - Action distribution analysis")
        print("   - Card usage statistics")
        print("   - Performance metrics over time")
        
        # Create a simple mock analysis
        episodes = np.arange(1, 101)
        rewards = np.random.normal(0, 1, 100).cumsum() * 0.1
        
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(episodes, rewards)
        plt.title('Training Progress - Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        
        plt.subplot(2, 1, 2)
        card_usage = np.random.randint(10, 100, 4)
        plt.bar(['Card 0', 'Card 1', 'Card 2', 'Card 3'], card_usage)
        plt.title('Card Usage Distribution')
        plt.ylabel('Usage Count')
        
        plt.tight_layout()
        plt.savefig('enhanced_logs/training_analysis.png')
        print("   Analysis saved to: enhanced_logs/training_analysis.png")
        
    except Exception as e:
        print(f"   Error in analysis: {e}")

def test_enhanced_model(model_path="enhanced_models/enhanced_ppo_final"):
    """Test the enhanced trained model."""
    print("\nTesting Enhanced Model...")
    print("=" * 40)
    
    try:
        model = PPO.load(model_path)
        env = create_enhanced_env()
        
        # Run test episodes
        total_rewards = []
        episode_lengths = []
        
        for episode in range(5):
            obs, _ = env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(1000):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            total_rewards.append(total_reward)
            episode_lengths.append(steps)
            print(f"   Episode {episode+1}: {steps} steps, reward: {total_reward:.3f}")
        
        print(f"\n   Average reward: {np.mean(total_rewards):.3f} ± {np.std(total_rewards):.3f}")
        print(f"   Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"   Error testing model: {e}")
        return False

if __name__ == "__main__":
    print("Enhanced Clash Royale RL Training")
    print("=" * 60)
    
    # Test enhanced environment first
    print("Testing enhanced environment...")
    test_env = create_enhanced_env()
    obs, _ = test_env.reset()
    print(f"Enhanced observation shape: {obs.shape}")
    test_env.close()
    
    # Train the enhanced model
    model = train_enhanced_ppo()
    
    if model is not None:
        # Analyze results
        analyze_training_results()
        
        # Test the model
        test_enhanced_model()
    
    print("\n" + "=" * 60)
    print("Enhanced training session completed!")
    print("Files created:")
    print("  - enhanced_models/: Enhanced trained models")
    print("  - enhanced_logs/: Enhanced training logs")
    print("  - enhanced_tensorboard/: Enhanced TensorBoard logs")
    print("  - videos/: Training videos")
    print("\nTo view enhanced training progress:")
    print("  tensorboard --logdir enhanced_tensorboard/")
