#!/usr/bin/env python3
"""
Extended PPO Training for Clash Royale - 5M+ Timesteps
This script implements long-term training with optimized hyperparameters,
comprehensive logging, and model checkpointing for transfer learning.
"""

import os
import sys
import time
import json
import gymnasium
import clash_royale
import numpy as np
from sb3_contrib import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, StopTrainingOnRewardThreshold, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, VecFrameStack, VecTransposeImage
import torch
from environment_wrappers import create_enhanced_environment
import matplotlib.pyplot as plt
from datetime import datetime
import random

# Define a pool of decks for sampling
DECK_POOL = [
    ["knight", "archer", "giant", "minions", "knight", "archer", "giant", "minions"],
    ["knight", "archer", "knight", "archer", "giant", "minions", "giant", "minions"],
    ["giant", "minions", "giant", "minions", "giant", "minions", "giant", "minions"],
    ["knight", "archer", "knight", "archer", "knight", "archer", "knight", "archer"],
]

class ExtendedTrainingCallback(BaseCallback):
    """
    Enhanced callback for extended training with detailed logging and analysis.
    """
    def __init__(self, log_dir="extended_logs", verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.action_counts = np.zeros(2304)  # Track action distribution
        self.training_start_time = time.time()
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize training log
        self.training_log = {
            "start_time": datetime.now().isoformat(),
            "episodes": [],
            "checkpoints": [],
            "hyperparameters": {}
        }
    
    def _on_training_start(self) -> None:
        """Called when training starts."""
        # Log hyperparameters
        if hasattr(self.model, 'learning_rate'):
            self.training_log["hyperparameters"] = {
                "learning_rate": float(self.model.learning_rate),
                "n_steps": self.model.n_steps,
                "batch_size": self.model.batch_size,
                "n_epochs": self.model.n_epochs,
                "gamma": self.model.gamma,
                "gae_lambda": self.model.gae_lambda,
                "clip_range": float(self.model.clip_range(1.0)),
                "ent_coef": self.model.ent_coef,
                "vf_coef": self.model.vf_coef,
            }
    
    def _on_step(self) -> bool:
        """Called at each step."""
        # Track actions
        if 'actions' in self.locals:
            actions = self.locals['actions']
            if hasattr(actions, '__iter__'):
                for action in actions:
                    if isinstance(action, (int, np.integer)) and 0 <= action < 2304:
                        self.action_counts[action] += 1
        
        # Log episode information
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    episode_info = {
                        "timestep": self.num_timesteps,
                        "reward": float(info['episode']['r']),
                        "length": int(info['episode']['l']),
                        "time": time.time() - self.training_start_time
                    }
                    self.training_log["episodes"].append(episode_info)
                    
                    # Log to TensorBoard
                    self.logger.record('episode/reward', episode_info["reward"])
                    self.logger.record('episode/length', episode_info["length"])
                    self.logger.record('episode/time_elapsed', episode_info["time"])
        
        # Log action distribution every 10k steps
        if self.num_timesteps % 10000 == 0:
            top_actions = np.argsort(self.action_counts)[-10:]
            for i, action in enumerate(top_actions):
                self.logger.record(f'actions/top_{i+1}_action', int(action))
                self.logger.record(f'actions/top_{i+1}_count', int(self.action_counts[action]))
        
        return True
    
    def save_training_log(self):
        """Save training log to file."""
        log_path = os.path.join(self.log_dir, "training_log.json")
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        
        # Save action distribution
        action_dist_path = os.path.join(self.log_dir, "action_distribution.npy")
        np.save(action_dist_path, self.action_counts)

def create_optimized_environment():
    """Create optimized environment for extended training."""
    import clash_royale
    import gymnasium
    from environment_wrappers import (
        EpisodeInfoWrapper, RewardShapingWrapper,
        ActionMaskingWrapper, DeckSamplingWrapper,
        ClashRoyaleRewardWrapper
    )

    # Create base environment
    env = gymnasium.make("clash-royale", render_mode="rgb_array")

    # Apply wrappers
    env = DeckSamplingWrapper(env, DECK_POOL)
    env = ClashRoyaleRewardWrapper(env)
    env = EpisodeInfoWrapper(env)
    # env = RewardShapingWrapper(env)
    # Note: Skipping ObservationNormalizationWrapper and FrameStackWrapper
    # to maintain compatibility with CnnPolicy (needs uint8 [0,255] observations)

    return env

def train_extended_ppo():
    """Train PPO for extended period with optimized settings."""
    print("[INFO] EXTENDED PPO TRAINING - 5M+ TIMESTEPS")
    print("=" * 60)
    
    # Create directories
    dirs = ["extended_models", "extended_logs", "extended_tensorboard", "extended_videos", "checkpoints"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    # Training configuration
    config = {
        "total_timesteps": 5_000_000,  # 5 million timesteps
        "n_envs": 8,  # 8 parallel environments
        "eval_freq": 50_000,  # Evaluate every 50k steps
        "checkpoint_freq": 250_000,  # Checkpoint every 250k steps
        "video_freq": 100_000,  # Record video every 100k steps
    }
    
    print(f"1. Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value:,}")
    
    # Create vectorized environment
    print("2. Creating optimized vectorized environment...")
    vec_env = make_vec_env(
        lambda: Monitor(create_optimized_environment(), "extended_logs/"),
        n_envs=config["n_envs"]
    )

    # Note: Removed VecFrameStack as it causes observation space issues with CnnPolicy
    # Frame stacking will be handled within the environment wrapper if needed
    
    # Add video recording
    vec_env = VecVideoRecorder(
        vec_env,
        "extended_videos/",
        record_video_trigger=lambda x: x % config["video_freq"] == 0,
        video_length=500
    )
    
    # Add action masking wrapper
    vec_env = ActionMaskingWrapper(vec_env)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"3. Using device: {device}")
    
    # Create PPO model with optimized hyperparameters for extended training
    print("4. Creating optimized PPO model...")
    model = PPO(
        "CnnPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="extended_tensorboard/",
        device=device,
        # Optimized hyperparameters for long-term training
        learning_rate=2.5e-4,  # Fixed learning rate
        n_steps=2048,  # Larger rollout buffer
        batch_size=256,  # Larger batch size
        n_epochs=10,  # More epochs per update
        gamma=0.995,  # Slightly higher discount factor
        gae_lambda=0.95,
        clip_range=0.2,  # Fixed clip range
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        # Additional stability parameters
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=0.01,  # Early stopping for stability
    )
    
    print(f"5. Model created with {sum(p.numel() for p in model.policy.parameters()):,} parameters")
    
    # Setup callbacks
    callbacks = []
    
    # Evaluation callback
    eval_env = make_vec_env(lambda: Monitor(create_optimized_environment(), "extended_logs/eval/"), n_envs=config["n_envs"])
    eval_env = VecTransposeImage(eval_env)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="extended_models/",
        log_path="extended_logs/eval/",
        eval_freq=config["eval_freq"],
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config["checkpoint_freq"],
        save_path="checkpoints/",
        name_prefix="ppo_clash_royale"
    )
    callbacks.append(checkpoint_callback)
    
    # Extended training callback
    extended_callback = ExtendedTrainingCallback("extended_logs/")
    callbacks.append(extended_callback)
    
    # Optional: Stop training on reward threshold
    # stop_callback = StopTrainingOnRewardThreshold(reward_threshold=100, verbose=1)
    # callbacks.append(stop_callback)
    
    print("6. Starting extended training...")
    print(f"   Total timesteps: {config['total_timesteps']:,}")
    print(f"   Parallel environments: {config['n_envs']}")
    print(f"   Estimated training time: ~{config['total_timesteps'] // (config['n_envs'] * 100)} minutes")
    print("   Monitor progress: tensorboard --logdir extended_tensorboard/")
    
    try:
        start_time = time.time()
        
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callbacks,
            tb_log_name="extended_ppo_5M",
            action_masks=vec_env.get_attr("action_mask")
        )
        
        training_time = time.time() - start_time
        print(f"7. Training completed in {training_time/3600:.2f} hours!")
        
        # Save final model with metadata
        final_model_path = "extended_models/ppo_clash_royale_5M_final"
        model.save(final_model_path)
        
        # Save training metadata
        metadata = {
            "training_time_hours": training_time / 3600,
            "total_timesteps": config["total_timesteps"],
            "final_model_path": final_model_path + ".zip",
            "config": config,
            "completion_time": datetime.now().isoformat()
        }
        
        with open("extended_models/training_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save training log
        extended_callback.save_training_log()
        
        print(f"   Final model saved: {final_model_path}.zip")
        print(f"   Training metadata saved: extended_models/training_metadata.json")
        
        return model, metadata
        
    except KeyboardInterrupt:
        print("\n   Training interrupted by user")
        interrupted_path = "extended_models/ppo_clash_royale_interrupted"
        model.save(interrupted_path)
        extended_callback.save_training_log()
        print(f"   Model saved: {interrupted_path}.zip")
        return model, {"status": "interrupted"}
        
    except Exception as e:
        print(f"\n   Error during training: {e}")
        error_path = "extended_models/ppo_clash_royale_error"
        model.save(error_path)
        extended_callback.save_training_log()
        return None, {"status": "error", "error": str(e)}
        
    finally:
        vec_env.close()
        eval_env.close()

if __name__ == "__main__":
    print("[GAME] Clash Royale Extended RL Training")
    print("=" * 60)
    
    # Check system requirements
    print("System Check:")
    print(f"  Python: {sys.version}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Device: {torch.cuda.get_device_name()}")
    print()
    
    # Start extended training
    model, metadata = train_extended_ppo()
    
    if model is not None and metadata.get("status") != "error":
        print("\n" + "=" * 60)
        print("[SUCCESS] EXTENDED TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Ready for transfer learning to real Clash Royale game!")
    else:
        print("\n" + "=" * 60)
        print("[WARNING] Training ended early - check logs for details")
        print("=" * 60)
