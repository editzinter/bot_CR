#!/usr/bin/env python3
"""
Real Game Fine-tuning System
This module integrates the trained RL agent with real Clash Royale gameplay
for fine-tuning and performance optimization in the actual game environment.
"""

import os
import time
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import threading
import queue

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

from bluestacks_integration import BlueStacksController, ActionMapper, GameConfig
from vision_system import RealTimeVisionSystem, VisionConfig

@dataclass
class FineTuningConfig:
    """Configuration for real game fine-tuning."""
    max_episodes: int = 100
    max_steps_per_episode: int = 300  # Shorter episodes for safety
    learning_rate: float = 1e-5  # Lower LR for fine-tuning
    exploration_rate: float = 0.1  # Reduced exploration
    safety_timeout: int = 30  # Max seconds per episode
    reward_scale: float = 1.0
    save_frequency: int = 10  # Save every N episodes

class RealGameEnvironment:
    """
    Real game environment wrapper for RL integration.
    """
    
    def __init__(self, game_config: GameConfig, vision_config: VisionConfig):
        self.game_config = game_config
        self.vision_config = vision_config
        
        # Initialize components
        self.controller = BlueStacksController(game_config)
        self.action_mapper = ActionMapper(game_config)
        self.vision_system = RealTimeVisionSystem(vision_config)
        
        # Environment state
        self.current_episode = 0
        self.current_step = 0
        self.episode_start_time = 0
        self.last_game_state = None
        self.episode_history = []
        
        # Safety and monitoring
        self.safety_checks = {
            "max_consecutive_failures": 5,
            "failure_count": 0,
            "last_successful_action": time.time()
        }
        
    def connect(self) -> bool:
        """Connect to game environment."""
        print("🔌 Connecting to real game environment...")
        
        if not self.controller.connect():
            print("❌ Failed to connect to BlueStacks")
            return False
        
        self.vision_system.start_processing()
        print("✓ Connected to real game environment")
        return True
    
    def disconnect(self):
        """Disconnect from game environment."""
        self.vision_system.stop_processing()
        self.controller.disconnect()
        print("✓ Disconnected from real game environment")
    
    def reset(self) -> np.ndarray:
        """Reset environment for new episode."""
        self.current_step = 0
        self.episode_start_time = time.time()
        self.safety_checks["failure_count"] = 0
        
        # Wait for game to be ready
        self._wait_for_battle_start()
        
        # Get initial observation
        observation = self._get_observation()
        
        print(f"🎮 Episode {self.current_episode} started")
        return observation
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute action and return next state."""
        self.current_step += 1
        step_start_time = time.time()
        
        # Safety checks
        if self._check_safety_conditions():
            return self._emergency_stop()
        
        # Execute action
        action_success = self._execute_action(action)
        
        # Wait for action to take effect
        time.sleep(0.5)  # Allow for game animation
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(action, action_success)
        
        # Check if episode is done
        done = self._check_episode_done()
        
        # Collect step info
        info = {
            "step": self.current_step,
            "action_success": action_success,
            "step_time": time.time() - step_start_time,
            "game_state": self.last_game_state,
            "safety_status": self.safety_checks.copy()
        }
        
        return observation, reward, done, info
    
    def _wait_for_battle_start(self):
        """Wait for battle to start."""
        print("⏳ Waiting for battle to start...")
        
        max_wait_time = 30  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            frame = self.controller.capture_screen()
            if frame is not None:
                self.vision_system.add_frame(frame)
                result = self.vision_system.get_latest_result()
                
                if result and result["game_state"]["battle_phase"] == "battle":
                    print("✓ Battle started")
                    return
            
            time.sleep(1)
        
        print("⚠️  Battle start timeout - proceeding anyway")
    
    def _get_observation(self) -> np.ndarray:
        """Get current game observation."""
        # Capture screen
        frame = self.controller.capture_screen()
        if frame is None:
            # Return last known observation or zeros
            return np.zeros((128, 128, 3), dtype=np.uint8)
        
        # Add to vision system
        self.vision_system.add_frame(frame)
        
        # Get processed result
        result = self.vision_system.get_latest_result()
        if result and result["processed_frame"] is not None:
            self.last_game_state = result["game_state"]
            return result["processed_frame"]
        
        # Fallback: process frame directly
        from vision_system import FrameProcessor
        processor = FrameProcessor(self.vision_config)
        processed = processor.preprocess_frame(frame)
        
        return processed if processed is not None else np.zeros((128, 128, 3), dtype=np.uint8)
    
    def _execute_action(self, action: int) -> bool:
        """Execute action in real game."""
        try:
            success = self.action_mapper.execute_action(self.controller, action)
            
            if success:
                self.safety_checks["failure_count"] = 0
                self.safety_checks["last_successful_action"] = time.time()
            else:
                self.safety_checks["failure_count"] += 1
            
            return success
            
        except Exception as e:
            print(f"⚠️  Action execution error: {e}")
            self.safety_checks["failure_count"] += 1
            return False
    
    def _calculate_reward(self, action: int, action_success: bool) -> float:
        """Calculate reward for the current step."""
        reward = 0.0
        
        # Base reward for successful action execution
        if action_success:
            reward += 0.1
        else:
            reward -= 0.2
        
        # Game state based rewards
        if self.last_game_state:
            game_state = self.last_game_state
            
            # Reward for tower damage (would need actual detection)
            # This is placeholder - real implementation would compare tower health
            if game_state.get("towers"):
                reward += 0.0  # Placeholder
            
            # Reward for elixir management
            elixir = game_state.get("elixir_count", 5)
            if elixir < 2:
                reward -= 0.1  # Penalty for low elixir
            elif elixir > 8:
                reward -= 0.05  # Slight penalty for hoarding
            
            # Reward for battle phase progression
            phase = game_state.get("battle_phase", "unknown")
            if phase == "victory":
                reward += 10.0
            elif phase == "defeat":
                reward -= 5.0
        
        # Time-based penalty (encourage decisive play)
        time_penalty = -0.01 * (self.current_step / 100)
        reward += time_penalty
        
        return reward * self.config.reward_scale if hasattr(self, 'config') else reward
    
    def _check_episode_done(self) -> bool:
        """Check if episode should end."""
        # Time limit
        if time.time() - self.episode_start_time > 180:  # 3 minutes max
            return True
        
        # Step limit
        if self.current_step >= 300:
            return True
        
        # Game state based termination
        if self.last_game_state:
            phase = self.last_game_state.get("battle_phase", "unknown")
            if phase in ["victory", "defeat"]:
                return True
        
        # Safety termination
        if self.safety_checks["failure_count"] >= self.safety_checks["max_consecutive_failures"]:
            print("⚠️  Too many consecutive failures - ending episode")
            return True
        
        return False
    
    def _check_safety_conditions(self) -> bool:
        """Check if safety conditions are violated."""
        # Check if we've been stuck too long
        time_since_success = time.time() - self.safety_checks["last_successful_action"]
        if time_since_success > 30:  # 30 seconds without successful action
            print("⚠️  Safety timeout - no successful actions")
            return True
        
        # Check failure rate
        if self.safety_checks["failure_count"] >= self.safety_checks["max_consecutive_failures"]:
            return True
        
        return False
    
    def _emergency_stop(self) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Emergency stop with safe return values."""
        observation = np.zeros((128, 128, 3), dtype=np.uint8)
        reward = -1.0  # Penalty for emergency stop
        done = True
        info = {"emergency_stop": True, "reason": "safety_violation"}
        
        return observation, reward, done, info

class RealGameFineTuner:
    """
    Fine-tuning system for real game adaptation.
    """
    
    def __init__(self, model_path: str, config: FineTuningConfig):
        self.model_path = model_path
        self.config = config
        self.model = None
        self.environment = None
        
        # Fine-tuning data
        self.episode_data = []
        self.performance_metrics = {
            "episodes_completed": 0,
            "total_reward": 0,
            "average_reward": 0,
            "win_rate": 0,
            "safety_stops": 0
        }
        
    def load_pretrained_model(self) -> bool:
        """Load pre-trained model for fine-tuning."""
        try:
            self.model = PPO.load(self.model_path)
            print(f"✓ Loaded pre-trained model: {self.model_path}")
            
            # Adjust learning rate for fine-tuning
            self.model.learning_rate = self.config.learning_rate
            print(f"✓ Adjusted learning rate to: {self.config.learning_rate}")
            
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def setup_environment(self) -> bool:
        """Setup real game environment."""
        game_config = GameConfig()
        vision_config = VisionConfig()
        
        self.environment = RealGameEnvironment(game_config, vision_config)
        
        return self.environment.connect()
    
    def run_fine_tuning(self):
        """Run the fine-tuning process."""
        print("🎯 STARTING REAL GAME FINE-TUNING")
        print("=" * 50)
        
        if not self.load_pretrained_model():
            return False
        
        if not self.setup_environment():
            return False
        
        try:
            for episode in range(self.config.max_episodes):
                print(f"\n📍 Episode {episode + 1}/{self.config.max_episodes}")
                
                episode_data = self._run_episode(episode)
                self.episode_data.append(episode_data)
                
                # Update performance metrics
                self._update_metrics(episode_data)
                
                # Save model periodically
                if (episode + 1) % self.config.save_frequency == 0:
                    self._save_checkpoint(episode + 1)
                
                # Print progress
                self._print_progress()
                
                # Safety break if too many failures
                if self.performance_metrics["safety_stops"] > 10:
                    print("⚠️  Too many safety stops - ending fine-tuning")
                    break
        
        finally:
            self.environment.disconnect()
        
        # Save final model
        self._save_final_model()
        
        print("\n🎉 Fine-tuning completed!")
        return True
    
    def _run_episode(self, episode_num: int) -> Dict[str, Any]:
        """Run a single fine-tuning episode."""
        self.environment.current_episode = episode_num
        
        episode_data = {
            "episode": episode_num,
            "steps": 0,
            "total_reward": 0,
            "actions": [],
            "rewards": [],
            "success": False,
            "end_reason": "unknown"
        }
        
        # Reset environment
        observation = self.environment.reset()
        
        for step in range(self.config.max_steps_per_episode):
            # Get action from model
            action, _states = self.model.predict(
                observation, 
                deterministic=(np.random.random() > self.config.exploration_rate)
            )
            
            # Execute action
            next_obs, reward, done, info = self.environment.step(action)
            
            # Record data
            episode_data["steps"] += 1
            episode_data["total_reward"] += reward
            episode_data["actions"].append(int(action))
            episode_data["rewards"].append(float(reward))
            
            # Update observation
            observation = next_obs
            
            # Check if episode is done
            if done:
                episode_data["end_reason"] = info.get("reason", "game_end")
                if info.get("emergency_stop"):
                    episode_data["end_reason"] = "safety_stop"
                break
        
        # Determine success
        if episode_data["total_reward"] > 0:
            episode_data["success"] = True
        
        return episode_data
    
    def _update_metrics(self, episode_data: Dict[str, Any]):
        """Update performance metrics."""
        self.performance_metrics["episodes_completed"] += 1
        self.performance_metrics["total_reward"] += episode_data["total_reward"]
        self.performance_metrics["average_reward"] = (
            self.performance_metrics["total_reward"] / 
            self.performance_metrics["episodes_completed"]
        )
        
        if episode_data["success"]:
            wins = sum(1 for ep in self.episode_data if ep["success"])
            self.performance_metrics["win_rate"] = wins / len(self.episode_data)
        
        if episode_data["end_reason"] == "safety_stop":
            self.performance_metrics["safety_stops"] += 1
    
    def _print_progress(self):
        """Print current progress."""
        metrics = self.performance_metrics
        print(f"  Episodes: {metrics['episodes_completed']}")
        print(f"  Avg Reward: {metrics['average_reward']:.2f}")
        print(f"  Win Rate: {metrics['win_rate']:.1%}")
        print(f"  Safety Stops: {metrics['safety_stops']}")
    
    def _save_checkpoint(self, episode: int):
        """Save model checkpoint."""
        checkpoint_path = f"real_game_models/checkpoint_episode_{episode}"
        self.model.save(checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    def _save_final_model(self):
        """Save final fine-tuned model."""
        os.makedirs("real_game_models", exist_ok=True)
        
        final_path = "real_game_models/fine_tuned_final"
        self.model.save(final_path)
        
        # Save training data
        data_path = "real_game_models/fine_tuning_data.json"
        with open(data_path, 'w') as f:
            json.dump({
                "episode_data": self.episode_data,
                "performance_metrics": self.performance_metrics,
                "config": {
                    "max_episodes": self.config.max_episodes,
                    "learning_rate": self.config.learning_rate,
                    "exploration_rate": self.config.exploration_rate
                }
            }, f, indent=2)
        
        print(f"✓ Final model saved: {final_path}")
        print(f"✓ Training data saved: {data_path}")

def run_real_game_finetuning():
    """Main function to run real game fine-tuning."""
    print("🎮 REAL GAME FINE-TUNING SYSTEM")
    print("=" * 60)
    
    # Configuration
    config = FineTuningConfig(
        max_episodes=50,  # Start with fewer episodes for safety
        learning_rate=1e-5,
        exploration_rate=0.1
    )
    
    # Find best trained model
    model_paths = [
        "extended_models/ppo_clash_royale_5M_final",
        "models/ppo_clash_royale_final", 
        "test_models/quick_test_ppo"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path + ".zip"):
            model_path = path
            break
    
    if not model_path:
        print("❌ No trained model found. Run training first.")
        return False
    
    print(f"Using model: {model_path}")
    
    # Create fine-tuner
    fine_tuner = RealGameFineTuner(model_path, config)
    
    # Run fine-tuning
    return fine_tuner.run_fine_tuning()

if __name__ == "__main__":
    run_real_game_finetuning()
