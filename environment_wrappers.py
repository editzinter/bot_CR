#!/usr/bin/env python3
"""
Environment wrappers for the Clash Royale environment.
These wrappers add preprocessing, reward shaping, and other enhancements.
"""

import gymnasium as gym
import numpy as np
from collections import deque
from typing import Dict, Any, Tuple, Optional
import cv2

class FrameStackWrapper(gym.Wrapper):
    """
    Stack multiple frames together to give the agent temporal information.
    """
    def __init__(self, env: gym.Env, num_stack: int = 4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        
        # Update observation space
        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Fill the frame stack with the initial observation
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_stacked_obs(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_stacked_obs(), reward, terminated, truncated, info
    
    def _get_stacked_obs(self):
        return np.array(self.frames)

class RewardShapingWrapper(gym.Wrapper):
    """
    Add reward shaping to encourage better strategies.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.previous_info = None
        self.step_count = 0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_info = info
        self.step_count = 0
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        
        # Apply reward shaping
        shaped_reward = self._shape_reward(reward, info, action)
        
        self.previous_info = info
        return obs, shaped_reward, terminated, truncated, info
    
    def _shape_reward(self, base_reward: float, info: Dict[str, Any], action: int) -> float:
        """
        Shape the reward to encourage good strategies.
        """
        shaped_reward = base_reward
        
        # Reward for surviving longer (small positive reward per step)
        shaped_reward += 0.001
        
        # Penalty for taking too long (encourage decisive play)
        if self.step_count > 1000:
            shaped_reward -= 0.01
        
        # Mock reward shaping based on game state
        if self.previous_info:
            # Reward for maintaining health
            health_diff = info.get('player_health', 100) - self.previous_info.get('player_health', 100)
            if health_diff > 0:
                shaped_reward += 0.1  # Reward for gaining health
            elif health_diff < 0:
                shaped_reward -= 0.05  # Penalty for losing health
            
            # Reward for damaging opponent
            opp_health_diff = info.get('opponent_health', 100) - self.previous_info.get('opponent_health', 100)
            if opp_health_diff < 0:
                shaped_reward += 0.2  # Reward for damaging opponent
        
        # Encourage elixir management
        elixir = info.get('elixir', 10)
        if elixir < 2:
            shaped_reward -= 0.02  # Penalty for low elixir
        elif elixir > 8:
            shaped_reward -= 0.01  # Slight penalty for hoarding elixir
        
        return shaped_reward

class ObservationNormalizationWrapper(gym.ObservationWrapper):
    """
    Normalize observations to [0, 1] range.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Update observation space to [0, 1] range
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, 
            shape=self.observation_space.shape, 
            dtype=np.float32
        )
    
    def observation(self, obs):
        # Normalize from [0, 255] to [0, 1]
        return obs.astype(np.float32) / 255.0

class ActionMaskingWrapper(gym.Wrapper):
    """
    Mask invalid actions to improve learning efficiency.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.valid_actions = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._update_valid_actions(info)
        return obs, info
    
    def step(self, action):
        # Ensure action is valid (for now, all actions are valid in our mock env)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_valid_actions(info)
        return obs, reward, terminated, truncated, info
    
    def _update_valid_actions(self, info: Dict[str, Any]):
        """
        Update the list of valid actions based on game state.
        In a real implementation, this would check elixir costs, card availability, etc.
        """
        # For now, all actions are valid in our mock environment
        self.valid_actions = list(range(self.action_space.n))

class EpisodeInfoWrapper(gym.Wrapper):
    """
    Track episode statistics for logging and analysis.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_stats = {}
        self.reset_stats()
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.reset_stats()
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update episode statistics
        self.episode_stats['total_reward'] += reward
        self.episode_stats['episode_length'] += 1
        self.episode_stats['actions_taken'].append(action)
        
        # Decode action for analysis
        if 'action_decoded' in info:
            decoded = info['action_decoded']
            self.episode_stats['card_plays'][decoded['card_index']] += 1
        
        # Add episode stats to info when episode ends
        if terminated or truncated:
            info['episode_stats'] = self.episode_stats.copy()
        
        return obs, reward, terminated, truncated, info
    
    def reset_stats(self):
        """Reset episode statistics."""
        self.episode_stats = {
            'total_reward': 0.0,
            'episode_length': 0,
            'actions_taken': [],
            'card_plays': [0, 0, 0, 0],  # Count for each card index
        }

def create_enhanced_environment(env_id: str = "clash-royale", **env_kwargs):
    """
    Create an enhanced Clash Royale environment with all wrappers applied.
    """
    import clash_royale
    import gymnasium
    
    # Create base environment
    env = gymnasium.make(env_id, **env_kwargs)
    
    # Apply wrappers in order
    env = EpisodeInfoWrapper(env)
    env = RewardShapingWrapper(env)
    env = ActionMaskingWrapper(env)
    env = ObservationNormalizationWrapper(env)
    env = FrameStackWrapper(env, num_stack=4)
    
    return env

# Example usage and testing
if __name__ == "__main__":
    print("Testing Enhanced Environment Wrappers")
    print("=" * 50)
    
    # Test the enhanced environment
    env = create_enhanced_environment(render_mode="rgb_array")
    
    print(f"Enhanced observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test a few steps
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, obs_shape={obs.shape}")
        
        if terminated or truncated:
            print("Episode ended")
            break
    
    env.close()
    print("Enhanced environment test completed!")
