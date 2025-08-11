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

from stable_baselines3.common.vec_env import VecEnvWrapper

class ActionMaskingWrapper(VecEnvWrapper):
    """
    A wrapper to handle action masking for vectorized environments.
    """
    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        for i, info in enumerate(infos):
            if "action_mask" in info:
                self.action_masks[i] = info["action_mask"]
        return observations, rewards, dones, infos

    def reset(self):
        obs = self.venv.reset()
        # Does not return action mask on reset, so we need to get it from the info dict
        return obs
    
    def action_masks(self) -> np.ndarray:
        return self.venv.get_attr("action_mask")

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

class DeckSamplingWrapper(gym.Wrapper):
    """
    Samples decks from a pool and assigns them to players at the start of each episode.
    """
    def __init__(self, env: gym.Env, deck_pool: list[list[str]]):
        super().__init__(env)
        self.deck_pool = deck_pool

    def reset(self, **kwargs):
        """
        Samples two decks and passes them to the environment's reset method.
        """
        deck1 = random.choice(self.deck_pool)
        deck2 = random.choice(self.deck_pool)
        if 'options' not in kwargs:
            kwargs['options'] = {}
        kwargs['options']['deck1'] = deck1
        kwargs['options']['deck2'] = deck2
        return self.env.reset(**kwargs)

class ClashRoyaleRewardWrapper(gym.Wrapper):
    """
    A wrapper that provides dense rewards for in-game events.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.prev_info = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_info = self._get_info_state()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_info_state = self._get_info_state()
        shaped_reward = self._shape_reward(reward, current_info_state)
        self.prev_info = current_info_state

        return obs, shaped_reward, terminated, truncated, info

    def _get_info_state(self) -> Dict[str, Any]:
        """
        Gathers and returns the current state of the game from the environment.
        """
        return self.env.get_state()

    def _shape_reward(self, base_reward: float, current_info: dict) -> float:
        """
        Calculates the shaped reward based on the change in game state.
        """
        shaped_reward = base_reward

        # Calculate damage dealt and received
        prev_entities = {e['id']: e for e in self.prev_info['entities']}
        current_entities = {e['id']: e for e in current_info['entities']}

        for entity_id, current_entity in current_entities.items():
            if entity_id in prev_entities:
                prev_entity = prev_entities[entity_id]
                health_diff = current_entity['health'] - prev_entity['health']
                if health_diff < 0:
                    if current_entity['team_id'] == 0: # Player's troop took damage
                        shaped_reward -= 0.01 * abs(health_diff)
                    else: # Opponent's troop took damage
                        shaped_reward += 0.01 * abs(health_diff)

        # Calculate elixir trade
        prev_player_elixir = self.prev_info.get('player1_elixir', 0)
        current_player_elixir = current_info.get('player1_elixir', 0)
        elixir_diff = current_player_elixir - prev_player_elixir
        shaped_reward += 0.001 * elixir_diff

        # Penalize losing troops
        for entity_id, prev_entity in prev_entities.items():
            if entity_id not in current_entities:
                if prev_entity['team_id'] == 0: # Player's troop died
                    shaped_reward -= 0.1 * prev_entity['elixir_cost']
                else: # Opponent's troop died
                    shaped_reward += 0.1 * prev_entity['elixir_cost']

        return shaped_reward

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
