from __future__ import annotations
from typing import Tuple, Optional, Any, Dict

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from clash_royale.envs.game_engine.game_engine import GameEngine

class ClashRoyaleEnv(gym.Env):
    """
    Clash Royale Gymnasium Environment

    This environment uses a custom game engine to simulate Clash Royale matches.

    Action Space: Discrete(width * height * num_cards_in_hand)
    Observation Space: Box(0, 255, (resolution_x, resolution_y, 3)) - RGB image
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, width: int = 18, height: int = 32):
        super().__init__()

        self.width = width
        self.height = height
        self.resolution = (128, 128)

        self.game_engine = GameEngine(width=self.width, height=self.height, resolution=self.resolution)

        # Action space: x * y * z = 18 * 32 * 4 = 2304
        self.action_space = spaces.Discrete(self.width * self.height * 4)

        # Observation space: RGB image
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.resolution + (3,), dtype=np.uint8
        )

        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_info(self) -> Dict[str, Any]:
        """Returns info dictionary based on the current game state."""
        return {
            "elixir": self.game_engine.player1.elixir,
            "opponent_elixir": self.game_engine.player2.elixir,
            "player_tower_count": self.game_engine.arena.tower_count(0),
            "opponent_tower_count": self.game_engine.arena.tower_count(1),
        }

    def _calculate_reward(self, prev_state: Dict[str, Any]) -> float:
        """Calculates the reward based on the change in game state."""
        reward = 0.0

        # Reward for damaging opponent towers
        damage_to_opponent = prev_state["opponent_tower_health"] - self.game_engine.arena.lowest_tower_health(1)
        reward += damage_to_opponent * 0.01

        # Penalty for taking damage
        damage_to_player = prev_state["player_tower_health"] - self.game_engine.arena.lowest_tower_health(0)
        reward -= damage_to_player * 0.01

        return reward

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to its initial state."""
        super().reset(seed=seed)
        self.game_engine.reset()
        observation = self.game_engine.make_image()
        info = self._get_info()
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one step in the environment."""
        # Decode the flat action into coordinates and card index
        card_idx = action % 4
        y = (action // 4) % self.height
        x = action // (self.height * 4)
        decoded_action = (x, y, card_idx)

        # Store pre-step state for reward calculation
        prev_state = {
            "player_tower_health": self.game_engine.arena.lowest_tower_health(0),
            "opponent_tower_health": self.game_engine.arena.lowest_tower_health(1),
        }

        # Apply the action
        action_success = self.game_engine.apply_action(player_id=0, action=decoded_action)

        # If the action was illegal, apply a penalty
        if not action_success:
            reward = -0.1
        else:
            # If action was legal, step the engine and calculate reward
            self.game_engine.step()
            reward = self._calculate_reward(prev_state)

        # Check for terminal state
        terminated = self.game_engine.is_terminal()
        print(f"terminated: {terminated}, type: {type(terminated)}")
        if terminated:
            winner = self.game_engine.get_terminal_value()
            if winner == 0:  # Player wins
                reward += 10.0
            elif winner == 1: # Opponent wins
                reward -= 10.0

        observation = self.game_engine.make_image()
        info = self._get_info()
        truncated = False  # No truncation in this environment

        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Renders the environment."""
        if self.render_mode is None:
            return None

        canvas = self.game_engine.make_image()
        canvas = np.transpose(canvas, (1, 0, 2)) # Pygame and numpy have different coordinate systems

        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                self.window = pygame.display.set_mode(self.resolution)
            if self.clock is None:
                self.clock = pygame.time.Clock()

            surf = pygame.surfarray.make_surface(canvas)
            self.window.blit(surf, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:  # rgb_array
            return canvas

    def close(self):
        """Cleans up resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None