"""
This file contains the GameEngine, the central component for managing the simulation.
"""

from __future__ import annotations
from clash_royale.envs.game_engine.entities.towers import KingTower
from typing import List, Tuple
import numpy as np
import numpy.typing as npt
import pygame
import random

from clash_royale.envs.game_engine.arena import Arena
from clash_royale.envs.game_engine.struct import Scheduler, DefaultScheduler, GAME_TICKS_PER_SECOND
from clash_royale.envs.game_engine.player import Player
from clash_royale.envs.game_engine.card import Card
from clash_royale.envs.game_engine.opponent_ai import OpponentAI

class GameEngine:
    """
    GameEngine - High-level simulation component

    This component is the entry point for the entire simulation.
    It manages the arena, players, and game rules, and renders the final result.
    """

    def __init__(
        self,
        width: int = 18,
        height: int = 32,
        resolution: Tuple[int, int] = (128, 128),
        deck1: Optional[List[str]] = None,
        deck2: Optional[List[str]] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.resolution = resolution

        # Define default decks if none are provided
        if deck1 is None:
            deck1 = ["knight", "archer", "knight", "archer", "knight", "archer", "knight", "archer"]
        if deck2 is None:
            deck2 = ["knight", "archer", "knight", "archer", "knight", "archer", "knight", "archer"]

        self.scheduler = Scheduler()
        self.game_scheduler = DefaultScheduler(self.scheduler)
        self.player1 = Player(deck1)
        self.player2 = Player(deck2)
        self.opponent_ai = OpponentAI(self, player_id=1)
        self.arena = Arena(width, height, self)

        pygame.init()
        self.screen = pygame.Surface(self.resolution)

    def reset(self) -> None:
        """
        Resets the game engine to its starting state.
        """
        self.scheduler.reset()
        self.player1.reset()
        self.player2.reset()
        self.arena.reset()

    def make_image(self) -> npt.NDArray[np.uint8]:
        """
        Renders the current game state to an image (numpy array).
        """
        # Simple gray background
        self.screen.fill((100, 100, 100))

        # Render entities
        self.arena.render(self.screen)

        # Convert to numpy array and return
        return np.array(pygame.surfarray.pixels3d(self.screen)).astype(np.uint8)

    def apply_action(self, player_id: int, action: Tuple[int, int, int]) -> bool:
        """
        Applies a given action for a player if it is legal.
        Action is a tuple: (x, y, card_index_in_hand).
        Returns True if the action was successful, False otherwise.
        """
        player = self.player1 if player_id == 0 else self.player2
        x, y, card_idx = action

        if card_idx not in player.get_pseudo_legal_cards():
            return False

        placement_mask = self.arena.get_placement_mask(player_id)
        if not placement_mask[y, x]:
            return False

        card_to_play = player.hand[card_idx]
        player.play_card(card_idx)
        self.arena.play_card(x, y, card_to_play, player_id)
        return True

    def _opponent_step(self) -> None:
        """
        The opponent (player 2) takes a step, controlled by the OpponentAI.
        """
        action = self.opponent_ai.get_action()
        if action:
            self.apply_action(player_id=1, action=action)

    def step(self) -> None:
        """
        Steps the simulation forward by one frame.
        """
        self.scheduler.step()
        elixir_rate = self.game_scheduler.elixir_rate()
        self.player1.step(elixir_rate)
        self.player2.step(elixir_rate)
        self.arena.step()
        self._opponent_step()

    def get_action_mask(self, player_id: int) -> npt.NDArray[np.bool_]:
        """
        Returns a flat mask of all legal actions for the given player.
        """
        player = self.player1 if player_id == 0 else self.player2
        mask = np.zeros(shape=(self.height, self.width, 4), dtype=bool)

        placement_mask = self.arena.get_placement_mask(player_id)
        legal_card_indices = player.get_pseudo_legal_cards()

        for card_idx in legal_card_indices:
            mask[:, :, card_idx] = placement_mask

        return mask.flatten()

    def is_terminal(self) -> bool:
        """
        Determines if the game has ended.
        """
        if self.game_scheduler.is_game_over():
            return True

        # Check if a king tower is destroyed
        king_towers_p1 = [e for e in self.arena.entities if isinstance(e, KingTower) and e.team_id == 0]
        king_towers_p2 = [e for e in self.arena.entities if isinstance(e, KingTower) and e.team_id == 1]
        if not king_towers_p1 or not king_towers_p2:
            return True

        return False

    def get_terminal_value(self) -> int:
        """
        Returns the winner of the game: 0 for player 1, 1 for player 2, -1 for a draw.
        """
        p1_towers = self.arena.tower_count(0)
        p2_towers = self.arena.tower_count(1)

        if p1_towers > p2_towers:
            return 0
        if p2_towers > p1_towers:
            return 1

        # Tie-breaker: lowest tower health
        p1_health = self.arena.lowest_tower_health(0)
        p2_health = self.arena.lowest_tower_health(1)

        if p1_health > p2_health:
            return 0
        if p2_health > p1_health:
            return 1

        return -1  # Draw