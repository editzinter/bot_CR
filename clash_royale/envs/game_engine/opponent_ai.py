from __future__ import annotations
import random
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from clash_royale.envs.game_engine.game_engine import GameEngine


class OpponentAI:
    """
    A rule-based AI for the opponent (player 2).
    """

    def __init__(self, game_engine: GameEngine, player_id: int):
        """
        Initializes the OpponentAI.

        Args:
            game_engine: The game engine instance.
            player_id: The ID of the player the AI is controlling.
        """
        self.game_engine = game_engine
        self.player_id = player_id
        self.player = self.game_engine.player2

    def get_action(self) -> Optional[Tuple[int, int, int]]:
        """
        Determines the best action for the opponent to take based on a set of heuristics.
        Returns an action tuple (x, y, card_idx) or None if no action is taken.
        """
        # Defensive play
        action = self._defensive_play()
        if action:
            return action

        # Offensive play
        action = self._offensive_play()
        if action:
            return action

        # Default random play
        return self._random_play()

    def _defensive_play(self) -> Optional[Tuple[int, int, int]]:
        """
        Plays a defensive card if a tower is under attack.
        """
        attacking_entities = self.game_engine.arena.get_attacking_entities(self.player_id)
        if not attacking_entities:
            return None

        # Find a defensive card in hand
        defensive_cards = ["knight", "minions"]
        card_to_play = None
        card_idx = -1
        for i, card in enumerate(self.player.hand):
            if card.name in defensive_cards and i in self.player.get_pseudo_legal_cards():
                card_to_play = card
                card_idx = i
                break

        if not card_to_play:
            return None

        # Place the card near the first attacking entity
        target_entity = attacking_entities[0]
        placement_mask = self.game_engine.arena.get_placement_mask(team_id=self.player_id)

        # Try to place the card in front of the attacker
        x, y = target_entity.x, target_entity.y
        if self.player_id == 1: # Opponent is on top side
            y += 1
        else:
            y -= 1

        if 0 <= x < self.game_engine.width and 0 <= y < self.game_engine.height and placement_mask[y, x]:
            return (x, y, card_idx)

        # Fallback to a random valid placement
        valid_placements = list(zip(*placement_mask.nonzero()))
        if not valid_placements:
            return None
        y, x = random.choice(valid_placements)
        return (x, y, card_idx)

    def _offensive_play(self) -> Optional[Tuple[int, int, int]]:
        """
        Plays an offensive card if elixir is high.
        """
        if self.player.elixir < 7:
            return None

        # Find an offensive card in hand
        offensive_cards = ["giant"]
        card_to_play = None
        card_idx = -1
        for i, card in enumerate(self.player.hand):
            if card.name in offensive_cards and i in self.player.get_pseudo_legal_cards():
                card_to_play = card
                card_idx = i
                break

        if not card_to_play:
            return None

        # Place the card over the bridge in a random lane
        placement_mask = self.game_engine.arena.get_placement_mask(team_id=self.player_id)
        bridge_y = self.game_engine.height // 2

        # Choose a random lane (left or right)
        lane_x = random.choice([self.game_engine.width // 4, self.game_engine.width * 3 // 4])

        # Try to place at the bridge
        y = bridge_y if self.player_id == 1 else bridge_y - 1
        if placement_mask[y, lane_x]:
            return (lane_x, y, card_idx)

        # Fallback to a random valid placement
        valid_placements = list(zip(*placement_mask.nonzero()))
        if not valid_placements:
            return None
        y, x = random.choice(valid_placements)
        return (x, y, card_idx)

    def _random_play(self) -> Optional[Tuple[int, int, int]]:
        """
        Plays a random legal card at a random valid location.
        """
        legal_cards = self.player.get_pseudo_legal_cards()
        if not legal_cards:
            return None

        card_idx_to_play = random.choice(legal_cards)

        placement_mask = self.game_engine.arena.get_placement_mask(team_id=self.player_id)
        valid_placements = list(zip(*placement_mask.nonzero()))
        if not valid_placements:
            return None

        y, x = random.choice(valid_placements)

        return (x, y, card_idx_to_play)
