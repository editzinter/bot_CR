"""
This file contains the Arena component, which manages the game board and all entities.
"""

from typing import TYPE_CHECKING, List
import numpy as np
import numpy.typing as npt

from clash_royale.envs.game_engine.entities.entity import Entity, EntityCollection
from clash_royale.envs.game_engine.card import Card
from clash_royale.envs.game_engine.entities.towers import PrincessTower, KingTower
from clash_royale.envs.game_engine.entities.troops import Knight, Archer

if TYPE_CHECKING:
    from clash_royale.envs.game_engine.game_engine import GameEngine

class Arena(EntityCollection):
    """
    Arena

    This component handles the high-level logic for the game Arena.
    It contains all entities in play and manages their simulation and placement.
    """

    def __init__(self, width: int = 18, height: int = 32, engine: 'GameEngine' = None) -> None:
        super().__init__()
        self.width: int = width
        self.height: int = height
        self.engine: 'GameEngine' = engine

        # Define tower positions
        self.tower_pos = {
            0: { # Player
                'king': (width // 2, 4),
                'left_princess': (width // 4, 6),
                'right_princess': (width * 3 // 4, 6),
            },
            1: { # Opponent
                'king': (width // 2, height - 4),
                'left_princess': (width // 4, height - 6),
                'right_princess': (width * 3 // 4, height - 6),
            }
        }

    def reset(self) -> None:
        """
        Resets the arena to its starting state.
        Clears all entities and places the towers for both teams.
        """
        self.entities.clear()

        # Add towers for player (team_id=0)
        self.add_entity(KingTower(0, *self.tower_pos[0]['king']))
        self.add_entity(PrincessTower(0, *self.tower_pos[0]['left_princess']))
        self.add_entity(PrincessTower(0, *self.tower_pos[0]['right_princess']))

        # Add towers for opponent (team_id=1)
        self.add_entity(KingTower(1, *self.tower_pos[1]['king']))
        self.add_entity(PrincessTower(1, *self.tower_pos[1]['left_princess']))
        self.add_entity(PrincessTower(1, *self.tower_pos[1]['right_princess']))

    def play_card(self, x: int, y: int, card: Card, team_id: int) -> None:
        """
        Plays a card by creating its corresponding entity in the arena.
        """
        entity_class = None
        if card.name == 'knight':
            entity_class = Knight
        elif card.name == 'archer':
            entity_class = Archer

        if entity_class:
            new_entity = entity_class(team_id=team_id, x=x, y=y)
            self.add_entity(new_entity)

    def get_placement_mask(self, team_id: int) -> npt.NDArray[np.bool_]:
        """
        Returns a boolean mask of the arena where a player can place troops.
        For simplicity, allows placement on the player's half of the arena.
        """
        mask = np.zeros(shape=(self.height, self.width), dtype=bool)
        if team_id == 0:
            mask[:self.height // 2, :] = True  # Player can place on bottom half
        else:
            mask[self.height // 2:, :] = True  # Opponent can place on top half
        return mask

    def tower_count(self, team_id: int) -> int:
        """
        Returns the number of standing towers for a given team.
        """
        count = 0
        for entity in self.entities:
            if entity.stats.team_id == team_id and isinstance(entity, (PrincessTower, KingTower)):
                count += 1
        return count

    def lowest_tower_health(self, team_id: int) -> int:
        """
        Returns the lowest health percentage among all towers of a given team.
        Used for tie-breaking.
        """
        lowest_health = float('inf')
        towers = [e for e in self.entities if e.stats.team_id == team_id and isinstance(e, (PrincessTower, KingTower))]
        if not towers:
            return 0

        for tower in towers:
            # A simple health value, not percentage, is fine for tie-breaking
            if tower.stats.health < lowest_health:
                lowest_health = tower.stats.health
        return int(lowest_health)