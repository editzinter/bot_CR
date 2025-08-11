"""
Logic components for targeting.

These components describe how targeting is performed.
"""

from typing import TYPE_CHECKING, List
import math

from clash_royale.envs.game_engine.entities.entity import Entity
from clash_royale.envs.game_engine.utils import distance

if TYPE_CHECKING:
    from clash_royale.envs.game_engine.entities.logic_entity import LogicEntity
    from clash_royale.envs.game_engine.entities.towers import KingTower, PrincessTower

class BaseTarget:
    """
    BaseTarget - Class all target components must inherit!
    """
    def __init__(self, entity: 'LogicEntity') -> None:
        self.entity = entity

    def find_target(self) -> Entity | None:
        """
        Finds a target in the arena and returns an entity.
        """
        raise NotImplementedError("Must be implemented in child class!")

class RadiusTarget(BaseTarget):
    """
    Finds the closest enemy entity within the sight radius.
    Can be configured to target only buildings.
    """
    def __init__(self, entity: 'LogicEntity', target_buildings: bool = False) -> None:
        super().__init__(entity)
        self.target_buildings = target_buildings

    def find_target(self) -> Entity | None:
        """
        Finds the closest valid target within the entity's sight range.
        A valid target is any entity from the opposing team.
        If target_buildings is True, only considers towers.
        """
        if self.entity.collection is None:
            return None

        closest_target: Entity | None = None
        min_dist = float('inf')

        # Get all entities from the opponent's team
        from clash_royale.envs.game_engine.entities.towers import KingTower, PrincessTower

        opponent_team_id = 1 - self.entity.team_id
        opponent_entities = [e for e in self.entity.collection.entities if e.team_id == opponent_team_id]

        if self.target_buildings:
            opponent_entities = [e for e in opponent_entities if isinstance(e, (KingTower, PrincessTower))]

        for target in opponent_entities:
            dist = distance(self.entity.x, self.entity.y, target.x, target.y)

            if dist <= self.entity.stats.sight_range:
                if dist < min_dist:
                    min_dist = dist
                    closest_target = target

        return closest_target