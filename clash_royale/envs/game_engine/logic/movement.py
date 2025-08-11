"""
Logic components for movement.

These components describe how movement is performed.
"""

import math
from typing import TYPE_CHECKING

from clash_royale.envs.game_engine.struct import GAME_TICKS_PER_SECOND

if TYPE_CHECKING:
    from clash_royale.envs.game_engine.entities.logic_entity import LogicEntity

class BaseMovement:
    """
    BaseMovement - Class all movement components must inherit!
    """
    def __init__(self, entity: 'LogicEntity') -> None:
        self.entity = entity

    def move(self) -> None:
        """
        Moves the entity towards its target.
        """
        raise NotImplementedError("MUST implement this method!")

class SimpleMovement(BaseMovement):
    """
    SimpleMovement - Simply move in a straight line to the target.
    """
    def move(self) -> None:
        """
        Moves in a straight line towards the target entity.
        If there is no target, the entity does not move.
        """
        if self.entity.target_entity is None:
            return

        target = self.entity.target_entity
        dx = target.x - self.entity.x
        dy = target.y - self.entity.y

        distance = math.sqrt(dx**2 + dy**2)
        if distance == 0:
            return

        # Move only if not already in attack range
        if distance > self.entity.stats.attack_range:
            # Calculate movement for this frame
            # Speed is in grid units per second, so we divide by ticks per second
            frame_speed = self.entity.stats.speed / GAME_TICKS_PER_SECOND

            self.entity.x += frame_speed * (dx / distance)
            self.entity.y += frame_speed * (dy / distance)