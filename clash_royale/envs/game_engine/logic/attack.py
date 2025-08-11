"""
Logic components for attacking.

These components describe how an entity attacks another.
"""

from typing import TYPE_CHECKING

from clash_royale.envs.game_engine.utils import distance

if TYPE_CHECKING:
    from clash_royale.envs.game_engine.entities.logic_entity import LogicEntity

class BaseAttack:
    """
    BaseAttack - Class all attacks should inherit!
    """
    def __init__(self, entity: 'LogicEntity') -> None:
        self.entity = entity
        self.last_attack_frame: int = -1000  # Allow attacking immediately

    def can_attack(self) -> bool:
        """
        Determines if this entity can attack its current target.
        """
        if self.entity.target_entity is None:
            return False

        # Check if attack is off cooldown
        current_frame = self.entity.collection.engine.scheduler.frame()
        if current_frame < self.last_attack_frame + self.entity.stats.attack_delay:
            return False

        # Check if target is in range
        dist = distance(self.entity.x, self.entity.y, self.entity.target_entity.x, self.entity.target_entity.y)
        if dist > self.entity.stats.attack_range:
            return False

        return True

    def attack(self) -> None:
        """
        Performs an attack on a target, if possible.
        """
        raise NotImplementedError("Must implement this function!")

class SingleAttack(BaseAttack):
    """
    SingleAttack - An attack on a singular entity.
    Deals damage to the target's health.
    """
    def attack(self):
        """
        Performs an attack operation if possible.
        """
        if self.can_attack():
            # Deal damage to the target
            self.entity.target_entity.stats.health -= self.entity.stats.damage

            # Reset attack cooldown
            self.last_attack_frame = self.entity.collection.engine.scheduler.frame()