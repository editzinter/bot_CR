"""
Entities that utilize the logical framework
"""

from clash_royale.envs.game_engine.entities.entity import Entity
from clash_royale.envs.game_engine.logic.attack import BaseAttack
from clash_royale.envs.game_engine.logic.target import BaseTarget
from clash_royale.envs.game_engine.logic.movement import BaseMovement

class LogicEntity(Entity):
    """
    An entity that operates within our logical framework.

    This entity allows for logic components to be attached,
    allowing for behavior to be generalized and attached to entities.
    """

    def __init__(self, team_id: int, x: int, y: int) -> None:
        super().__init__(team_id, x, y)

        self.attack: BaseAttack | None = None
        self.target: BaseTarget | None = None
        self.movement: BaseMovement | None = None

        self.target_entity: Entity | None = None

    def simulate(self):
        """
        Performs an entity simulation by executing logic components in order:
        1. Find a target.
        2. Attack the target if possible.
        3. Move towards the target.
        """

        # 1. Find a target if we don't have one or if the current one is dead
        if self.target is not None:
            if self.target_entity is None or not self.target_entity.is_alive:
                self.target_entity = self.target.find_target()

        # 2. Attack the target if we have one and can attack
        if self.attack is not None and self.target_entity is not None:
            self.attack.attack()

        # 3. Move towards the target if we have one
        if self.movement is not None and self.target_entity is not None:
            self.movement.move()