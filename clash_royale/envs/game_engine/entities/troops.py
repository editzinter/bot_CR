from __future__ import annotations

from clash_royale.envs.game_engine.entities.logic_entity import LogicEntity
from clash_royale.envs.game_engine.card_data import get_card_by_name
from clash_royale.envs.game_engine.logic.attack import SingleAttack
from clash_royale.envs.game_engine.logic.target import RadiusTarget
from clash_royale.envs.game_engine.logic.movement import SimpleMovement

class Knight(LogicEntity):
    """
    Knight troop entity.
    A melee troop with moderate health and damage.
    """
    def __init__(self, team_id: int, x: int, y: int):
        super().__init__(team_id, x, y)
        self.stats = get_card_by_name("knight").stats
        self.stats.team_id = team_id

        # Logic components
        self.target = RadiusTarget(self)
        self.attack = SingleAttack(self)
        self.movement = SimpleMovement(self)

class Archer(LogicEntity):
    """
    Archer troop entity.
    A ranged troop with low health and moderate damage.
    """
    def __init__(self, team_id: int, x: int, y: int):
        super().__init__(team_id, x, y)
        self.stats = get_card_by_name("archer").stats
        self.stats.team_id = team_id

        # Logic components
        self.target = RadiusTarget(self)
        self.attack = SingleAttack(self)
        self.movement = SimpleMovement(self)
