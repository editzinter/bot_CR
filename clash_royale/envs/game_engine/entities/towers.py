from __future__ import annotations

from clash_royale.envs.game_engine.entities.logic_entity import LogicEntity
from clash_royale.envs.game_engine.card_data import get_card_by_name
from clash_royale.envs.game_engine.logic.attack import SingleAttack
from clash_royale.envs.game_engine.logic.target import RadiusTarget

class PrincessTower(LogicEntity):
    """
    Princess Tower entity.
    It has targeting and attacking capabilities but does not move.
    """
    def __init__(self, team_id: int, x: int, y: int):
        super().__init__(team_id, x, y)
        self.stats = get_card_by_name("princess_tower").stats
        self.stats.team_id = team_id

        # Logic components
        self.target = RadiusTarget(self)
        self.attack = SingleAttack(self)
        self.movement = None  # Towers don't move

class KingTower(LogicEntity):
    """
    King Tower entity.
    It has targeting and attacking capabilities but does not move.
    It typically has higher health and different attack stats.
    """
    def __init__(self, team_id: int, x: int, y: int):
        super().__init__(team_id, x, y)
        self.stats = get_card_by_name("king_tower").stats
        self.stats.team_id = team_id

        # Logic components
        self.target = RadiusTarget(self)
        self.attack = SingleAttack(self)
        self.movement = None  # Towers don't move
