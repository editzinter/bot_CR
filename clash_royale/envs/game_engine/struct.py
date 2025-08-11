"""
Various structures to be utilized
"""

from __future__ import annotations
import dataclasses

# Game Ticks Per Second
# This is the number of times the game logic will be updated per second.
# All game timings are based on this value.
GAME_TICKS_PER_SECOND = 30

@dataclasses.dataclass(slots=True)
class Stats:
    """
    Stats - Various stats to be utilized by entities

    This class contains definitions for entity stats,
    which describe the current state of the entity (health),
    and behavior of the entity (speed, damage, range, ect.)
    """

    name: str = ''
    health: int = 0
    damage: int = 0
    attack_range: int = 0  # In grid units
    sight_range: int = 0  # In grid units
    speed: int = 0  # Grid units per second
    attack_delay: int = 0  # In game ticks
    elixir_cost: int = 0
    troop_size: int = 1  # Radius in grid units
    team_id: int = 0  # 0 for player, 1 for opponent

class Scheduler:
    """
    Scheduling class to handle all timings.
    """
    def __init__(self):
        self.frame_num: int = 0

    def reset(self):
        self.frame_num = 0

    def step(self, frames: int = 1):
        self.frame_num += frames

    def frame(self) -> int:
        return self.frame_num

class GameScheduler:
    """
    Template class for game scheduling
    """
    def __init__(self, scheduler: Scheduler):
        self.scheduler: Scheduler = scheduler

class DefaultScheduler(GameScheduler):
    """
    Class for default 1v1 game scheduling
    """
    SINGLE_ELIXIR_FRAMES = 2 * 60 * GAME_TICKS_PER_SECOND  # 2 minutes
    DOUBLE_ELIXIR_FRAMES = 1 * 60 * GAME_TICKS_PER_SECOND  # 1 minute
    TOTAL_GAME_FRAMES = SINGLE_ELIXIR_FRAMES + DOUBLE_ELIXIR_FRAMES

    def elixir_rate(self) -> float:
        """
        Returns the number of elixir points to generate per frame.
        - 1 elixir every 2.8 seconds during single elixir time.
        - 1 elixir every 1.4 seconds during double elixir time.
        """
        if self.scheduler.frame() < self.SINGLE_ELIXIR_FRAMES:
            return 1.0 / (2.8 * GAME_TICKS_PER_SECOND)
        else:
            return 1.0 / (1.4 * GAME_TICKS_PER_SECOND)

    def is_game_over(self) -> bool:
        """
        Determines if the game has ended based on time.
        """
        return self.scheduler.frame() >= self.TOTAL_GAME_FRAMES

    def is_overtime(self) -> bool:
        """
        Determines if the game is in overtime (double elixir).
        """
        return self.scheduler.frame() >= self.SINGLE_ELIXIR_FRAMES