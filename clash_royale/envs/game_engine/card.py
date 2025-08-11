from __future__ import annotations
from clash_royale.envs.game_engine.struct import Stats

class Card():
    '''
    Card class.

    This class holds the necessary information to create a game entity.
    It stores the name, elixir cost, and the stats of the troop to be spawned.
    '''

    def __init__(self, name: str, elixir_cost: int, stats: Stats) -> None:
        self.name = name
        self.elixir_cost = elixir_cost
        self.stats = stats
        self.stats.elixir_cost = elixir_cost
        self.stats.name = name