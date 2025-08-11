from clash_royale.envs.game_engine.card import Card
from clash_royale.envs.game_engine.struct import Stats, GAME_TICKS_PER_SECOND

# A central database for all card statistics in the game.
# All stats are defined here and then used to create Card objects.

# Troop Stats
KNIGHT_STATS = Stats(
    health=1400,
    damage=160,
    attack_range=1,
    sight_range=5,
    speed=1.0,  # 1 grid unit per second
    attack_delay=int(1.1 * GAME_TICKS_PER_SECOND), # 1.1 seconds
)

ARCHER_STATS = Stats(
    health=250,
    damage=86,
    attack_range=5,
    sight_range=5.5,
    speed=1.0,
    attack_delay=int(1.2 * GAME_TICKS_PER_SECOND),
)

GIANT_STATS = Stats(
    health=3300,
    damage=211,
    attack_range=1,
    sight_range=5,
    speed=0.5,
    attack_delay=int(1.5 * GAME_TICKS_PER_SECOND),
)

MINION_STATS = Stats(
    health=190,
    damage=84,
    attack_range=2,
    sight_range=5,
    speed=1.5,
    attack_delay=int(1.0 * GAME_TICKS_PER_SECOND),
)

# Tower Stats
PRINCESS_TOWER_STATS = Stats(
    health=2500,
    damage=90,
    attack_range=7,
    sight_range=7.5,
    speed=0,
    attack_delay=int(0.8 * GAME_TICKS_PER_SECOND),
)

KING_TOWER_STATS = Stats(
    health=4000,
    damage=60,
    attack_range=7,
    sight_range=7.5,
    speed=0,
    attack_delay=int(1.0 * GAME_TICKS_PER_SECOND),
)

# Card Definitions
CARD_DATABASE = {
    "knight": Card(name="knight", elixir_cost=3, stats=KNIGHT_STATS),
    "archer": Card(name="archer", elixir_cost=3, stats=ARCHER_STATS),
    "giant": Card(name="giant", elixir_cost=5, stats=GIANT_STATS),
    "minions": Card(name="minions", elixir_cost=3, stats=MINION_STATS),

    # Towers are treated as special cards for instantiation
    "princess_tower": Card(name="princess_tower", elixir_cost=0, stats=PRINCESS_TOWER_STATS),
    "king_tower": Card(name="king_tower", elixir_cost=0, stats=KING_TOWER_STATS),
}

def get_card_by_name(name: str) -> Card:
    """Fetches a card from the database by its name."""
    card = CARD_DATABASE.get(name.lower())
    if card is None:
        raise ValueError(f"Card with name '{name}' not found in the database.")
    return card
