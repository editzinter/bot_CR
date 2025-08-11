"""
Base entity components
"""

from __future__ import annotations
from typing import TYPE_CHECKING, List
import pygame

from clash_royale.envs.game_engine.struct import Stats

if TYPE_CHECKING:
    from clash_royale.envs.game_engine.arena import Arena

class Entity:
    """
    Entity - a generic entity class.

    This class represents a basic entity present in the arena.
    It manages its own state, position, and stats.
    """
    id_counter = 0

    def __init__(self, team_id: int, x: int = 0, y: int = 0) -> None:
        self.id = Entity.id_counter
        Entity.id_counter += 1

        self.team_id = team_id
        self.x: int = x
        self.y: int = y

        self.stats: Stats = Stats()
        self.collection: Arena | None = None

    @property
    def health(self) -> int:
        """Returns the entity's current health."""
        return self.stats.health

    @property
    def is_alive(self) -> bool:
        """Checks if the entity's health is above zero."""
        return self.stats.health > 0

    def simulate(self) -> None:
        """
        Performs entity simulation for this frame.
        This method is where all the simulation will occur.
        """
        pass

    def render(self, surface: pygame.Surface) -> None:
        """
        Renders the entity on a Pygame surface.
        Default implementation draws a circle.
        """
        color = (0, 0, 255) if self.team_id == 0 else (255, 0, 0)
        pygame.draw.circle(surface, color, (self.x, self.y), self.stats.troop_size)

class EntityCollection:
    """
    EntityCollection - Manages a collection of entities.
    """

    def __init__(self) -> None:
        self.entities: List[Entity] = []

    def add_entity(self, entity: Entity) -> None:
        """
        Adds an entity to the collection.
        """
        entity.collection = self
        self.entities.append(entity)

    def remove_entity(self, entity: Entity) -> None:
        """
        Removes an entity from the collection.
        """
        if entity in self.entities:
            self.entities.remove(entity)

    def step(self) -> None:
        """
        Simulates one step for all entities in the collection.
        Removes entities that are no longer alive.
        """
        # Simulate each entity
        for entity in self.entities[:]:  # Iterate over a copy
            entity.simulate()

        # Remove dead entities
        self.entities = [entity for entity in self.entities if entity.is_alive]

    def render(self, surface: pygame.Surface) -> None:
        """
        Renders all entities in the collection.
        """
        for entity in self.entities:
            entity.render(surface)