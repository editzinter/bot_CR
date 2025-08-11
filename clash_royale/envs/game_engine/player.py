from typing import List
import random

from clash_royale.envs.game_engine.card import Card
from clash_royale.envs.game_engine.card_data import get_card_by_name

class Player():
    """
    Player class

    This class represents the current state of a player's cards and elixir.
    It handles the logic of drawing, playing, and cycling cards.
    """

    def __init__(self, deck_card_names: List[str]) -> None:
        """
        Initializes the Player with a deck defined by a list of card names.
        """
        if len(deck_card_names) != 8:
            raise ValueError("A deck must contain exactly 8 cards.")

        self.starting_deck: List[Card] = [get_card_by_name(name) for name in deck_card_names]
        self.elixir: float = 0.0

        self.deck: List[Card] = []
        self.hand: List[Card] = []
        self.next_card: Card | None = None

    def reset(self, elixir: int = 5) -> None:
        """
        Resets the player's state for a new game.
        Shuffles the deck, draws a new hand, and sets starting elixir.
        """
        self.elixir = float(elixir)

        # Create a shuffled copy of the starting deck
        self.deck = self.starting_deck.copy()
        random.shuffle(self.deck)

        # Draw initial hand and next card
        self.hand = self.deck[:4]
        self.next_card = self.deck[4]
        self.deck = self.deck[5:] # The rest of the cards are the draw pile

    def get_pseudo_legal_cards(self) -> List[int]:
        """
        Returns a list of indices for cards in hand that are affordable
        with the current amount of elixir.
        """
        legal_cards: List[int] = []
        for i, card in enumerate(self.hand):
            if card.elixir_cost <= self.elixir:
                legal_cards.append(i)
        return legal_cards

    def step(self, elixir_rate: float) -> None:
        """
        Updates the player's elixir, ensuring it doesn't exceed the max of 10.
        """
        self.elixir = min(10.0, self.elixir + elixir_rate)

    def _draw_card(self, played_card: Card) -> None:
        """
        Handles the card cycle mechanism.
        The played card is added to the back of the draw pile.
        The 'next_card' is moved to the hand.
        A new 'next_card' is drawn from the pile.
        """
        # Find the played card in hand and replace it with the next card
        for i in range(len(self.hand)):
            if self.hand[i] is played_card:
                self.hand[i] = self.next_card
                break

        # Add the played card to the back of the deck
        self.deck.append(played_card)

        # Draw a new next card
        self.next_card = self.deck.pop(0)

    def play_card(self, card_index: int) -> Card:
        """
        Plays a card from the hand.
        Reduces elixir, cycles the card, and returns the played card.
        """
        if not 0 <= card_index < len(self.hand):
            raise IndexError("Card index out of bounds.")

        played_card = self.hand[card_index]

        if played_card.elixir_cost > self.elixir:
            raise ValueError("Not enough elixir to play this card.")

        self.elixir -= played_card.elixir_cost
        self._draw_card(played_card)

        return played_card