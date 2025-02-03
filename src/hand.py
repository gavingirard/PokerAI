import numpy as np

from typing import List, Tuple, Optional

class Suit:
    HEARTS   = 0
    SPADES   = 1
    DIAMONDS = 2
    CLUBS    = 3

    strings = {
        HEARTS:   ("Hearts",   "h", "♥"),
        SPADES:   ("Spades",   "s", "♠"),
        DIAMONDS: ("Diamonds", "d", "♦"),
        CLUBS:    ("Clubs",    "c", "♣")
    }

class Rank:
    TWO   = 0
    THREE = 1
    FOUR  = 2
    FIVE  = 3
    SIX   = 4
    SEVEN = 5
    EIGHT = 6
    NINE  = 7
    TEN   = 8
    JACK  = 9
    QUEEN = 10
    KING  = 11
    ACE   = 12

    strings = {
        TWO:   ("Two",   "2"),
        THREE: ("Three", "3"),
        FOUR:  ("Four",  "4"),
        FIVE:  ("Five",  "5"),
        SIX:   ("Six",   "6"),
        SEVEN: ("Seven", "7"),
        EIGHT: ("Eight", "8"),
        NINE:  ("Nine",  "9"),
        TEN:   ("Ten",   "T"),
        JACK:  ("Jack",  "J"),
        QUEEN: ("Queen", "Q"),
        KING:  ("King",  "K"),
        ACE:   ("Ace",   "A")
    }

class Hand:
    HIGH     = 0
    PAIR     = 1
    TWO_PAIR = 2
    TRIPS    = 3
    STRAIGHT = 4
    FLUSH    = 5
    BOAT     = 6
    QUADS    = 7
    SFLUSH   = 8
    RFLUSH   = 9

    # Hand name, Fmt string ($ = hand, % = rank(0))
    strings = {
        HIGH:     ("High Card",       "% High"),
        PAIR:     ("Pair",            "$, %s"),
        TWO_PAIR: ("Two Pair",        None),
        TRIPS:    ("Three of a Kind", "Three of a Kind, %s"),
        STRAIGHT: ("Straight",        "$, % High"),
        FLUSH:    ("Flush",           "$, % High"),
        BOAT:     ("Full House",      None),
        QUADS:    ("Four of a Kind",  "$, %s"),
        SFLUSH:   ("Straight Flush",  "$, % High"),
        RFLUSH:   ("Royal Flush"      "$")
    }

# Pretty format the name of a hand given the deciders
def get_hand_name(hand: Hand, deciders: np.ndarray) -> str:
    def rank(x: int) -> str:
        s = Rank.strings[deciders[x]][0]
        return s if s == "Six" else s + "e"
    rank = lambda x: Rank.strings[deciders[x]][0]
    if hand == Hand.TWO_PAIR:
        return f"Two Pair, {rank(0)}s and {rank(2)}s"
    elif hand == Hand.BOAT:
        return f"Full House, {rank(0)}s Full of {rank(3)}s"
    else:
        hand_name = Hand.strings[hand][0]
        hand_fmt = Hand.strings[hand][1]
        return hand_fmt.replace("$", hand_name).replace("%", rank(0))
    
# Turn a card ID into its suit and rank
def parse_card(card: int) -> Tuple[int, int]:
    return divmod(card, 13)
        
# Format a card into a string like "Ah" or "3s"
def card_fmt(card: int) -> str:
    suit, rank = parse_card(card)
    return f"{Rank.strings[rank][1]}{Suit.strings[suit][2]}"

class HandEvaluation:
    """
    Class to evaluate the winner of a hand of poker at showdown.
    """
    def __init__(self, flop_cards: List[int], turn_card: int, river_card: int, 
                 blind_cards: List[int], button_cards: List[int]):
        """
        Initialize the HandEvaluation object with the cards on the board and in the hands of
        the players.
        
        Args:
            flop_cards (List[int]): The three cards on the flop.
            turn_card (int): The turn card.
            river_card (int): The river card.
            blind_cards (List[int]): The cards in the hand of the blind.
            button_cards (List[int]): The cards in the hand of the button.    
        """
        self.flop_cards = flop_cards
        self.turn_card = turn_card
        self.river_card = river_card
        self.blind_cards = blind_cards
        self.button_cards = button_cards

        # Save data for formatting if desired
        self.winning_hand     = None
        self.winning_deciders = None

    def blind_won(self) -> Optional[bool]:
        """
        Decide if the blind won the hand.

        Returns:
            bool: True if the blind won the hand, False otherwise, and None if the hands are
                  identical.
        """
        blind_rank, blind_deciders = self._get_best_hand(self.blind_cards)
        button_rank, button_deciders = self._get_best_hand(self.button_cards)
        
        # Update the winning hand and deciders for formatting
        blind_winner = self._blind_won(blind_rank, blind_deciders, button_rank, button_deciders)
        if blind_winner == True:
            self.winning_hand     = blind_rank
            self.winning_deciders = blind_deciders
        elif blind_winner == False:
            self.winning_hand     = button_rank
            self.winning_deciders = button_deciders
        return blind_winner

    # Wrapper function to decide who won the hand given the best hands of each player
    def _blind_won(self, blind_rank: Hand, blind_deciders: np.ndarray, button_rank: Hand,
                   button_deciders: np.ndarray) -> Optional[bool]:

        # Special tiebreakers for ace to five straights/straight flushes
        if blind_rank == button_rank and (blind_rank == Hand.STRAIGHT or blind_rank == Hand.SFLUSH):

            # Check if the blind has an ace to five straight
            if blind_deciders[0] == Rank.FIVE:
                # If the button has an ace to five straight too, it's a chop
                if button_deciders[0] == Rank.FIVE:
                    return None
                else:
                    return False
                
            # Check if the button has an ace to five straight
            if button_deciders[0] == Rank.FIVE:
                # If the blind has an ace to five straight too, it's a chop
                if blind_deciders[0] == Rank.FIVE:
                    return None
                else:
                    return True

        # Normal win logic for other hands which don't have special rules like straights
        if blind_rank > button_rank:
            return True
        elif blind_rank < button_rank:
            return False
        else:
            for i in range(len(blind_deciders)):
                # Try and decide based on the cards in the hand for identical ranks to see who has
                # the better hand. If it's something like where the board plays, then it's a chop
                if blind_deciders[i] > button_deciders[i]:
                    return True
                elif blind_deciders[i] < button_deciders[i]:
                    return False
            # Return None if the hands are identical
            return None
        
    # Get the best hand that can be made with the community cards and the given hand. Returns in
    # the form of hand rank, list of decider card ranks ordered highest to lowest
    def _get_best_hand(self, hole_cards: List[int]) -> Tuple[Hand, np.ndarray]:

        # Full list of card IDs and a bit of preprocessing to speed things up
        cards = np.array(self.flop_cards + [self.turn_card, self.river_card] + hole_cards)
        suits = cards // 13
        ranks = cards % 13
        
        # Easy way to check the count of ranks for finding X-of-a-kind
        rank_counts = np.array([np.sum(ranks == r) for r in range(13)])
        quads = np.where(rank_counts == 4)[0]
        trips = sorted(np.where(rank_counts == 3)[0], reverse=True)
        pairs = sorted(np.where(rank_counts == 2)[0], reverse=True)

        # If there is a flush, check for straight flushes and royal flushes
        flush_cards = None
        for s in range(4):
            if np.sum(suits == s) >= 5:
                flush_cards = np.array(sorted(cards[suits == s], reverse=True))

        if flush_cards is not None:
            # Look for a straight inside of the flush cards
            flush_ranks = flush_cards % 13
            sf_cards = self.find_straight(flush_ranks)

            if sf_cards is not None:
                # If there is a straight and the highest card is an ace with the lowest card as a 
                # ten, it's a royal flush but this code probably won't run more than a couple times
                if np.max(flush_ranks) == Rank.ACE and np.min(flush_ranks[:5]) == Rank.TEN:
                    return Hand.RFLUSH, flush_cards[:5]
                else:
                    return Hand.SFLUSH, flush_cards[:5]
                
            else:
                # I can return this without checking for quads or a boat since the hands are 
                # mutually exclusive, so there is a lot that you can't have with a flush
                return Hand.FLUSH, flush_cards[:5]

        # Look for quads
        if len(quads) > 1:
            # Find the highest count rank and then the next highest kicker card
            r = quads[0]
            kicker = sorted(ranks[ranks != r], reverse=True)[0]
            return Hand.QUADS, np.array([r, r, r, r, kicker])
        
        # Look for full houses (ignore 3 + 4 since quads are handled earlier)
        if 3 in rank_counts and (2 in rank_counts or len(trips) > 1):

            # If there are double trips
            if len(trips) > 1:
                # Get the ranks of the trips and then return the highest possible full house
                h, l = trips[0], trips[1]
                return Hand.BOAT, np.array([h, h, h, l, l])
            
            # There is only one triple in the hand
            t = trips[0]
            # If there are double pairs
            if len(pairs) > 1:
                # Get the doubles and return the highest possible full house
                top = np.max(pairs)
                return Hand.BOAT, np.array([t, t, t, top, top])
            
            # Standard full house
            d = pairs[0]
            return Hand.BOAT, np.array([t, t, t, d, d])
                
        # Look for straights
        straight_cards = self.find_straight(ranks)
        if straight_cards is not None:
            return Hand.STRAIGHT, straight_cards
        
        # Look for three of a kind
        if len(trips) > 0:
            # Get the highest triple and then the next two highest kicker cards
            r = trips[0]
            kickers = sorted(ranks[ranks != r], reverse=True)[:2]
            return Hand.TRIPS, np.concatenate((np.array([r, r, r]), kickers))
        
        # Look for two pair
        if len(pairs) > 1:
            # Get the two highest pairs and then a kicker card
            top, bot = pairs[:2]
            kicker = sorted(ranks[(ranks != top) & (ranks != bot)], reverse=True)[0]
            return Hand.TWO_PAIR, np.array([top, top, bot, bot, kicker])
        
        # Look for a pair
        if len(pairs) > 0:
            # Get highest pair and then three kickers
            top = pairs[0]
            kickers = sorted(ranks[ranks != top], reverse=True)[:3]
            return Hand.PAIR, np.concatenate((np.array([top, top]), kickers))
        
        # If no special hand was made return the top five cards in the hand
        return Hand.HIGH, sorted(ranks, reverse=True)[:5]
    
    # Find a straight if one exists in the given cards
    def find_straight(self, ranks: np.ndarray) -> Optional[np.ndarray]:
        
        # Remove duplicates and sort
        sorted_unique = list(reversed(np.unique(ranks)))
        for start in range(len(sorted_unique) - 4):
            
            # If the difference between the first and last card is four then there is a straight
            if sorted_unique[start] - sorted_unique[start + 4] == 4:
                # Return the ranks of the straight
                return sorted_unique[start:start + 5]
            
        # Check for Ace to 5 straight since it's a special case. Does this by seeing if for every
        # item in A, 2, 3, 4, 5 there is that item in the unique rank array
        ace_to_five = np.array([Rank.FIVE, Rank.FOUR, Rank.THREE, Rank.TWO, Rank.ACE])
        if np.all(np.isin(ace_to_five, sorted_unique)):
            return ace_to_five
