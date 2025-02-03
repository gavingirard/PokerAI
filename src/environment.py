import numpy as np

from typing import Tuple, Optional

from hand import HandEvaluation

class BettingRound:
    PREFLOP  = 0
    FLOP     = 1
    TURN     = 2
    RIVER    = 3
    SHOWDOWN = 4

round_names = {
    BettingRound.PREFLOP:  "Pre-Flop",
    BettingRound.FLOP:     "Flop",
    BettingRound.TURN:     "Turn",
    BettingRound.RIVER:    "River",
    BettingRound.SHOWDOWN: "Showdown"
}

class ActionType:
    FOLD  = 0
    CHECK = 1
    CALL  = 2
    RAISE = 3

    options = [FOLD, CHECK, CALL, RAISE]

action_names = {
    ActionType.FOLD:  "Fold",
    ActionType.CHECK: "Check",
    ActionType.CALL:  "Call",
    ActionType.RAISE: "Raise"
}

class Player:
    Player1 = 1
    Player2 = 2

# Get the opposite player from the one given
other_player = lambda p: Player.Player2 if p == Player.Player1 else Player.Player1

# Data holder class to store information about the state of one of the players
class PlayerState:
    def __init__(self):
        self.player        = None
        self.stack         = None
        self.orig_stack    = None
        self.total_bet     = 0
        self.hand          = None
        self.round_history = [None for _ in range(4)]
        self.raise_history = [None for _ in range(4)]

    # Reset data for one betting round
    def new_round(self, other_stack: int):
        self.round_total_bet = 0
        self.round_to_call   = 0
        self.round_min_raise = 0
        self.round_max_bet   = min(self.stack, other_stack)

    # Update the the player's betting history if relevant. History is only updated if the currently
    # saved action is not a raise-- if a player raises and then ends up calling a re-raise we still
    # want to record that they original raised and took aggressive action
    def update_history(self, action: ActionType, rnd: BettingRound, amt: int):
        if self.round_history[rnd] != ActionType.RAISE:
            self.round_history[rnd] = action

class HeadsUpPoker:
    """
    Environment for a heads up poker game. Works similarly to a gym environment but with a few
    modifications (like more return values from step) that make it better tailored to this project.
    """

    def __init__(self, blind_amount: int = 1, fold_penalty: int = 0, being_played: bool = False, 
                 verbose: bool = False):
        """
        Create a new heads up poker environment with the given blind size and the given
        hyperparameters which all default to 0.

        Args:
            blind_amount (Optional[int]): The size of the small blind in the game.
            fold_penalty (Optional[int]): The penalty for folding a hand when a check was legal.
            being_played (Optional[bool]): If true, the environment will act like a more realistic
                poker game where stacks are persistant between hands, blinds rotate, and illegal
                actions do not immediately end the game.
            verbose (Optional[bool]): If true, the environment will print out debug information 
                about the internal logic of the game as it runs which is helpful for fixing stuff.
        """

        # Game parameters
        self.blind_amount    = blind_amount
        self.fold_penalty    = fold_penalty
        self.being_played    = being_played
        self.verbose         = verbose

        # Starting players
        self.blind_player  = Player.Player2
        self.button_player = Player.Player1

        # Persistent stacks for gameplay
        self.p1_stack_persistent = None
        self.p2_stack_persistent = None

    def reset(self, p1_stack: int = None, p2_stack: int = None):
        """
        Reset the environment to the start of a new hand. Rotates the blinds if the property
        `being_played` is set to true. If the stacks are not given and the environment is being
        used as a game, the stacks will persist between hands, but will throw an error if the
        stack sizes weren't initialized once at the start of the game (first use of `reset()`).

        Args:
            p1_stack (Optional[int]): The starting stack size for player 1. Can be left blank if
                the environment is being played as a game since stacks persist.
            p2_stack (Optional[int]): The starting stack size for player 2. Can be left blank if
                the environment is being played as a game since stacks persist.
        """

        if self.being_played:

            # Rotate blind
            self.blind_player, self.button_player = self.button_player, self.blind_player
            self.log(f"Rotating blinds. Blind player is now Player {self.blind_player}.")

            # Get stacks from persistent variable if not given
            if p1_stack is None or p2_stack is None:
                self.log("Using previous stack values for the new hand.")
                if self.p1_stack_persistent is None or self.p2_stack_persistent is None:
                    raise ValueError("Stack sizes must be initialized before the first hand.")
                p1_stack = self.p1_stack_persistent
                p2_stack = self.p2_stack_persistent

            else:
                self.log("Using new stack values for the new hand.")
                self.p1_stack_persistent = p1_stack
                self.p2_stack_persistent = p2_stack

        # Initialize game state for the new hand
        self.current_player = self.button_player
        self.current_round  = BettingRound.PREFLOP
        self.pot            = 0
        self.hand_eval      = None

        cards = list(np.random.choice(52, 9, replace=False))
        self.flop_cards = cards[4:7]
        self.turn_card  = cards[7]
        self.river_card = cards[8]

        # Blind player data
        self.blind_state            = PlayerState()
        self.blind_state.player     = self.blind_player
        self.blind_state.stack      = p1_stack if self.blind_player == Player.Player1 else p2_stack
        self.blind_state.orig_stack = self.blind_state.stack
        self.blind_state.total_bet  = self.blind_amount
        self.blind_state.hand       = [cards[0], cards[2]]

        # Make the blind position post the blind
        if self.blind_state.stack < self.blind_amount:
            self.log("Blind's stack is less than the blind amount.")
            raise RuntimeError("Blind player was not able to post blind.")
        else:
            self.pot += self.blind_amount
            self.blind_state.stack -= self.blind_amount
            self.log(f"Blind player posted a blind of {self.blind_amount}, now has " \
                     f"{self.blind_state.stack}.")

        # Button player data
        self.button_state            = PlayerState()
        self.button_state.player     = self.button_player
        self.button_state.stack      = p1_stack if self.button_player == Player.Player1 else p2_stack
        self.button_state.orig_stack = self.button_state.stack
        self.button_state.total_bet  = 0
        self.button_state.hand       = [cards[1], cards[3]]

        # Update starting max bets and other per-round data points
        self.blind_state.new_round(self.button_state.stack)
        self.button_state.new_round(self.blind_state.stack)

        # Update starting per-round data for pre-flop
        self.blind_state.round_total_bet = self.blind_amount
        self.button_state.round_total_bet = 0
        self.button_state.round_to_call = self.blind_amount

    def step(self, action: ActionType, raise_amount: int = None) \
        -> Tuple[float, bool, bool, Optional[Player], Optional[int]]:
        """
        Take an action in the environment.

        Args:
            action (ActionType): The type of action to take.
            raise_amount (int): The amount to raise by, if applicable.

        Returns:
            Tuple: A tuple containing the reward from the action, if the game is over, if the 
                action was illegal, the winner if applicable, and the amount won by the winner
                if applicable.
        """

        # Shortcut to showdown at the end of a game if given a none action
        if self.current_round == BettingRound.SHOWDOWN:
            self.log("Showdown triggered at the end of a game.")
            return self.showdown()
            
        # Assign who the current and opponent player are for this round
        current = self.get_current_player_state()
        opponent = self.get_opponent_player_state()

        # Record the action in history
        current.update_history(action, self.current_round, raise_amount)

        # Fold, call, and raise actions are the same for all positions but check has some extra
        # logic to it so it is separated by a button function and a blind function
        if action == ActionType.FOLD:
            self.log(f"Handling a fold by Player {current.player}.")
            return self._action_fold(current, opponent)
        
        elif action == ActionType.CALL:
            self.log(f"Handling a call by Player {current.player}.")
            return self._action_call(current, opponent)
        
        elif action == ActionType.RAISE:
            self.log(f"Handling a raise by Player {current.player} by {raise_amount} chip(s).")
            return self._action_raise(current, opponent, raise_amount)
        
        elif action == ActionType.CHECK:
            if current.player == self.button_player:
                self.log(f"Handling a check by Player {current.player} (button).")
                return self._action_btn_check(current, opponent)
            else:
                self.log(f"Handling a check by Player {current.player} (blind).")
                return self._action_bld_check(current, opponent)
        else:
            raise ValueError("Invalid action type given.")
            
    # Handle a fold action
    def _action_fold(self, current: PlayerState, opponent: PlayerState):
        # Reward for a fold is the negative of the contribution so far in the hand
        reward = -current.total_bet

        if current.round_to_call == 0:
            # If the amount needed to call is zero there is a penalty for folding unnecessarily
            # on top of losing the pot
            self.log(f"Player {current.player} folded when they could have checked.")
            reward -= self.fold_penalty * self.blind_amount

        # Return that the game is over, the action was legal, and that the opponent won the
        # current player's total contribution to the pot
        self.log(f"Player {current.player} folded. Reward: {reward}. (Game Over, Legal)")
        return reward, True, False, opponent.player, current.total_bet
    
    # Handle a check action as the blind player
    def _action_bld_check(self, blind: PlayerState, opponent: PlayerState):
        if self.current_round == BettingRound.PREFLOP:
            # If the blind checks pre-flop check if there was a raise from the button
            # before moving to the next betting round
            if blind.round_to_call == 0:
                self.log("Blind checked pre-flop with no action from the button.")
                self._next_betting_round()
            else:
                self.log("Blind illegally checked pre-flop after a raise from the button.")
                return self._illegal_action_return(blind, opponent)
        else:
            if blind.round_to_call == 0:
                self.log("Blind checked post-flop, going to the button to check/raise.")
                self._next_player()
            else:
                self.log("Blind illegally checked post-flop after a raise from the button.")
                return self._illegal_action_return(blind, opponent)
            
        # Generic return
        return self._generic_step_return()

    # Handle a check action as the button player
    def _action_btn_check(self, button: PlayerState, opponent: PlayerState):
        if self.current_round == BettingRound.PREFLOP:
            # Betting before the flop as the button is not possible
            self.log("Button illegally checked pre-flop.")
            return self._illegal_action_return(button, opponent)
        else:
            # If the button checks post-flop check if there was a raise from the blind before
            # moving to the next betting round
            if button.round_to_call == 0:
                self.log("Button checked post-flop with no action from the blind.")
                self._next_betting_round()
            else:
                # The button checked even though the blind raised which is illegal
                self.log("Button illegally checked post-flop after a raise from the blind.")
                return self._illegal_action_return(button, opponent)
            
        # Generic return
        return self._generic_step_return()
        
    # Handle a call action
    def _action_call(self, current: PlayerState, opponent: PlayerState):
        # Calling instead of checking is illegal
        if current.round_to_call == 0:
            self.log(f"Player {current.player} called when they should have checked.")
            return self._illegal_action_return(current, opponent)
        
        # Make the call and change player stacks
        current.stack -= current.round_to_call
        current.total_bet += current.round_to_call
        current.round_total_bet += current.round_to_call
        current.round_to_call = 0
        self.pot += current.round_to_call
        
        # Change max bets
        max_bet = min(current.stack, opponent.stack)
        current.round_max_bet = max_bet
        opponent.round_max_bet = max_bet

        # The only time you move to the next player is if the button is calling pre-flop
        if current.player == self.button_player and self.current_round == BettingRound.PREFLOP and \
                current.total_bet == self.blind_amount:
            self.log("Button called pre-flop, now moving to the blind to check/raise.")
            self._next_player()
        else:
            self.log(f"Player {current.player} called. New stack: {current.stack}. Moving to next " \
                     "betting round.")
            self._next_betting_round()

        # Generic return
        return self._generic_step_return()

    # Handle a raise action as either player, where raise_amount is the amount to raise by on top
    # of making a call if one is needed
    def _action_raise(self, current: PlayerState, opponent: PlayerState, raise_amount: int):
        bet_amount = current.round_to_call + raise_amount

        # If the player is all in then they can't raise
        if current.stack == 0:
            self.log(f"Player {current.player} attempted to raise when they were all in.")
            return self._illegal_action_return(current, opponent)
        
        # If the raise amount is too small, check if it's an all in or if it's just illegal
        if raise_amount < current.round_min_raise:
            if bet_amount == current.round_max_bet:
                self.log(f"Player {current.player} went all in or pulled opponent all in.")
            else:
                self.log(f"Raise amount of {raise_amount} was less than the minimum raise amount.")
                return self._illegal_action_return(current, opponent)

        # If the raise amount is too large then it is illegal
        if bet_amount > current.round_max_bet:
            self.log(f"Raise amount of {bet_amount} total was greater than the maximum bet amount.")
            return self._illegal_action_return(current, opponent)

        # If everything is allowed, update the data to perform the raise
        current.total_bet += bet_amount
        current.round_total_bet += bet_amount
        current.stack -= bet_amount
        current.round_to_call = 0
        opponent.round_to_call = raise_amount
        opponent.round_min_raise = raise_amount

        # Update max bets
        max_bet = min(current.stack, opponent.stack)
        current.round_max_bet = max_bet
        opponent.round_max_bet = max_bet

        # Move to the next player to respond to the raise and generic return
        self._next_player()
        self.log(f"Player {current.player} raised by {raise_amount}. New stack: {current.stack}.")
        return self._generic_step_return()

    # Get the return for a step with an illegal action    
    def _illegal_action_return(self, current: PlayerState, opponent: PlayerState):
        reward = -current.stack - current.total_bet
        if self.being_played:
            return reward, False, True, None, None
        else:
            return reward, True, True, opponent.player, current.total_bet
        
    # Return a standard response to a step with no noteworthy events
    def _generic_step_return(self):
        return 0, False, False, None, None
    
    # Perform the showdown at the end of the game
    def showdown(self):
        # Evaluate the hand to see who wins if it goes to showdown (does not actually perform the 
        # evaluation since that is super data intensive)
        self.log("Creating new hand evaluation object to determine the winner.")
        self.hand_eval = HandEvaluation(self.flop_cards, self.turn_card, self.river_card, 
                                        self.blind_state.hand, self.button_state.hand)
        # Get the amount of chips in the pot and who the winner is, then return the result data
        self.log("Evaluating the hands to determine the winner.")
        blind_won = self.hand_eval.blind_won()
        # Check if there was a chop pot
        if blind_won is None:
            self.log("Round ended in a chop pot.")
            return 0, True, False, None, 0
        else:
            if not self.blind_state.total_bet == self.button_state.total_bet:
                raise RuntimeError("Players have not completed betting.")
            # Deal out chips to people's stacks
            amount = self.blind_state.total_bet
            total_chips_won = self.blind_state.total_bet + self.button_state.total_bet
            winner = self.blind_player if blind_won else self.button_player
            self.log(f"Player {winner} won the showdown and {total_chips_won} chips.")

            # Update persistent stacks if the environment is being played
            if self.being_played:
                self.log("Updating persistent stacks with the results of the showdown.")
                if winner == Player.Player1:
                    self.p1_stack_persistent += amount
                    self.p2_stack_persistent -= amount
                else:
                    self.p2_stack_persistent += amount
                    self.p1_stack_persistent -= amount

            return 0, True, False, winner, total_chips_won
        
    # Move to the next player at the table
    def _next_player(self):
        self.current_player = other_player(self.current_player)

    # Move to the next betting round from the current one, if possible
    def _next_betting_round(self):

        # Send the hand to showdown if the round is the river
        if self.current_round == BettingRound.RIVER:
            self.log("Moving to showdown after the river.")
            self.current_round = BettingRound.SHOWDOWN
            return
        
        # Fail if bets are not even between the two players
        if self.blind_state.round_total_bet != self.button_state.round_total_bet:
            raise RuntimeError("Players have not completed the current betting round.")
        
        # Update values to the start of a new betting round
        self.current_round += 1
        self.current_player = self.blind_player
        self.blind_state.new_round(self.button_state.stack)
        self.button_state.new_round(self.blind_state.stack)

    # Get the player state for the current player
    def get_current_player_state(self) -> PlayerState:
        return self.blind_state if self.current_player == self.blind_player else self.button_state
    
    # Get the player state for the opponent player
    def get_opponent_player_state(self) -> PlayerState:
        return self.button_state if self.current_player == self.blind_player else self.blind_state

    # Print a debug message if verbose is set to True
    def log(self, msg: str):
        if self.verbose:
            print(" [DEBUG] " + msg)
