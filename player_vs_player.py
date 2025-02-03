# Import game files from src directory
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from hand import *
from environment import *

clear = lambda: os.system("cls" if os.name == "nt" else "clear")

def game(env: HeadsUpPoker):
    env.reset()
    clear()

    if env.blind_player == Player.Player1:
        p1_cards = env.blind_state.hand
        p2_cards = env.button_state.hand
    else:
        p1_cards = env.button_state.hand
        p2_cards = env.blind_state.hand

    # Show Player 1 their cards:
    input(" [Player 1] - Press ENTER to view your cards.")
    clear()
    input(f" [Player 1] - Your hand is [ {card_fmt(p1_cards[0])} {card_fmt(p1_cards[1])} ]. Press ENTER to continue.")
    clear()

    # Show Player 2 their cards:
    input(" [Player 2] - Press ENTER to view your cards.")
    clear()
    input(f" [Player 2] - Your hand is [ {card_fmt(p2_cards[0])} {card_fmt(p2_cards[1])} ]. Press ENTER to continue.")
    clear()

    # Post the blind
    if env.blind_player == Player.Player1:
        print(f" [Player 1] -> Posts the blind of {1} chip(s).")
    else:
        print(f" [Player 2] -> Posts the blind of {1} chip(s).")

    # Start of gameplay
    done = False
    winner = None
    won = None
    while not done:

        # Show the current board
        if env.current_round == BettingRound.PREFLOP:
            pass
        elif env.current_round == BettingRound.FLOP:
            print(f" [Board] -> [ {' '.join([card_fmt(c) for c in env.flop_cards])} ]")
        elif env.current_round == BettingRound.TURN:
            print(f" [Board] -> [ {' '.join([card_fmt(c) for c in env.flop_cards])} {card_fmt(env.turn_card)} ]")
        else:
            print(f" [Board] -> [ {' '.join([card_fmt(c) for c in env.flop_cards])} {card_fmt(env.turn_card)} {card_fmt(env.river_card)} ]")
        if env.current_round == BettingRound.SHOWDOWN:
            print(f" [Player 1] -> [ {' '.join([card_fmt(c) for c in p1_cards])} ]")
            print(f" [Player 2] -> [ {' '.join([card_fmt(c) for c in p2_cards])} ]")
        
        # Print stacks
        if env.blind_player == Player.Player1:
            print(f" [Player 1] -> Stack: {env.blind_state.stack}")
            print(f" [Player 2] -> Stack: {env.button_state.stack} (D)")
        else:
            print(f" [Player 1] -> Stack: {env.button_state.stack} (D)")
            print(f" [Player 2] -> Stack: {env.blind_state.stack}")

        if env.current_round == BettingRound.SHOWDOWN:
            _, done, _, winner, won = env.step(ActionType.CHECK)
            continue

        # Request action from the current player
        action = input(f" [Player {env.current_player}] - Choose what to do: [F]old, [Ch]eck, [C]all, [R]aise > ").lower()

        illegal = False
        if action == "f" or action == "fold":
            _, done, _, winner, won = env.step(ActionType.FOLD)
        elif action == "ch" or action == "check":
            _, done, illegal, _, _ = env.step(ActionType.CHECK)
        elif action == "c" or action == "call":
            _, done, illegal, _, _ = env.step(ActionType.CALL)
        elif action == "r" or action == "raise":
            amt = int(input(" [Game] Raise amount > "))
            _, done, illegal, _, _ = env.step(ActionType.RAISE, amt)

        if illegal:
            print(" [Game] Illegal action. Please try again.")

    # End of game
    if winner is None:
        print(" [Game] Game ends in a chop pot.")
    if env.hand_eval is None:
        print(f" [Game] Player {winner} mucks and wins a pot of {won} chip(s).")
    else:
        name = get_hand_name(env.hand_eval.winning_hand, env.hand_eval.winning_deciders)
        print(f" [Game] Player {winner} wins a pot of {won} chip(s) with {name}.")
    input(" [Game] Press ENTER to continue.")

if __name__ == "__main__":
    env = HeadsUpPoker(blind_amount=1, being_played=True)
    env.reset(50, 50)
    while True:
        try:
            game(env)
        except KeyboardInterrupt:
            print()
            break