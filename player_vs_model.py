# Import game files from src directory
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from hand import *
from environment import *
from train import *

clear = lambda: os.system("cls" if os.name == "nt" else "clear")

model = DeepQNetwork("model.pth").to(device)

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
    input(" [Player] - Press ENTER to view your cards.")
    clear()
    input(f" [Player] - Your hand is [ {card_fmt(p1_cards[0])} {card_fmt(p1_cards[1])} ]. Press ENTER to continue.")
    clear()

    # Post the blind
    if env.blind_player == Player.Player1:
        print(f" [Player] -> Posts the blind of {1} chip(s).")
    else:
        print(f" [AI] -> Posts the blind of {1} chip(s).")

    # Start of gameplay
    done = False
    winner = None
    won = None
    last_round = None
    while not done:

        # Round header
        if env.current_round != last_round:
            print(f" [Game] -> Starting Round: {round_names[env.current_round]}")
            last_round = env.current_round

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
            print(f" [Player] -> [ {' '.join([card_fmt(c) for c in p1_cards])} ]")
            print(f" [AI] -> [ {' '.join([card_fmt(c) for c in p2_cards])} ]")
        
        # Print stacks
        if env.blind_player == Player.Player1:
            print(f" [Player] -> Stack: {env.blind_state.stack}")
            print(f" [AI] -> Stack: {env.button_state.stack} (D)")
        else:
            print(f" [Player] -> Stack: {env.button_state.stack} (D)")
            print(f" [AI] -> Stack: {env.blind_state.stack}")

        if env.current_round == BettingRound.SHOWDOWN:
            _, done, _, winner, won = env.step(ActionType.CHECK)
            continue

        if env.current_player == Player.Player1:
            # Request action from the current player
            action = input(f" [Player] - Choose what to do: [F]old, [Ch]eck, [C]all, [R]aise > ").lower()

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
        
        else:
            # Request optimal action from the model
            s_vector = torch.tensor(get_state_vector(env), dtype=torch.float32).to(device)
            q_values = model.forward(s_vector).cpu().detach().numpy().argsort()[::-1]

            for action_id in q_values:
                action = QVActionType.options[action_id]
                a_type, a_amt = qv_action_to_action(action, env)
                if a_type == ActionType.RAISE and a_amt == 0:
                    a_type = ActionType.CHECK

                _, temp_done, illegal, winner, won = env.step(a_type, a_amt)
                if not illegal:
                    # The model still makes mistakes so check if it's legal just in case
                    done = temp_done
                    if a_type == ActionType.FOLD:
                        print( " [AI] - Folds.")
                    elif a_type == ActionType.CHECK:
                        print(" [AI] - Checks.")
                    elif a_type == ActionType.CALL:
                        print(" [AI] - Calls.")
                    elif a_type == ActionType.RAISE:
                        print(f" [AI] - Raises by {a_amt} chip(s).")
                    break

    # End of game
    if winner is None:
        print(" [Game] Game ends in a chop pot.")
    elif env.hand_eval is None:
        print(f" [Game] {'Player' if winner == 1 else 'AI'} mucks and wins a pot of {won} chip(s).")
    else:
        name = get_hand_name(env.hand_eval.winning_hand, env.hand_eval.winning_deciders)
        print(f" [Game] {'Player' if winner == 1 else 'AI'} wins a pot of {won} chip(s) with {name}.")
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