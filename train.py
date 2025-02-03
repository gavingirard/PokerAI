# Import game files from src directory
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from hand import *
from environment import *

import torch, datetime, tqdm, time, collections, random, math
import numpy as np

START_EPISODE   = 0      # Training start episode, for resuming
SAVE_EVERY      = 1000   # Checkpoint to save the model and report progress
EPISODES        = 300000 # Number of hands to play
BLIND_SIZE      = 1      # Default for the entire training so everything is divisible
FOLD_PENALTY    = 1      # Bot loses X blinds if it folds unnecessarily)
RAISE_PRIORITY  = 0.2    # Percent of exploration actions which will be raises
LOSS_WEIGHT     = 0.95   # Multiply the chip loss by this to make it prioritize winning
LEGAL_PSREWARD  = 0.02   # Small psuedo-reward for making a legal move
SHOWDOWN_REWARD = 0.05   # Small psuedo-reward for making it to showdown
LEARNING_RATE   = 0.001  # Small learning rate to have it learn slowly
DISCOUNT_FACTOR = 0.99   # Prioritize long term rewards more
EPSILON_START   = 1.0    # Start at 100% exploration
EPSILON_MIN     = 0.2    # Minimum exploration rate so the model still explores and makes mistakes
EPSILON_DECAY   = 0.9996 # Decay rate for exploration
MEMORY_SIZE     = 50000  # Number of experiences to store in the experience replay buffer
BATCH_SIZE      = 128    # Number of experiences to train on at once
REPLAY_EVERY    = 128    # Number of actions to do between training on the replay buffer
UPDATE_TARGET   = 2000   # Number of episodes to train on before updating the target model
NN_INPUT_SIZE   = 206    # Size of the state vector
FIRST_LSIZE     = 128    # First layer size
SECOND_LSIZE    = 192    # Second layer size
THIRD_LSIZE     = 64     # Third layer size
DROPOUT_RATE    = 0.1    # Dropout rate for the model

def log(message: str):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")

# Get which device to run the training on
if torch.cuda.is_available():
    device = torch.device("cuda") # NVIDIA
    log("CUDA is available, running on GPU.")
elif torch.backends.mps.is_available():
    device = torch.device("mps") # Neural Engine
    log("MPS is available, running on MPS.")
else:
    device = torch.device("cpu")
    log("CUDA and MPS are not available, something went wrong.")
    exit() # Do NOT let me accidentally train this on CPU

# Take the current state of the environment and return the state vector to go into the model. This
# returns the game state specifically for the current player
def get_state_vector(env: HeadsUpPoker) -> np.ndarray:
    current = env.get_current_player_state()
    opponent = env.get_opponent_player_state()
    state = np.zeros(NN_INPUT_SIZE)

    # Cards are stored as a 17 long vector where the first 4 are the suit and the next 13 are rank,
    # ordered as player's hand first, then flop, then turn, then river for a total of 119 features
    def write_card(card_num: int, card: int):
        suit, rank = parse_card(card)
        state[card_num * 17 + suit] = 1
        state[card_num * 17 + 4 + rank] = 1

    # Write all of the cards to the state vector
    for i, card in enumerate(current.hand):
        write_card(i, card)
    # Write opponent's hand if it is showdown
    if env.current_round == BettingRound.SHOWDOWN:
        for i, card in enumerate(opponent.hand):
            write_card(i + 2, card)
    if env.current_round >= BettingRound.FLOP:
        for i, card in enumerate(env.flop_cards):
            write_card(i + 4, card)
    if env.current_round >= BettingRound.TURN:
        write_card(7, env.turn_card)
    if env.current_round == BettingRound.RIVER:
        write_card(8, env.river_card)

    # Player betting histories come next as 4 4-long vectors for each round and each action,
    # the first 4 being for the current player and the next 4 for the opponent for 32 features
    def write_round_history(player: PlayerState, o: int):
        for i, action in enumerate(player.round_history):
            if action is not None:
                state[o + (i * 4) + action] = 1
                    
    write_round_history(current, 153)
    write_round_history(opponent, 169)

    # Current betting round as a 5-long vector
    state[185 + env.current_round] = 1

    # Now features for stack sizes, pot sizes, to call amounts, etc. which are all scaled so that
    # 0 is 0 chips and 1 is the maximum original stack size between the two players
    max_stack = max(current.orig_stack, opponent.orig_stack)
    scale = lambda x: x / max_stack

    # Bit value for if the current player is the button
    if env.button_player == current.player:
        state[190] = 1

    # Blind size, scaled
    state[191] = scale(BLIND_SIZE)

    # Player data is saved as starting stack, current stack, total bet, round total bet, round to
    # call, round min raise, and round max bet
    def set_player_data(player: PlayerState, o: int):
        state[o]     = scale(player.orig_stack)
        state[o + 1] = scale(player.stack)
        state[o + 2] = scale(player.total_bet)
        state[o + 3] = scale(player.round_total_bet)
        state[o + 4] = scale(player.round_to_call)
        state[o + 5] = scale(player.round_min_raise)
        state[o + 6] = scale(player.round_max_bet)

    # Set player data and return
    set_player_data(current, 192)
    set_player_data(opponent, 199)
    return state

# Deep-Q neural network which gives the model an estimated Q value for every possible action given
# the current game state and any information about it that I provide on top
class DeepQNetwork(torch.nn.Module):
    def __init__(self, path: str = None):
        super().__init__()

        # Create a new model with the following architecture:
        # - Input layer is the size of the state vector and output is just the list of actions
        # - Each layer then has a LeakyReLU activation function which is just like ReLU but it
        #   lets a bit of negative values through for the "dying ReLU problem". I like LReLU
        #   here because all of my values are scaled between 0 and 1 which this function likes
        # - I also included a dropout layer after the final activation function since I read 
        #   that it can make the model not overfit by making it become more robust and not just
        #   rely on a couple nodes to make all of its decisions
        # - Softmax or anything like that is not used since this isn't classification but just
        #   predicting the expected reward for each action
        self.stack = torch.nn.Sequential(
            # First layer functions
            torch.nn.Linear(NN_INPUT_SIZE, FIRST_LSIZE),
            torch.nn.LeakyReLU(),
            # Second layer functions
            torch.nn.Linear(FIRST_LSIZE, SECOND_LSIZE),
            torch.nn.LeakyReLU(),
            # Third layer functions
            torch.nn.Linear(SECOND_LSIZE, THIRD_LSIZE),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(DROPOUT_RATE),
            # Output layer functions
            torch.nn.Linear(THIRD_LSIZE, len(QVActionType.options)))
        
        if path is not None:
            # Load model from disk onto whatever device is being used
            state = torch.load(path, map_location=device)
            self.load_state_dict(state)
        
    # Run the given state through the model. Returns the Q values of all four possible actions as
    # a tensor plus the recommended raise amount which should only be considered if raise is chosen
    def forward(self, state) -> torch.Tensor:
        return self.stack(state)
    
    # Save the model to disk at a checkpoint
    def save_checkpoint(self, path: str, name: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_dict(), f"{path}/{name}.pth")

# Separate from the environment ActionType, these are the action options for the DQN
class QVActionType:
    FOLD          = 0
    CHECK         = 1
    CALL          = 2
    RAISE_MIN     = 3
    # These QVActions were removed for simplicity but a next step for this model is adding support
    # for these actions so the model can be more nuanced than just min-raising
    # RAISE_2_MIN   = 4
    # RAISE_ALL_IN  = 5
    # RAISE_0_5_POT = 6
    # RAISE_POT     = 7
    # RAISE_1_5_POT = 8
    # RAISE_2_POT   = 9
    
    # Set of options that can be chosen by the DQN
    options = [FOLD, CHECK, CALL, RAISE_MIN]
    
# Translate a QV action to an actual action that the environment recognizes
def qv_action_to_action(qv_action: QVActionType, env: HeadsUpPoker) -> Tuple[ActionType, int]:
    curr = env.get_current_player_state()
    # Standard actions
    match qv_action:
        case QVActionType.FOLD:
            return ActionType.FOLD, 0
        case QVActionType.CHECK:
            return ActionType.CHECK, 0
        case QVActionType.CALL:
            return ActionType.CALL, 0
    # Different raise variations
    ra = 0
    match qv_action:
        case QVActionType.RAISE_MIN:
            if curr.round_min_raise == 0:
                ra = min(env.blind_amount, curr.round_max_bet)
            else:
                ra = min(curr.round_min_raise, curr.round_max_bet)
        #case QVActionType.RAISE_2_MIN:
        #    ra = curr.round_min_raise * 2
        #case QVActionType.RAISE_ALL_IN:
        #    ra = curr.round_max_bet
        #case QVActionType.RAISE_0_5_POT:
        #    ra = env.pot * 0.5
        #case QVActionType.RAISE_POT:
        #    ra = env.pot
        #case QVActionType.RAISE_1_5_POT:
        #    ra = env.pot * 1.5
        #case QVActionType.RAISE_2_POT:
        #    ra = env.pot * 2
    return ActionType.RAISE, math.floor(ra) # Floor so env gets an int

# Class which encapsulates some of the logic for training the model so that the training driver 
# code can be a lot simpler
class PokerTrainingAgent:
    def __init__(self, load_checkpoint: int = None):

        self.losses = []

        # Either set epsilon to the starting value or resume from checkpoint
        if load_checkpoint is None:
            self.epsilon = EPSILON_START
        else:
            self.epsilon = EPSILON_START * (EPSILON_DECAY ** load_checkpoint)
            if self.epsilon < EPSILON_MIN:
                self.epsilon = EPSILON_MIN

        # Create experience memory, model, and optimizer
        self.experience_memory = collections.deque(maxlen=MEMORY_SIZE)
        self.loss_function     = torch.nn.MSELoss()
        if load_checkpoint is None:
            self.model         = DeepQNetwork().to(device)
            self.target_model  = DeepQNetwork().to(device).eval()
        else:
            self.model         = DeepQNetwork(f"model/save/{load_checkpoint}.pth").to(device)
            self.target_model  = DeepQNetwork(f"model/save/{load_checkpoint}.pth").to(device).eval()
        self.optimizer         = torch.optim.Adam(self.model.parameters(), LEARNING_RATE)

        # Numpy docs said to use Generator since it is newer and performs better
        self.rand = np.random.default_rng()

    # Get the next action that should be taken by the model, either exploration or exploitation
    # depending on the current value of epsilon
    def get_next_action(self, state: np.ndarray, env: HeadsUpPoker) -> QVActionType:
        if self.rand.random() < self.epsilon:
            # Manually mess with probabilities here to make the model (almost) equally likely to
            # raise (of any size) as it is to check, call, or fold. I give raises a slight bit of 
            # priority as a hyperparameter since I am trying to create an aggressive poker bot
            if self.rand.random() < RAISE_PRIORITY:
                return QVActionType.RAISE_MIN
            else:
                if env.get_current_player_state().round_to_call == 0:
                    return self.rand.choice(QVActionType.options[:3])
                else:
                    return self.rand.choice([QVActionType.FOLD, QVActionType.CALL])
        else:
            # Best action for exploitation, done without keeping track of gradients so the model
            # just does inference and doesn't learn from this
            with torch.no_grad():
                q_values = self.target_model(self.tensor(state))
            action = torch.argmax(q_values).item()
            return action
            
    # Train the model on a batch of experiences from the experience replay buffer
    def replay_train(self) -> None:
        # Only train if there are enough experiences in the buffer
        if len(self.experience_memory) < BATCH_SIZE:
            return
        
        # Take a random sample from the experience replay buffer and train on it. I considered
        # using separate experience buffers for blind and button but since position is part of the
        # state it is fine to just have a single experience replay buffer for all positions
        minibatch = random.sample(self.experience_memory, BATCH_SIZE)
        total_batch_loss = 0

        # Go through every experience
        for experience in minibatch:

            # Note: qv_action_type is one of QVActionType, not ActionType; QVActionType gets
            # interpreted as an ActionType potentially with a raise amount but the model does not
            # output a continuous value for the raise amount
            start_state, qv_action_type, reward, next_state, terminal = experience

            if terminal:
                target_q_value = reward
            else:
                # Calculate the "correct" Q value for the action taken
                future_q_values = self.model(self.tensor(next_state))
                target_q_value = reward + (DISCOUNT_FACTOR * torch.max(future_q_values).item())

            # Replace the value and then calculate the loss
            current_q_values = self.model(self.tensor(start_state))
            modified_q_values = current_q_values.clone()
            modified_q_values[qv_action_type] = target_q_value
            total_batch_loss += self.loss_function(current_q_values, modified_q_values)

        # Reset gradients, backpropagate loss, and update model weights
        self.optimizer.zero_grad()
        avg_loss = total_batch_loss / BATCH_SIZE
        self.losses.append(avg_loss.item())
        avg_loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    # Update the target model to match the current training model
    def update_target(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())

    # Shortcut function to turn a numpy array into an on device tensor
    def tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, dtype=torch.float32).to(device)
    
    # Add an experience to the experience replay buffer
    def add_experience(self, start_state: np.ndarray, action_type: QVActionType, reward: float,
                       next_state: np.ndarray, terminal: bool) -> None:
        self.experience_memory.append((start_state, action_type, reward, next_state, terminal))

def train():
    log("Creating environment...")
    env = HeadsUpPoker(blind_amount=BLIND_SIZE, fold_penalty=FOLD_PENALTY)
    log("Creating DQN training agent...")
    if START_EPISODE > 0:
        agent = PokerTrainingAgent(START_EPISODE)
    else:
        agent = PokerTrainingAgent()
    log("Starting training...")
    progress_bar = tqdm.tqdm(range(START_EPISODE, EPISODES))

    start = time.time()
    illegal_count = 0
    action_count = 0
    showdown_count = 0

    action_counts = []
    showdowns = []
    illegal_counts = []

    for episode in progress_bar:

        if episode % UPDATE_TARGET == 0:
            agent.update_target()

        # Normally choose different amounts for the stacks so the model can learn to play from a 
        # variety of different levels of chips (should not use the same stack size for all games)
        # I use a mean of 120 and a stdv of 200 - 20 / 4 since I want 95% of my stack sizes to be
        # between 20 and 200- also clip values to make sure blind can be paid
        chip_values = np.random.normal(120, (200 - 20) / 4, 2)
        chip_values = np.clip(chip_values, BLIND_SIZE * 2, None)
        env.reset(*chip_values)

        def checkpoint():
            progress_bar.clear()
            log(f"Checkpoint: {str(episode).rjust(6)} / {EPISODES} => Illegal={str(illegal_count).ljust(3)} " \
                f"AvgCount={str(action_count / SAVE_EVERY).ljust(6)} Showdowns={str(showdown_count).ljust(3)}")
            # Save data to graph potentially
            action_counts.append(action_count / SAVE_EVERY)
            illegal_counts.append(illegal_count)
            showdowns.append(showdown_count)
            progress_bar.display()

        # Save progress and report on model stats for graphing
        if episode % SAVE_EVERY == 0 and episode > 0:
            progress_bar.clear()
            agent.model.save_checkpoint("model/save", f"checkpoint_{episode}")
            checkpoint()
            illegal_count = 0
            action_count = 0
            showdown_count = 0

        # Array for game values (state, action, reward, next, done)
        last_vars = [None, None]
        winner = -1
        amount_won = None

        done = False
        iterations = 0
        while not done and iterations < 40:
            iterations += 1

            # At showdown get the winner
            if env.current_round == BettingRound.SHOWDOWN:
                showdown_count += 1
                _, _, _, winner, amount_won = env.step(ActionType.CHECK)
                break

            # Current environment state and next action
            current_state = get_state_vector(env)
            current_player = env.get_current_player_state()
            qv_next_action = agent.get_next_action(current_state, env)
            
            # If the current player is all in, just check and then continue
            if current_player.stack == 0:
                env.step(ActionType.CHECK, 0)
                continue

            # Take action, remember it, and train from replay buffer
            reward, done, illegal, winner, amount_won = env.step(*qv_action_to_action(qv_next_action, env))
            # Always give a small reward for making a move that is legal
            if not illegal:
                reward += LEGAL_PSREWARD
            scaled_reward = reward / chip_values[current_player.player - 1]
            result_state = get_state_vector(env)
            if illegal:
                illegal_count += 1
            agent.add_experience(current_state, qv_next_action, scaled_reward, result_state, done)

            action_count += 1
            if action_count % REPLAY_EVERY == 0:
                agent.replay_train()

            # Now update the saved state variables
            last_vars[current_player.player - 1] = (current_state, qv_next_action, scaled_reward, result_state, done)

        # Compute the final rewards
        if amount_won is None:
            bet_per_player = 0
        else:
            bet_per_player = amount_won // 2

        # Chop pot just splits and takes a reward for making it to showdown
        if winner is None:
            if last_vars[0] is not None:
                p1 = last_vars[0]
                agent.add_experience(p1[0], p1[1], SHOWDOWN_REWARD, None, True)
            if last_vars[1] is not None:
                p2 = last_vars[1]
                agent.add_experience(p2[0], p2[1], SHOWDOWN_REWARD, None, True)

        # Winner gets a reward and loser gets a penalty
        win_index = 0 if winner == Player.Player1 else 1
        lose_index = 1 if winner == Player.Player1 else 0

        # Winner gets scaled chips won plus an extra reward for making it to showdown
        if last_vars[win_index] is not None:
            r = bet_per_player / chip_values[win_index]
            if env.current_round == BettingRound.SHOWDOWN:
                r += SHOWDOWN_REWARD
            d = last_vars[win_index]
            agent.add_experience(d[0], d[1], r, d[3], True)

        # Loser gets a penalty for losing the chips but gets credit for making it to showdown
        if last_vars[lose_index] is not None:
            r = -LOSS_WEIGHT * (bet_per_player / chip_values[lose_index])
            if env.current_round == BettingRound.SHOWDOWN:
                r += SHOWDOWN_REWARD
            d = last_vars[lose_index]
            agent.add_experience(d[0], d[1], r, d[3], True)

        action_count += 1
        if action_count % REPLAY_EVERY == 0:
            agent.replay_train()

        # Final checkpoint
        if episode == EPISODES - 1:
            episode += 1
            checkpoint()

    log("Training completed successfully.")
    elapsed = round(time.time() - start)
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    log(f"Time elapsed: {hours}h {minutes}m {seconds}s")
    agent.model.save_checkpoint("model/final", "final")
    log(f"Saved final model to model/final/final.pth")

    # Save data to graph if I need it later
    with open("model/final/data.txt", "w") as f:
        f.write("action_counts = " + str(action_counts) + "\n")
        f.write("illegal_counts = " + str(illegal_counts) + "\n")
        f.write("showdowns = " + str(showdowns) + "\n")
        f.write("losses = " + str(agent.losses) + "\n")

if __name__ == "__main__":
    train()