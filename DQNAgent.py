import random

import keras
import numpy as np
import tensorflow as tf


class DQNAgent:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.95  # 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99  # Discount factor
        self.tau = 1  # 0.001  # Soft update factor
        self.learning_rate = 0.001
        self.memory = []
        self.batch_size = 2 ** 10

        # self.model = QNetwork(state_size, action_size)
        self.model = keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(action_size),
        ])
        self.model.build((state_size,))
        self.model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
        )
        # self.target_model = QNetwork(state_size, action_size)
        self.target_model = keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(action_size),
        ])
        self.target_model.build((state_size,))

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate)

    def forget(self):
        self.memory = self.memory[-(self.batch_size * 1):]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, exploration: bool = True):
        if exploration and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(
            np.expand_dims(state, axis=0),
            verbose=0,
            use_multiprocessing=True,
        )
        return np.argmax(q_values[0])

    # @profile(stream=open('replay_profile.txt', 'w+'))
    def replay(self):
        if len(self.memory) < self.batch_size:
            # print(f"Memory size {len(self.memory)}")
            return
        minibatch = random.sample(self.memory, self.batch_size)

        states = []
        targets = []

        for state, action, reward, next_state, done in minibatch:
            target = self.target_model.predict(
                np.expand_dims(state, axis=0),
                verbose=0,
                use_multiprocessing=True,
            )
            if done:
                target[0][action] = reward
            else:
                q_future = np.max(
                    self.target_model.predict(
                        np.expand_dims(next_state, axis=0),
                        verbose=0,
                        use_multiprocessing=True,
                    )
                )
                target[0][action] = reward + self.gamma * q_future
            states.append(state)
            # states.append(np.expand_dims(state, axis=0))
            targets.append(target)

        self.model.fit(
            np.array(states),
            np.array(targets),
            epochs=1,
            verbose=0,
            use_multiprocessing=True,
        )

    # def update_target_model(self):
    #     self.target_model.set_weights(self.model.get_weights())

    def update_target_model(self):
        q_network_weights = self.model.get_weights()
        target_network_weights = self.target_model.get_weights()

        for i in range(len(q_network_weights)):
            target_network_weights[i] = self.tau * q_network_weights[i] + (1 - self.tau) * target_network_weights[i]

        self.target_model.set_weights(target_network_weights)

    @staticmethod
    def check_last_n_elements(lst, n):
        # This function takes two parameters: a list (lst) and a number (n).
        # It returns True if the last n elements of the list are the same, and False otherwise.

        # lst[-n:] extracts the last n elements of the list.
        # set(lst[-n:]) converts these elements into a set, which only contains unique elements.
        # len(set(lst[-n:])) gets the number of unique elements among the last n elements.
        # If all the last n elements are the same, this will be 1.
        return len(set(lst[-n:])) == 1

    def train(self, env, episodes):
        score_history = []
        for episode in range(episodes):
            print(f"Episode: {episode}, simulating")

            if len(score_history) > 50 and self.check_last_n_elements(score_history, 10):
                return score_history

            state = env.reset()
            total_reward = 0
            total_hits = 0
            self.forget()

            for _ in range(self.batch_size):
                # Add several new examples
                done = False
                while not done:
                    action = self.act(state)
                    total_hits += action

                    next_state, reward, done, _ = env.step(action)
                    total_reward += reward

                    self.remember(state, action, reward, next_state, done)
                    state = next_state

            # print(f"Episode: {episode}, training")
            self.replay()
            if self.epsilon > self.epsilon_min:
                print(f"epsilon {self.epsilon}")
                self.epsilon *= self.epsilon_decay
            self.update_target_model()

            # Free up GPU memory after each episode
            tf.keras.backend.clear_session()

            # print(f"Episode: {episode}, Total Reward: {total_reward}\n")
            # print(f"Episode: {episode}, Total Hits: {total_hits}")
            score = self.test_model_vs_basic_strategy(verbose=False)
            score_history.append(score)
            print(f"Score history size {len(score_history)}")

        return score_history

    def test_model_vs_basic_strategy(self, verbose: bool):
        total_reward_model = 0
        total_reward_basic_strategy = 0

        # Create all possible player and dealer hand combinations
        player_hands = list(range(4, 20))  # Player hands range from 4 to 20
        soft_player_hands = list(range(12, 20))  # Player hands range from 12 to 20
        dealer_upcards = list(range(2, 11))  # Dealer upcards range from 2 to 11

        for player_hand in player_hands:
            for dealer_upcard in dealer_upcards:
                # Initialize the state for each combination of player and dealer hands
                state = (player_hand, dealer_upcard, False)

                # Model's action
                model_action = self.act(state, exploration=False)

                # Basic strategy action
                basic_strategy_action = self.get_basic_strategy_action(player_hand, dealer_upcard, False)

                total_reward_basic_strategy += 1 if model_action == basic_strategy_action else 0

                if verbose and model_action != basic_strategy_action:
                    model_predictions = self.model.predict(
                        np.expand_dims(state, axis=0),
                        verbose=0
                    )
                    print(f"{state} - {model_predictions}\n"
                          f"{model_action} {basic_strategy_action}\n")

        for soft_player_hand in soft_player_hands:
            for dealer_upcard in dealer_upcards:
                # Initialize the state for each combination of player and dealer hands
                state = (soft_player_hand, dealer_upcard, True)

                # Model's action
                model_action = self.act(state, exploration=False)

                # Basic strategy action
                basic_strategy_action = self.get_basic_strategy_action(soft_player_hand, dealer_upcard, True)

                total_reward_basic_strategy += 1 if model_action == basic_strategy_action else 0

                if verbose and model_action != basic_strategy_action:
                    model_predictions = self.model.predict(
                        np.expand_dims(state, axis=0),
                        verbose=0
                    )
                    print(f"{state} - {model_predictions}\n"
                          f"{model_action} {basic_strategy_action}\n")

        print(
            f"Basic Strategy Total Reward: {total_reward_basic_strategy} "
            f"/ {len(player_hands) * len(dealer_upcards) + len(soft_player_hands) * len(dealer_upcards)}\n")

        return total_reward_model

    @staticmethod
    def get_basic_strategy_action(player_hand, dealer_upcard, usable_ace):
        """
        Returns the ideal play for the player given a blackjack game state.

        Args:
            player_hand (int): The value of the player's hand.
            dealer_upcard (int): The value of the dealer's up-card.
            usable_ace (bool): True if the player has a soft ace, False otherwise.

        Returns:
            int: The recommended play for the player. 0 for 'stand', 1 for 'hit'.
        """

        if (not usable_ace and player_hand < 4) or (usable_ace and player_hand < 12) or player_hand > 21:
            raise ValueError(f"Invalid player hand value {player_hand}. Must be between 4 and 21.")

        if dealer_upcard < 2 or dealer_upcard > 11:
            raise ValueError(f"Invalid dealer up-card value {dealer_upcard}. Must be between 2 and 11.")

        i_player_hand = player_hand - 12 if usable_ace else player_hand - 4
        i_dealer_upcard = dealer_upcard - 2

        if usable_ace:
            return basic_strategy_soft_grid[i_player_hand][i_dealer_upcard]
        else:
            return basic_strategy_hard_grid[i_player_hand][i_dealer_upcard]


basic_strategy_hard_grid = [
    # Dealer Upcard
    # 2, 3, 4, 5, 6, 7, 8, 9, 10, A
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],  # *4*#
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],  # *5*#
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],  # *6*#
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],  # *7*#
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],  # *8*#
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],  # *9*#
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],  # *10*#
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],  # *11*#
    [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, ],  # *12*#
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, ],  # *13*#
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, ],  # *14*#
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, ],  # *15*#
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, ],  # *16*#
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # *17*#
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # *18*#
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # *19*#
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # *20*#
]

basic_strategy_soft_grid = [
    # Dealer Upcard
    # 2, 3, 4, 5, 6, 7, 8, 9, 10, A
    # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],  # *4*#
    # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],  # *5*#
    # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],  # *6*#
    # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],  # *7*#
    # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],  # *8*#
    # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],  # *9*#
    # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],  # *10*#
    # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],  # *11*#
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],  # *12*#
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],  # *13*#
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],  # *14*#
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],  # *15*#
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],  # *16*#
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],  # *17*#
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, ],  # *18*#
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # *19*#
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # *20*#
]
