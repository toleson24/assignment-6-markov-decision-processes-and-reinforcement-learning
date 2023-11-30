# Wrapper for Blackjack game:
# 1) initialize 2 Blackjack agents? 1 human, 1 computer
# 2) dealer draws and distributes cards
# 3) dealer draws and shows face-up card
# 4) human decides whether to stand or hit
# 5) computer decides whether to stand or hit
# 6) calculate win
#       7) if human (or computer) has over 21
#       8) loses
#       9) other plays until has over 21 (loses) or wins (21, or closer to 21 than other)

from __future__ import annotations

from collections import defaultdict

import numpy as np
from tqdm import tqdm

import gymnasium as gym

LEARNING_RATE = 0.01
N_EPISODES = 100_000
START_EPSILON = 1.0
EPSILON_DECAY = START_EPSILON / (N_EPISODES / 2)
FINAL_EPSILON = 0.1
ENV = gym.make("Blackjack-v1", sab=True)


class BlackjackAgent:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


class BlackjackAgentTrained(BlackjackAgent):

    def __init__(
            self,
            env=ENV,
            learning_rate=LEARNING_RATE,
            initial_epsilon=N_EPISODES,
            epsilon_decay=EPSILON_DECAY,
            final_epsilon=FINAL_EPSILON
    ):
        super().__init__(env, learning_rate, initial_epsilon, epsilon_decay, final_epsilon)

    def train(self):
        env = gym.make("Blackjack-v1", sab=True)
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=N_EPISODES)
        for _ in tqdm(range(N_EPISODES)):
            obs, info = env.reset()
            done = False

            # play one episode
            while not done:
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)

                # update the agent
                self.update(obs, action, reward, terminated, next_obs)

                # update if the environment is done and the current obs
                done = terminated or truncated
                obs = next_obs

            self.decay_epsilon()
        self.env.close()
