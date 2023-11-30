import math
from random import randint
from random import random
from enum import Enum
import numpy as np

# This program uses reinforcement learning to determine the optimal policy
# for Approach.
# Recall that approach works like this:
# Both players agree on a limit n.
# Player 1 rolls first. They go until they either exceed n or hold.
# Then player 2 rolls. They go until they either exceed n or beat player 1's score.
# The player who is closest to n without going over wins.
# Note:
# We can reduce this to the problem of player 1 choosing the best value at which to hold.
# This is called a policy; once we know the best number to hold at, we can act optimally.

epsilon = 0.1
alpha = 0.1
discount = 0.9


def approach(n):
    q_table = [[random() / 100.0, random() / 100.0] for i in range(n)]
    hold = 0
    roll = 1

    for i in range(100000):
        # print('\n')
        # Select an initial state.
        s = randint(0, (n-1))
        # print(s)
        a = randint(hold, roll)

        # Take the best move with p=epsilon, and the worst move with p=1-epsilon.
        # Continue playing until the game is done.
        while True:
            if s == n:
                break

            if random() <= epsilon:
                a = randint(hold, roll)
            else:
                a = hold if q_table[s][hold] >= q_table[s][roll] else roll

            if a == hold:
                break
            else:
                # s += randint(1, 6)
                # if s >= n:
                #     break
                roll_val = randint(1, 6)
                if s + roll_val >= n:
                    #print('no addition')
                    break
                s += roll_val
                #print('addition')

        # If you win, reward = 1.
        # If you lose, reward = 0.
        r = 1 if s == n else 0
        # Use Q-learning to update the q-table for each state-action pair visited.
        # print(s)
        val = q_table[s][a]
        new_val = np.max(q_table[s])
        q_table[s][a] = (1 - alpha) * val + alpha * (r + (1 - epsilon) * new_val)

    action_sequence = ['roll' if q_table[i][roll] > q_table[i][hold] else 'hold' for i in range(n - 1)]
    print(action_sequence)
    return q_table
    # After 100000 iterations, print out your q-table.
