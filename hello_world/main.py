from environment import MiningGame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# instantiate the game
game = MiningGame(n_mines=2)
cumulative_reward = 0

# in class
values = np.array([1,1]) # optimistic initialization both mines (why?)
values_history = np.zeros((150,2))
alpha = 0.1 # learning rate

# repeatedly ask user to choose a place to mine, and execute their choice
for step in range(150):
    
    #choice = input("choose a place to mine (0 or 1). q to quit: ")
    #if choice == 'q':
    #    break
    # started with greedy policy, choose highest
    # instead of greedy, can sometimes (10%?) pick a random action to continue exploration instead of self-confirming
    # world is always changing, if zero in on optimal sln, can't commit too hard to self-confirming loop, needs to adapt
    if np.random.rand() < 0.1:
        choice = np.random.choice([0,1])
    else:
        choice = np.argmax(values)

    # actual payout - perceived payout. Positive => payout higher than expected, bump value up
    # amount you bump it up by scaled by alpha (learning rate)

    reward = game.choose_mine(int(choice))
    cumulative_reward += reward
    
    values[choice] += alpha * (reward - values[choice])
    values_history[step, :] = values
    
    print(f"reward: {reward}, cumulative reward: {cumulative_reward}")
    
# print the true reward probabilities for mines 0 and 1
print('The true reward probabilities for mine 0/1 were:', game.reward_probabilities)
plt.plot(values_history)