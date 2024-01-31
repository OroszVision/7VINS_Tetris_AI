import numpy as np
import cv2 as cv
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
import sys
from engine import Tetris
from agent import Agent

# Initialize tetris environment
env = Tetris(10, 20)

# Initialize training variables
max_episode = 3000
max_steps = 25000


import numpy as np
import cv2 as cv
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
import sys
from engine import Tetris
from agent import Agent

# Initialize tetris environment
env = Tetris(10, 20)

# Initialize training variables
max_episode = 3000
max_steps = 25000

# Define the episode generation string
EPISODE_GEN = "900_1050"

# Use f-string to format the full file path
file_path = f'Generation {EPISODE_GEN}/tetris_agent_generation_{EPISODE_GEN}.pkl'

with open(file_path, 'rb') as agent_file:
    agent = pickle.load(agent_file)
    # Determine the last completed episode
    last_completed_episode = agent.epsilon
    print(last_completed_episode)

# The rest of your code remains the same
# ...


with open(file_path, 'rb') as agent_file:
    agent = pickle.load(agent_file)
    # Determine the last completed episode
    last_completed_episode = agent.epsilon
    print(last_completed_episode)
# Lists to store data for plotting
episode_rewards = []
episode_steps = []
highest_score = 0
highest_score_episode = 0

# Create the figure outside the loop
fig, ax = plt.subplots(figsize=(12, 6))

# Initialize empty data for the graph
game_numbers = []
game_scores = []

for episode in range(max_episode):
    current_state = env.reset()
    done = False
    total_reward = 0
    episode_step = 0

    while not done and episode_step < max_steps:
        env.render(total_reward)

        next_states = env.get_next_states()

        if not next_states:
            break

        best_state = agent.act(next_states.values())

        best_action = None
        for action, state in next_states.items():
            if (best_state == state).all():
                best_action = action
                break

        reward, done = env.step(best_action)
        total_reward += reward

        agent.add_to_memory(current_state, next_states[best_action], reward, done)

        current_state = next_states[best_action]

        episode_step += 1

    # Store the episode's statistics
    episode_rewards.append(total_reward)
    episode_steps.append(episode_step)

    game_numbers.append(episode)
    game_scores.append(total_reward)

    if total_reward > highest_score:
        highest_score = total_reward
        highest_score_episode = episode

    agent.replay()

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon -= agent.epsilon_decay

    # Update the plot data
    ax.clear()
    sns.lineplot(x=game_numbers, y=game_scores, label='Episode Reward', ax=ax)
    ax.set_xlabel('Game Number')
    ax.set_ylabel('Total Reward')
    ax.set_title(f'Game Rewards - Highest: {highest_score} (Game {highest_score_episode})')
    ax.grid(True)

    fig.tight_layout()
    
    # Display the plot during training
    plt.pause(0.01)


# Keep the plot window open until manually closed
plt.show()
