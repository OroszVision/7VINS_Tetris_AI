import numpy as np
import cv2 as cv
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os  # Import the os module
import sys  # Import the sys module

sns.set(style="darkgrid")

from engine import Tetris
from agent import Agent

# Define the episode generation string
EPISODE_GEN = "1050_1200"

# Initialize tetris environment
env = Tetris(10, 20)

# Initialize training variables
max_episode = 3000
max_steps = 25000

if os.path.exists('tetris_agent.pkl'):
    with open('tetris_agent.pkl', 'rb') as agent_file:
        agent = pickle.load(agent_file)
        # Determine the last completed epsilon
        last_completed_epsilon = agent.epsilon
    print(f"Agent loaded from 'tetris_agent.pkl'. Continuing training from epsilon value {last_completed_epsilon}.")
else:
    agent = Agent(env.state_size)
    last_completed_epsilon = 1.0
    print("Agent initialized from scratch.")

# Lists to store data for plotting
episode_rewards = []
episode_steps = []
highest_score = 0
highest_score_episode = 0

# Create the figure outside the loop
fig, ax = plt.subplots(figsize=(12, 6))

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

    if total_reward > highest_score:
        highest_score = total_reward
        highest_score_episode = episode

    agent.replay()

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon -= agent.epsilon_decay

    # Update the plot data
    ax.clear()
    ax1 = plt.subplot(111)
    sns.lineplot(x=range(episode + 1), y=episode_rewards, label='Episode Reward', ax=ax1)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title(f'GENERATION ({EPISODE_GEN})')  # Moved the generation title here
    ax1.set_title(f'Episode Rewards - Highest: {highest_score} (Episode {highest_score_episode})', loc='left')  # Moved the original title to the left
    ax1.grid(True)

    # Add text to display the episode generation string
    ax1.text(episode, episode_rewards[-1], episode, fontsize=12, ha='right', va='bottom')

    fig.tight_layout()

    # Display the plot during training
    plt.pause(0.1)

    # Save the agent every 10 episodes into the same file
    if episode % 10 == 0:
        with open('tetris_agent.pkl', 'wb') as agent_file:
            pickle.dump(agent, agent_file)

    # Save the agent and graph after the specified episode into seperated folder
    if episode == 150:
        # Create the folder for the current episode generation
        generation_folder = f"Generation {EPISODE_GEN}"
        if not os.path.exists(generation_folder):
            os.makedirs(generation_folder)

        # Save the agent with the episode generation string in the folder
        agent_filename = os.path.join(generation_folder, f'tetris_agent_generation_{EPISODE_GEN}.pkl')
        with open(agent_filename, 'wb') as agent_file:
            pickle.dump(agent, agent_file)

        # Save the graph with the episode generation string in the folder
        graph_filename = os.path.join(generation_folder, f'training_graph_generation_{EPISODE_GEN}.png')
        plt.savefig(graph_filename)

        # Notify the user that the agent and graph have been saved
        print(f'Agent and graph saved for episode generation: {EPISODE_GEN} in the folder: {generation_folder}')
        print(f"Agent saved to 'tetris_agent_generation_{EPISODE_GEN}.pkl'. with its epsilon value {agent.epsilon}.")
        # Stop the code execution
        sys.exit()

# Keep the plot window open until manually closed
plt.show()
