import gymnasium as gym
import numpy as np
from sac import SAC  # Import your SAC implementation

# Environment setup
env = gym.make('InvertedPendulum-v4')
obs_space = env.observation_space
action_space = env.action_space

# Hyperparameters and logger setup (you'll need to define these)
single_agent_env_dict = {"obs_space": obs_space, "action_space": action_space}
hp_dict = {}  # Fill this with your hyperparameters
logger_kwargs = {}  # Fill this with your logger settings

# Initialize SAC agent
sac_agent = SAC(single_agent_env_dict, hp_dict, logger_kwargs, ma=False, train_or_test="train")

# Training loop
total_num_episodes = 10000  # Adjust as needed
seed = 42  # Set your desired seed

reward_over_episodes = []

for episode in range(total_num_episodes):
    obs, info = env.reset(seed=seed)
    episode_reward = 0
    done = False

    while not done:
        action = sac_agent.sample_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Store the transition in SAC's replay buffer
        sac_agent.store_transition(obs, action, reward, next_obs, terminated)
        
        obs = next_obs
        episode_reward += reward
        done = terminated or truncated

    reward_over_episodes.append(episode_reward)
    
    # Update the SAC agent
    sac_agent.update()

    if episode % 100 == 0:  # Print every 100 episodes
        avg_reward = np.mean(reward_over_episodes[-100:])
        print(f"Episode: {episode}, Average Reward (last 100 episodes): {avg_reward:.2f}")

# Close the environment
env.close()