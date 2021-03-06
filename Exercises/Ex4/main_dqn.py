#%%
try:
    notebook = True
    from IPython import get_ipython

    ipython = get_ipython()
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
    from Exercises.Ex4.rbf_agent import Agent as RBFAgent  # Use for Tasks 1-3
    from Exercises.Ex4.dqn_agent import Agent as DQNAgent  # Task 4
    from Exercises.Ex4.utils import plot_rewards,plot_final_rewards

except:
    notebook = False
    import matplotlib
    matplotlib.use('agg')
    from utils import plot_rewards,plot_final_rewards
    from rbf_agent import Agent as RBFAgent  # Use for Tasks 1-3
    from dqn_agent import Agent as DQNAgent  # Task 4

import gym
import numpy as np
from matplotlib import pyplot as plt

from itertools import count
import torch


#env_name = "CartPole-v0"
env_name = "LunarLander-v2"
env = gym.make(env_name)
env.reset()

# Set hyperparameters
# Values for RBF (Tasks 1-3)
glie_a = 50
num_episodes = 1000

# Values for DQN  (Task 4)
if "CartPole" in env_name:
    TARGET_UPDATE = 20
    glie_a = 200
    num_episodes = 5000
    hidden = 12
    gamma = 0.98
    replay_buffer_size = 50000
    batch_size = 32
elif "LunarLander" in env_name:
    TARGET_UPDATE = 20
    glie_a = 5000
    num_episodes = 15#000
    hidden = 64
    gamma = 0.95
    replay_buffer_size = 50000
    batch_size = 128
else:
    raise ValueError("Please provide hyperparameters for %s" % env_name)

# Get number of actions from gym action space
n_actions = env.action_space.n
state_space_dim = env.observation_space.shape[0]

# Tasks 1-3 - RBF
# agent = RBFAgent(n_actions)

# Task 4 - DQN
agent = DQNAgent(state_space_dim, n_actions, replay_buffer_size, batch_size,
              hidden, gamma)

# Training loop
#%%
cumulative_rewards = []
for ep in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    done = False
    eps = glie_a/(glie_a+ep)
    cum_reward = 0
    while not done:
        # Select and perform an action
        action = agent.get_action(state, eps)
        next_state, reward, done, _ = env.step(action)
        cum_reward += reward

        # Task 1: TODO: Update the Q-values        
        # Task 2: TODO: Store transition and batch-update Q-values
        # Task 4: Update the DQN
        # agent.single_update(state,action,next_state,reward,done)
        # agent.update_estimator()
        # Move to the next state
        agent.store_transition(state,action,next_state,reward,done)
        agent.update_network()
        state = next_state
    cumulative_rewards.append(cum_reward)
    if ep%50 == 0:
        print(f"Episode: {ep} Avg reward {np.mean(cumulative_rewards[max(0,ep-100):])}")
    # plot_rewards(cumulative_rewards)

    # Update the target network, copying all weights and biases in DQN
    # Uncomment for Task 4
    if ep % TARGET_UPDATE == 0:
        agent.update_target_network()

    # Save the policy
    # Uncomment for Task 4
    if ep % 1000 == 0:
        torch.save(agent.policy_net.state_dict(),
                  "weights_%s_%dmdl." % (env_name, ep))


print('Complete')
# plt.ioff()
# plt.show()
plot_final_rewards(cumulative_rewards,env_name)

# %%
