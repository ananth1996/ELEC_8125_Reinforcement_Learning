import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from agent import Agent, Policy
from utils import get_space_dim
from pathlib import Path

# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None,
                        help="Model to be tested")
    parser.add_argument("--env", type=str, default="CartPole-v0",
                        help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=500,
                        help="Number of episodes to train for")
    parser.add_argument("--render_training", action='store_true',
                        help="Render each frame during training. Will be slower.")
    parser.add_argument("--x0", type=int,default=0,
                        help="Location to train near")
    parser.add_argument("--episode_length",type=int,default=200,
                        help="The number of timesteps in an episode")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    return parser.parse_args(args)


# Policy training function
def train(agent, env, train_episodes, early_stop=True, render=False,
          silent=False, train_run_id=0,x0=0,args=None,policy=None):
    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []

    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state
        observation = env.reset()

        # Loop until the episode is over
        while not done:
            # Get action from the agent
            action, action_probabilities = agent.get_action(observation)
            previous_observation = observation

            # Perform the action on the environment, get new state and reward
            observation, reward, done, info = env.step(action)

            # TODO: Task 1 - change the reward function
            reward = new_reward(observation,x0)

            # Store action's outcome (so that the agent can improve its policy)
            agent.store_outcome(previous_observation, action_probabilities, action, reward)

            # Draw the frame, if desired
            if render:
                env.render()

            # Store total episode reward
            reward_sum += reward
            timesteps += 1

        if not silent:
            print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  .format(episode_number, reward_sum, timesteps))

        if episode_number%20 ==0 :
            model_folder = Path(__file__).parent / "models"
            model_file = f"{args.env}_params{episode_number}.mdl"
        torch.save(policy.state_dict(), model_folder/model_file)
        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        # If we managed to stay alive for 15 full episodes, assume it's learned
        # (in the default setting)
        if early_stop and np.mean(timestep_history[-15:]) == env._max_episode_steps:
            if not silent:
                print("Looks like it's learned. Finishing up early")
            break

        # Let the agent do its magic (update the policy)
        agent.episode_finished(episode_number)

    # Store the data in a Pandas dataframe for easy visualization
    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id]*len(reward_history),
                         "reward": reward_history,
                         "mean_reward": average_reward_history})
    return data


# Function to test a trained policy
def test(agent, env, episodes, render=False,x0=0):
    test_reward, test_len = 0, 0
    for ep in range(episodes):
        done = False
        observation = env.reset()
        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(observation, evaluation=True)
            observation, reward, done, info = env.step(action)
            reward = new_reward(observation,x0)
            if render:
                env.render()
            test_reward += reward
            test_len += 1
    print("Average test reward:", test_reward/episodes, "episode length:", test_len/episodes)


def new_reward(state,x0):
    #! Swap function based on task
    return x0_reward(state,x0)

def fast_reward(state):
    return  0.3 + 0.7*np.abs(np.tanh(state[1]/2)) # little more ideal

def x0_reward(state,x0):
    return 1/(1+200*(x0-state[0])**2)  # for x0=1

def failed_x0_reward(state,x0):
    if x0 >= 0:
        if state[0]<=x0:
            return 1/(1+20*(x0-state[0])**2)
        else:
            return np.power(np.e,-100*np.abs(state[0]-x0))
    else:
        if state[0]<=x0:
            return np.power(np.e,-100*np.abs(state[0]-x0))
        else:
            return 1/(1+20*(x0-state[0])**2)
# The main function
def main(args):
    # Create a Gym environment
    env = gym.make(args.env)

    # Exercise 1
    env._max_episode_steps = args.episode_length

    # Get dimensionalities of actions and observations
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy)

    # Print some stuff
    print("Environment:", args.env)
    print("Training device:", agent.train_device)
    print("Observation space dimensions:", observation_space_dim)
    print("Action space dimensions:", action_space_dim)

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.
    if args.test is None:
        training_history = train(agent, env, args.train_episodes, False, args.render_training,x0=args.x0,args=args,policy=policy)

        # Save the model
        model_file = "%s_params.mdl" % args.env
        torch.save(policy.state_dict(), model_file)
        print("Model saved to", model_file)

        # Plot rewards
        sns.lineplot(x="episode", y="reward", data=training_history)
        sns.lineplot(x="episode", y="mean_reward", data=training_history)
        plt.legend(["Reward", "100-episode average"])
        plt.title("Reward history (%s)" % args.env)
        plt.show()
        print("Training finished.")
    else:
        print("Loading model from", args.test, "...")
        state_dict = torch.load(args.test)
        policy.load_state_dict(state_dict)
        print("Testing...")
        test(agent, env, args.train_episodes, args.render_test,x0=args.x0)


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)

