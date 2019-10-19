#%%
import gym
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

np.random.seed(123)

env = gym.make('CartPole-v0')
env.seed(321)

episodes = 20000
test_episodes = 10
num_of_actions = 2

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

# For LunarLander, use the following values:
#         [  x     y  xdot ydot theta  thetadot cl  cr
# s_min = [ -1.2  -0.3  -2.4  -2  -6.28  -8       0   0 ]
# s_max = [  1.2   1.2   2.4   2   6.28   8       1   1 ]

# Parameters
gamma = 0.98
alpha = 0.1
target_eps = 0.1
a = int(20000/9)  # TODO: Set the correct value.
initial_q = 0  # T3: Set to 50

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)

discrete_grid = np.vstack((x_grid,v_grid,th_grid,av_grid))

q_grid = np.zeros((discr, discr, discr, discr, num_of_actions)) + initial_q
#%%
def plot_heatmap(q_grid,it):
    values = np.zeros(q_grid.shape[:-1])  # TODO: COMPUTE THE VALUE FUNCTION FROM THE Q-GRID
    values = np.max(q_grid,axis=4)
    heatmap = np.mean(values,axis=(1,3))
    # Plot the heatmap
    # TODO: Plot the heatmap here using Seaborn or Matplotlib
    sns.heatmap(heatmap)
    plt.title(f"Value function heatmap at {it} iterations")
    plt.xlabel("x")
    plt.ylabel(r"$\theta$")
    plt.xticks(range(16),[str(round(x,1)) for x in x_grid])
    plt.yticks(range(16),[str(round(x,1)) for x in th_grid])
    plt.show()

def state_grid_loc(state):
    # subtract the state from the discrete grid buckets
    # Then find the location of the bucket by argmin over the columns
    loc = np.argmin(np.abs(state-discrete_grid.T),axis=0)
    loc = tuple(loc)
    return loc 

def update_q(q_grid,state,action,reward,new_state):
    state_action = (*state_grid_loc(state),action)
    q_sa = q_grid[state_action]
    new_action = np.argmax(q_grid[state_grid_loc(new_state)])
    new_state_action = (*state_grid_loc(new_state),new_action)
    q_sa = q_sa + alpha*(reward+gamma*q_grid[new_state_action] - q_sa)
    q_grid[state_action] = q_sa

# Training loop
ep_lengths, epl_avg = [], []
for ep in range(episodes+test_episodes):
    test = ep > episodes
    state, done, steps = env.reset(), False, 0
    epsilon = 0 #a/(a+ep)  # T1: GLIE/constant, T3: Set to 0
    while not done:
        # TODO: IMPLEMENT HERE EPSILON-GREEDY
        if np.random.rand() <= epsilon:
            action = int(np.random.rand()*2)
        else:
            action = q_grid[state_grid_loc(state)].argmax()
        new_state, reward, done, _ = env.step(action)
        if not test:
            # TODO: ADD HERE YOUR Q_VALUE FUNCTION UPDATE
            update_q(q_grid,state,action,reward,new_state)
        else:
            env.render()
        state = new_state
        steps += 1
    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[max(0, ep-500):]))
    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep-200):])))
    if ep ==0 or ep == episodes/2:
        plot_heatmap(q_grid,ep)
# Save the Q-value array
np.save("q_values.npy", q_grid)  # TODO: SUBMIT THIS Q_VALUES.NPY ARRAY

# Calculate the value function
values = np.zeros(q_grid.shape[:-1])  # TODO: COMPUTE THE VALUE FUNCTION FROM THE Q-GRID
values = np.max(q_grid,axis=4)
np.save("value_func.npy", values)  # TODO: SUBMIT THIS VALUE_FUNC.NPY ARRAY

plot_heatmap(q_grid,episodes)
# Draw plots
plt.plot(ep_lengths)
plt.plot(epl_avg)
plt.legend(["Episode length", "500 episode average"])
plt.title("Episode lengths")
plt.show()

