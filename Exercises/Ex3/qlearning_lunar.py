#%%
import gym
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(123)

env = gym.make('LunarLander-v2')
env.seed(321)

episodes = 20000
test_episodes = 10
num_of_actions = 4

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -1.2, 1.2
y_min, y_max = -0.3, 1.2
vx_min, vx_max = -2.4, 2.4
vy_min, vy_max = -2, 2
th_min, th_max = -6.28, 6.28
av_min, av_max = -8, 8
cl_min, cl_max = 0, 1
cr_min, cr_max = 0, 1


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
y_grid = np.linspace(y_min, y_max, discr)
vx_grid = np.linspace(vx_min, vx_max, discr)
vy_grid = np.linspace(vy_min, vy_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)
# cl_grid = np.linspace(cl_min,cl_max,2)
# cr_grid = np.linspace(cr_min,cr_max,2)

discrete_grid = np.vstack((x_grid,y_grid,vx_grid,vy_grid,th_grid,av_grid))

q_grid = np.zeros((discr, discr,discr,discr, discr, discr,2,2, num_of_actions)) + initial_q
#%%

def state_grid_loc(state):
    # subtract the state from the discrete grid buckets
    # Then find the location of the bucket by argmin over the columns
    loc = np.argmin(np.abs(state[:-2]-discrete_grid.T),axis=0)
    loc = np.append(loc,state[-2:].astype(np.int))
    loc = tuple(loc)
    return loc 

def update_q(q_grid,state,action,reward,new_state,done):
    state_action = (*state_grid_loc(state),action)
    q_sa = q_grid[state_action]
    new_action = np.argmax(q_grid[state_grid_loc(new_state)])
    new_state_action = (*state_grid_loc(new_state),new_action)
    if not done:
        q_next_sa = q_grid[new_state_action]
    else:
        q_next_sa = 0
    q_sa = q_sa + alpha*(reward+gamma*q_next_sa - q_sa)
    q_grid[state_action] = q_sa

# Training loop
ep_lengths, epl_avg = [], []
rewards, rewards_avg =[], []

for ep in range(episodes+test_episodes):
    test = ep > episodes
    state, done, steps = env.reset(), False, 0
    cumulative_rewards =0
    epsilon = a/(a+ep)  # T1: GLIE/constant, T3: Set to 0
    while not done:
        # TODO: IMPLEMENT HERE EPSILON-GREEDY
        if np.random.rand() <= epsilon:
            action = np.random.randint(4)
        else:
            action = q_grid[state_grid_loc(state)].argmax()
        new_state, reward, done, _ = env.step(action)
        if not test:
            # TODO: ADD HERE YOUR Q_VALUE FUNCTION UPDATE
            update_q(q_grid,state,action,reward,new_state,done)
        else:
            env.render()
        state = new_state
        steps += 1
        cumulative_rewards+= reward
    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[max(0, ep-500):]))
    rewards.append(cumulative_rewards)
    rewards_avg.append(np.mean(rewards[max(0,ep-500):]))
    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}, average rewards: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep-200):]), np.mean(rewards[max(0, ep-200):])))

# Save the Q-value array
np.save("q_values.npy", q_grid)  # TODO: SUBMIT THIS Q_VALUES.NPY ARRAY

# # Calculate the value function
# values = np.zeros(q_grid.shape[:-1])  # TODO: COMPUTE THE VALUE FUNCTION FROM THE Q-GRID
# values = np.max(q_grid,axis=4)
# np.save("value_func.npy", values)  # TODO: SUBMIT THIS VALUE_FUNC.NPY ARRAY

# heatmap = np.mean(values,axis=(1,3))
# # Plot the heatmap
# # TODO: Plot the heatmap here using Seaborn or Matplotlib
# plt.imshow(heatmap)
# plt.show()
# Draw plots
plt.plot(ep_lengths)
plt.plot(epl_avg)
plt.legend(["Episode length", "500 episode average"])
plt.title("Episode lengths")
plt.show()

plt.plot(rewards)
plt.plot(rewards_avg)
plt.legend(["Cumulative Rewards", "500 episode average"])
plt.title("Cumulative Rewards ")
plt.show()


#%%
