#%%
import numpy as np
from time import sleep
from sailing import SailingGridworld
from itertools import product

# Set up the environment
env = SailingGridworld(rock_penalty=-10)
value_est = np.zeros((env.w, env.h))
env.draw_values(value_est)

def get_action_values(transitions,gamma,value_est):
    a = np.zeros(len(transitions))
    for i,t in enumerate(transitions):
        for next_s,reward,done,prob in t:
            a[i] += prob*(reward + gamma* (value_est[next_s] if not done else 0))
    return a

#%%
if __name__ == "__main__":
    # Reset the environment
    state = env.reset()
#%%
    # Compute state values and the policy
    # TODO: Compute the value function and policy (Tasks 1, 2 and 3)
    value_est, policy = np.zeros((env.w, env.h)), np.zeros((env.w, env.h))
    gamma = 0.9
    epsilon = 1e-4
    delta = 0
    iterations = 0 
#%%
    N =100
    while True:
        delta=0
        for state in  product(range(env.w),range(env.h)):
            v = value_est[state]
            transitions = env.transitions[state]
            action_values = get_action_values(transitions,gamma,value_est)
            max_action_value = action_values.max()
            delta = max(delta,np.abs(v-max_action_value))
            value_est[state] = max_action_value
        iterations+=1
        print(f"{delta:e}")
        if delta < epsilon:
            break
        # env.draw_values(value_est)
        # env.render()
    print(f"Value Iteration took {iterations} number of iterations")

    for state in product(range(env.w),range(env.h)):
        transitions = env.transitions[state]
        action_values = get_action_values(transitions,gamma,value_est)    
        policy[state] = action_values.argmax()

#%%
    # Show the values and the policy
    env.draw_values(value_est)
    env.draw_actions(policy)
    env.render()
    sleep(1)
    input("Something bro")
    # Save the state values and the policy
    fnames = "values.npy", "policy.npy"
    np.save(fnames[0], value_est)
    np.save(fnames[1], policy)
    print("Saved state values and policy to", *fnames)

    # Run a single episode
    # TODO: Run multiple episodes and compute the discounted returns (Task 4)
    done = False
    while not done:
        # Select a random action
        # TODO: Use the policy to take the optimal action (Task 2)
        # action = int(np.random.random()*4)
        action = policy[state]
        # Step the environment
        state, reward, done, _ = env.step(action)

        # Render and sleep
        env.render()
        # sleep(0.5)



#%%
