import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        self.fc2_state_val  = torch.nn.Linear(self.hidden,1)
        self.sigma = torch.tensor([np.sqrt(10)],dtype=torch.float32,device=self.train_device)  # TODO: Implement accordingly (T1, T2)
        self.sigma = torch.nn.Parameter(self.sigma)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x,ep):
        x = self.fc1(x)
        x = F.relu(x)
        mu = self.fc2_mean(x)
        sigma  = self.sigma
        # TODO: Instantiate and return a normal distribution
        # with mean mu and std of sigma (T1)
        # TODO: Add a layer for state value calculation (T3)
        v_s = self.fc2_state_val(x)
        return Normal(mu,sigma),v_s
        

class Agent(object):
    def __init__(self, policy,normalize=False):
        self.train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.state_values =[]
        self.normalize = normalize

    def episode_finished(self, episode_number):
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        state_values = torch.stack(self.state_values, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards, self.state_values = [], [], [],[]


        next_states = torch.cat((state_values[1:], torch.tensor([0.])))
        delta = rewards + self.gamma * next_states - state_values 
        critic_loss = torch.mean(-delta.detach()*state_values)
        # critic_loss = torch.mean(torch.pow(delta,2))/2

        policy_loss = torch.mean(-delta.detach()*action_probs)
        loss = policy_loss + critic_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm(list(self.policy.parameters()) , max_norm=0.8)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False,ep=0):
        x = torch.from_numpy(observation).float().to(self.train_device)

        # TODO: Pass state x through the policy network (T1)
        act_dist,state_value = self.policy.forward(x,ep)
        # TODO: Return mean if evaluation, else sample from the distribution
        # returned by the policy (T1)
        if evaluation :
            action = act_dist.mean
        else:
            action = act_dist.sample()
        # TODO: Calculate the log probability of the action (T1)
        act_log_prob = act_dist.log_prob(action)
        # TODO: Return state value prediction, and/or save it somewhere (T3)

        return action, act_log_prob, state_value

    def store_outcome(self, observation, action_prob, action_taken, reward,state_value):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.state_values.append(state_value)

