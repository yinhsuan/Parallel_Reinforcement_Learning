# Spring 2022, IOC 5259 Reinforcement Learning
# HW1-partII: REINFORCE and baseline

import gym
from itertools import count
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler

import matplotlib.pyplot as plt
import time

##############################
random_seed = 20  
lr = 0.01
environment = 'LunarLander-v2'
# number of episode for 1 update
batch_size = 4
##############################

        
class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the shared layer(s), the action layer(s), and the value layer(s)
            2. Random weight initialization of each layer
    """
    def __init__(self):
        super(Policy, self).__init__()
        
        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]

        self.c1 = nn.Linear(self.observation_dim, 16)
        self.c2 = nn.Linear(16, 16)

        self.a1 = nn.Linear(16, 16)
        self.a2 = nn.Linear(16, self.action_dim)

        self.v1 = nn.Linear(16, 16)
        self.v2 = nn.Linear(16, 1)
        

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding 
              action probability distirbution and the state value
            TODO:
                1. Implement the forward pass for both the action and the state value
        """

        x = F.relu(self.c1(state))
        x = F.relu(self.c2(x))

        a = F.relu(self.a1(x))
        action_prob = F.softmax(self.a2(a), dim=1)

        v = F.relu(self.v1(x))
        state_value = F.relu(self.v2(v))

        return action_prob, state_value


    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply 
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """

        state = torch.FloatTensor(state).unsqueeze(0)

        # Tensor: state.shape = [8] -> [8, batch_size]

        # Tensor: probs.shape = [1, 4] 
        # Tensor: state_value.shape = [1, 1] -> [1, batch_size]
        probs, state_value = self.forward(state)

        m = Categorical(probs)

        # Tensor: action.shape = [1]
        action = m.sample()        

        return action.item(), m.log_prob(action), state_value

def calculate_return(rewards, gamma=0.99):
    reversed_rewards = np.flip(rewards, 0)
    g_t = 0
    returns = []
    for r in reversed_rewards:
        g_t = r + gamma * g_t
        returns.insert(0, g_t)
    returns = torch.tensor(returns).float()
    return torch.squeeze(returns)


def calculate_loss(log_probs, values, returns):
    log_probs = torch.cat(log_probs)
    values = torch.cat(values) # (before) List: values/ (after) Tensor: values = [N, 1]
    returns = torch.tensor(returns)

    values = torch.squeeze(values) # Tensor: values = [N] -> [N, batch_size]
    returns = (returns - returns.mean()) / (returns.std())

    advantage = returns  - values
    policy_lose = sum(-log_probs * advantage)
    value_loss = sum((returns - values)**2)
    
    return policy_lose + value_loss


def train(lr=0.01):
    '''
        Train the model using SGD (via backpropagation)
        TODO: In each episode, 
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode
    '''    
    ER = []
    R = []

    print("goal: ", env.spec.reward_threshold)
    
    # Instantiate the policy model and the optimizer
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    # EWMA reward for tracking the learning progress
    ewma_reward = 0

    time_start = time.time()
    
    # run inifinitely many episodes
    i_episode = 0
    while True:
        i_episode += batch_size

        # reset environment and episode reward
        ep_reward = 0
        t = 0

        log_probs = []
        values = []
        returns = []

        for i in range(batch_size):
            state = env.reset()
            t_ep = 0
            rewards = []
            while t_ep < 9999:
                t_ep += 1
                # take a step
                
                # Tensor: log_prob.shape = [1, 1] -> [N , 1, 1] or [N, 1]

                # Int: action
                # Tensor: log_prob.shape = [1] -> [N, 1]
                # Tensor: value.shape = [1, 1] -> [N , 1, 1] or [N, 1] -> [batch_size, 1]
                action, log_prob, value = model.select_action(state)
                # Tensor: state.shape = [8] -> [batch_size, 8]
                # Scalar: reward
                # Bool: done
                state, reward, done, _ = env.step(action)

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                if done:
                    break
            t += t_ep

            # Float: ep_reward
            ep_reward += sum(rewards)
            # list: returns
            returns += calculate_return(rewards)

        # Tensor: loss.shape = [] -> [N]
        loss = calculate_loss(log_probs, values, returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        ep_reward /= batch_size
        t /= batch_size
            
        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        ER.append(ewma_reward)
        R.append(ep_reward)

        # check if we have "solved" the cart pole problem
        if ewma_reward > env.spec.reward_threshold or i_episode >= 2000:
            time_end = time.time()
            torch.save(model.state_dict(), './preTrained/LunarLander_{}.pth'.format(lr))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))

            # plt.plot(range(1, i_episode+1), R, 'r:')
            # plt.plot(range(1, i_episode+1), ER, 'b')
            # plt.legend(['ewma reward', 'ep reward'])
            # plt.savefig('LunarLander.png')
            # plt.show()
            time_c= time_end - time_start
            print('time cost', time_c, 's')
            break


def test(name, n_episodes=10):
    '''
        Test the learned model (no change needed)
    '''      
    model = Policy()
    
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    
    render = True
    max_episode_len = 10000
    
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
    

if __name__ == '__main__':
    env = gym.make(environment)
    # Seed should be different among processes
    env.seed(random_seed)  
    torch.manual_seed(random_seed)  
    train(lr)
    # if rank == 0:
    #     test('LunarLander_0.01.pth')