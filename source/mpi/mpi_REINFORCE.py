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
# Initialize MPI
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
##############################

##############################
random_seed = 20  
lr = 0.01
environment = 'LunarLander-v2'
# number of episode for 1 update
batch_size = 8
##############################

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

        
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
        
        # action & reward memory
        self.saved_actions = []
        self.rewards = []
        self.returns = []

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
        probs, state_value = self.forward(state)
        m = Categorical(probs)
        action = m.sample()        
        
        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()

    def calculate_return(self, gamma=0.99):
        reversed_rewards = np.flip(self.rewards, 0)
        g_t = 0
        returns = []
        for r in reversed_rewards:
            g_t = r + gamma * g_t
            returns.insert(0, g_t)
        returns = torch.tensor(returns).float()
        self.returns.append(torch.squeeze(returns))
        del self.rewards[:]


    def calculate_loss(self):
        # Initialize the lists and variables
        R = 0
        saved_actions = self.saved_actions
        # policy_losses = [] 
        # value_losses = [] 

        log_probs = torch.cat([a.log_prob for a in saved_actions])
        values = torch.cat([a.value for a in saved_actions])
        returns = torch.cat(self.returns)

        # print(returns)
        values = torch.squeeze(values)

        returns = (returns - returns.mean()) / (returns.std())

        advantage = returns  - values
        policy_lose = sum(-log_probs * advantage)
        value_loss = sum((returns - values)**2)
        
        return policy_lose + value_loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.returns[:]
        del self.rewards[:]
        del self.saved_actions[:]


def train(lr=0.01):
    '''
        Train the model using SGD (via backpropagation)
        TODO: In each episode, 
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode
    '''    
    action_time = 0
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

    if rank == 0:
        time_start = time.time()
    
    # run inifinitely many episodes
    i_episode = 0

    ep_process = int(batch_size / size)
    remain = batch_size % size
    if rank < remain:
        ep_process += 1

    while True:
        i_episode += batch_size

        # reset environment and episode reward
        ep_reward = 0
        t = 0
        # Learning rate scheduler
        
        # For each episode, only run 9999 steps so that we don't 
        # infinite loop while learning

        if rank == 0:
            start_action = MPI.Wtime()
        weights = comm.bcast(model.state_dict(), root = 0)
        if rank != 0:
            model.load_state_dict(weights)

        for i in range(ep_process):
            state = env.reset()
            t_ep = 0
            while t_ep < 9999:
                t_ep += 1
                action = model.select_action(state)
                state, reward, done, _ = env.step(action)
                model.rewards.append(reward)
                if done:
                    break
            t += t_ep
            ep_reward += sum(model.rewards)
            model.calculate_return()

        loss = model.calculate_loss()
        optimizer.zero_grad()
        loss.backward()

        grads = [p.grad for p in model.parameters()]
        grads = comm.gather(grads, root=0)

        ep_reward = comm.gather(ep_reward, root=0)

        # print('[DEGUB] t: ', t)
        t = comm.gather(t, root=0)
        # print(t)

        if rank == 0:
            ep_reward = np.sum(ep_reward) / batch_size
            t = np.sum(t) / batch_size
            # print(grads)
            for i in range(1, size):
                j = 0
                for p in model.parameters():
                    p.grad += grads[i][j]
                    j += 1
            # print('[DEBUG] sum: ', [p.grad for p in model.parameters()])
            end_action = MPI.Wtime()
            action_time += end_action - start_action

            optimizer.step()
            scheduler.step()
                
            # update EWMA reward and log the results
            ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
            print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

            ER.append(ewma_reward)
            R.append(ep_reward)

        model.clear_memory()

        solved = False

        # check if we have "solved" the cart pole problem
        # ewma_reward > env.spec.reward_threshold or 
        if rank == 0 and (i_episode >= 3000):
            solved = True

        solved = comm.bcast(solved, root = 0)

        if rank == 0 and solved:
            time_end = time.time()
            # torch.save(model.state_dict(), './preTrained/LunarLander_{}.pth'.format(lr))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))

            # plt.plot(range(1, i_episode+1), R, 'r:')
            # plt.plot(range(1, i_episode+1), ER, 'b')
            # plt.legend(['ewma reward', 'ep reward'])
            # plt.savefig('LunarLander.png')
            # plt.show()
            time_c= time_end - time_start
            print('time cost', time_c, 's')
            print('time taking action', action_time, 's')
            break
        elif solved:
            break

        # break


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
    env.seed(random_seed * rank)  
    torch.manual_seed(random_seed * rank)  
    train(lr)
    # if rank == 0:
    #     test('LunarLander_0.01.pth')