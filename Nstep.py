#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class NstepQLearningAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, n):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n = n
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        
        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")

            actions = self.Q_sa[s]
            probability = np.random.sample()

            if probability < epsilon:           # Exploration
                a = np.random.randint(0, self.n_actions)
            else:
                a = argmax(actions)             # Exploitation
            
            
                
        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")

            actions = self.Q_sa[s]

            a = np.random.choice(self.n_actions, p=softmax(actions, temp))
            
            
        return a
        
    def update(self, states, actions, rewards, done):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''

        # Compute n-step targets and upate
        T = len(states)-1
        for t in range(T):
            m = min(self.n, T-t)

            backup_estimate = 0
            # If s+m is terminal
            if t+m == len(rewards):
                for i in range(m-1):
                    backup_estimate += pow(self.gamma,i)*rewards[i]
            else:
                for i in range(m-1):
                    backup_estimate += pow(self.gamma,i)*rewards[i]+pow(self.gamma,m)*max(self.Q_sa[states[m]])
            self.Q_sa[states[t],actions[t]] += self.learning_rate * backup_estimate
            

def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma, n)

    all_rewards = []

    # TO DO: Write your n-step Q-learning algorithm here!
    for i in range(n_timesteps):
        s = env.reset()
        states = [s]
        actions = []
        rewards = []
        
        # Collect episode
        t = 0
        for t in range(max_episode_length-1):
            a = pi.select_action(s, policy, epsilon, temp)
            s_next,r,done = env.step(a)
            states.append(s_next)
            actions.append(a)
            rewards.append(r)
            all_rewards.append(r)
            if done:
                break
        
        T = t + 1
        pi.update(states, actions, rewards, done)

            
        
        if plot and i % 200 == 0:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution

    return all_rewards 

def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    print("Obtained rewards: {}".format(rewards))    
    
if __name__ == '__main__':
    test()
