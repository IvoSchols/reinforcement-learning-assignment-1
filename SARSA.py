#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class SarsaAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
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
        
    def update(self,s,a,r,s_next,a_next,done):
        backup_estimate = r+self.gamma*self.Q_sa[s_next,a_next]
        self.Q_sa[s,a] += self.learning_rate * (backup_estimate - self.Q_sa[s,a])

        
def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []

    s = env.reset()
    a = pi.select_action(s, policy, epsilon, temp)

    for i in range(n_timesteps):
        s_next,r,done = env.step(a)
        a_next = pi.select_action(s_next, policy, epsilon, temp)
        rewards.append(r)

        pi.update(s, a, r, s_next, a_next, done)

        if done:
            s = env.reset()
            a = pi.select_action(s, policy, epsilon, temp)
            print('done')
        else:
            s = s_next
            a = a_next
    
        # if plot and i % 200 == 0:
        #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during SARSA execution

    return rewards 


def test():
    n_timesteps = 2500
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(rewards))        
    
if __name__ == '__main__':
    test()
