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

class MonteCarloAgent:

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
        
    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        T = len(states)-1
        backup_estimates = [0]
        for t in range(T):
            i = T-t # Reverse iterator
            backup_estimate = rewards[i] + self.gamma * backup_estimates[t] # Backup_estimates is chronological (confusing..)
            backup_estimates.append(backup_estimate)
            self.Q_sa[states[i]][actions[i]] += self.learning_rate * (backup_estimate - self.Q_sa[states[i]][actions[i]])


def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)

    all_rewards = []

    i = 0
    while i < n_timesteps:
        s = env.reset()

        states = []
        actions = []
        rewards = []

        for t in range(max_episode_length-1):
            i += 1
            a = pi.select_action(s, policy, epsilon, temp)
            s_next, r, done = env.step(a)

            states.append(s_next)
            actions.append(a)
            rewards.append(r)
            all_rewards.append(r)

            if done or i >= n_timesteps:
                break

        

        pi.update(states, actions, rewards)

        if plot and i % 200 == 0:
            print('timestep:{}', i)
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Monte Carlo RL execution

    return all_rewards 
    
def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(rewards))  
            
if __name__ == '__main__':
    test()
