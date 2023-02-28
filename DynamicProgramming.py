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
from Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states    # n_states = gride_size = length * width
        self.n_actions = n_actions  # n_actions = [up, right, down, left]
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))

    def select_action(self,s): # Down
        ''' Returns the greedy best action in state s '''
        actions = self.Q_sa[s]
        #a = np.argmax(actions)
        a = argmax(actions)
        return a
        
    def update(self,s,a,p_sas,r_sas): # Up
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        Q_sa_new = 0
        for state in range(self.n_states):
            Q_sa_new += p_sas[s,a,state] * (r_sas[s,a,state]+self.gamma*self.Q_sa[state, self.select_action(state)])

            
        self.Q_sa[s][a] = Q_sa_new
        return Q_sa_new

    
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)
    QIagent.i = 0
    while True:
        max_error = 0
        QIagent.i += 1
        for state in range(QIagent.n_states):
            for action in range(QIagent.n_actions):
                # Q_sa
                x = QIagent.Q_sa[state,action]
                # Q_sa_prime
                QIagent.Q_sa[state,action] = QIagent.update(state,action,env.p_sas,env.r_sas)
                # Keep max error: max_error or |Q_sa-Q_sa_prime|
                max_error = max(max_error,abs(x-QIagent.Q_sa[state,action]))

        # Plot current Q-value estimates & print max error
        #env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.05)
        print("Q-value iteration, iteration {}, max error {}".format(QIagent.i,max_error))
        if max_error < threshold:
            return QIagent



def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)
    
    sum_reward_per_timestep = 0
    i = 0

    # View optimal policy
    done = False
    s = env.reset()
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        sum_reward_per_timestep += r
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.5)
        s = s_next
        i += 1

    #mean_reward_per_timestep = sum_reward_per_timestep/QIagent.i
    mean_reward_per_timestep = sum_reward_per_timestep/i
    print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))

if __name__ == '__main__':
    experiment()
