#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning policies
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import numpy as np
from queue import PriorityQueue
from MBRLEnvironment import WindyGridworld

class DynaAgent:
    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma # discount factor
        self.Q_sa = np.zeros((n_states, n_actions))
        # TO DO: Initialize count tables, and reward sum tables.
        self.n = np.zeros((n_states, n_actions, n_states))
        self.r = np.zeros((n_states, n_actions, n_states))
        
    def select_action(self, state, epsilon):
        # If a random number is within the explore rate we explore by choosing a random action
        if np.random.uniform(0,1) <= epsilon:
            return np.random.choice(range(self.n_actions))
        
        # If we don't explore we choose the action with the highest reward 
        # If there are multiple actions with the same reward we choose the first one
        return np.argmax(self.Q_sa[state])
        
    def update(self, state, action, reward, done, next_state, n_planning_updates):
        if done:
            return
        
        self.n[state][action][next_state] += 1 
        self.r[state][action][next_state] += reward
        
        self.Q_sa[state][action] += self.learning_rate * (reward + self.gamma * max(self.Q_sa[next_state])-self.Q_sa[state][action])

        for _ in range(n_planning_updates):
            # Choose random state with n>0
            state_sums = np.sum(self.n, axis=(1,2))
            state = np.random.choice(np.arange(state_sums.size)[state_sums>0])

            # Choose previously taken action for state s  
            action_sums = np.sum(self.n[state], axis=1)
            action = np.random.choice(np.arange(action_sums.size)[action_sums>0])
            
            # Choose next state
            p_hat = self.n[state][action]/np.sum(self.n[state][action])
            next_state = np.random.choice(np.arange(self.n_states), p=p_hat)

            reward = self.r[state][action][next_state]/self.n[state][action][next_state]
            
            # Update q value
            self.Q_sa[state][action] += self.learning_rate * (reward + self.gamma * max(self.Q_sa[next_state])-self.Q_sa[state][action])
    
class PrioritizedSweepingAgent:
    def __init__(self, n_states, n_actions, learning_rate, gamma, max_queue_size=200, priority_cutoff=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma # discount factor
        self.priority_cutoff = priority_cutoff
        self.queue = PriorityQueue()
        
        self.Q_sa = np.zeros((n_states,n_actions))
        # TO DO: Initialize count tables, and reward sum tables. 
        self.n = np.zeros((n_states, n_actions, n_states))
        self.r = np.zeros((n_states, n_actions, n_states))

    def select_action(self, state, epsilon):
        # If a random number is within the explore rate we explore by choosing a random action
        if np.random.uniform(0,1) <= epsilon:
            return np.random.choice(range(self.n_actions))
        
        # If we don't explore we choose the action with the highest reward 
        # If there are multiple actions with the same reward we choose the first one
        return np.argmax(self.Q_sa[state])
        
    def update(self,state,action,reward,done,next_state,n_planning_updates):
        # TO DO: Add own code
        
        # Helper code to work with the queue
        # Put (s,a) on the queue with priority p (needs a minus since the queue pops the smallest priority first)
        # self.queue.put((-p,(s,a))) 
        # Retrieve the top (s,a) from the queue
        # _,(s,a) = self.queue.get() # get the top (s,a) for the queue
        if done:
            return
        
        self.n[state][action][next_state] += 1 
        self.r[state][action][next_state] += reward
        
        self.Q_sa[state][action] += self.learning_rate * (reward + self.gamma * max(self.Q_sa[next_state])-self.Q_sa[state][action])

        p = abs(reward + self.gamma * max(self.Q_sa[next_state])-self.Q_sa[state][action])
        if p > self.priority_cutoff:
            self.queue.put((-p,(state,action)))


        for _ in range(n_planning_updates):
            if self.queue.empty():
                break
            p,(state, action) = self.queue.get()

             # Choose next state
            p_hat = self.n[state][action]/np.sum(self.n[state][action])
            next_state = np.random.choice(np.arange(self.n_states), p=p_hat)
            
            reward = self.r[state][action][next_state]/self.n[state][action][next_state]

            # Update q value
            self.Q_sa[state][action] += self.learning_rate * (reward + self.gamma * max(self.Q_sa[next_state])-self.Q_sa[state][action])

            # Loop over all state actions that may lead to state s
            for previous_state in range(self.n_states):
                for action in range(self.n_actions):
                    if self.n[previous_state][action][state] > 0:

                        # Calculate reward and priority value
                        reward = self.r[previous_state][action][state]/self.n[previous_state][action][state]
                        p = abs(reward + self.gamma * max(self.Q_sa[previous_state])-self.Q_sa[previous_state][action])
                        if p > self.priority_cutoff:
                            self.queue.put((-p,(previous_state,action)))

def test():

    n_timesteps = 1000
    gamma = 0.99

    # Algorithm parameters
    policy = 'ps' # 'ps' or 'dyna' 
    epsilon = 0.1
    learning_rate = 0.5
    n_planning_updates = 5

    # Plotting parameters
    plot = True
    plot_optimal_policy = True
    step_pause = 0.0001
    
    # Initialize environment and policy
    env = WindyGridworld()
    if policy == 'dyna':
        pi = DynaAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize Dyna policy
    elif policy == 'ps':    
        pi = PrioritizedSweepingAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize PS policy
    else:
        raise KeyError('Policy {} not implemented'.format(policy))
    
    # Prepare for running
    s = env.reset()  
    continuous_mode = False
    cumulative_r = 0
    for t in range(n_timesteps):            
        # Select action, transition, update policy
        a = pi.select_action(s,epsilon)
        s_next,r,done = env.step(a)
        cumulative_r +=r
        pi.update(s, a, r, done, s_next,n_planning_updates=n_planning_updates)
        
        # Render environment
        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=plot_optimal_policy,
                       step_pause=step_pause)
            
        # Ask user for manual or continuous execution
        if not continuous_mode:
            key_input = input("Press 'Enter' to execute next step, press 'c' to run full algorithm")
            continuous_mode = True if key_input == 'c' else False

        # Reset environment when terminated
        if done:
            s = env.reset()
        else:
            s = s_next
    print(f"Cumulative reward {cumulative_r}")
            
    
if __name__ == '__main__':
    test()
