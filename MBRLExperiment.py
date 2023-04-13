#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning experiments
Practical for course "Reinforcement Learning",
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import numpy as np
from MBRLEnvironment import WindyGridworld
from MBRLAgents import DynaAgent,PrioritizedSweepingAgent
from Helper import LearningCurvePlot, smooth

def run_repetitions(policy, n_repetitions, n_timesteps, smoothing_window, learning_rate, gamma,
                    epsilon, n_planning_updates):
    # Write all your experiment code here
    # Look closely at the code in test() of MBRLAgents.py for an example of the execution loop
    # Log the obtained rewards during a single training run of n_timesteps, and repeat this proces n_repetitions times
    # Average the learning curves over repetitions, and then additionally smooth the curve
    # Be sure to turn environment rendering off! It heavily slows down your runtime
    
    cumulative_rewards = np.zeros(n_timesteps)
    
    for i in range(n_repetitions):
        rewards = np.zeros(n_timesteps)

        # Initialize environment and policy
        env = WindyGridworld()
        if policy == "Dyna":
            pi = DynaAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize Dyna policy
        elif policy == "Prioritized Sweeping":    
            pi = PrioritizedSweepingAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize PS policy
        else:
            raise KeyError(f"Policy {policy} not implemented")
        
        # Prepare for running
        s = env.reset()  
        total_r = 0
        for t in range(n_timesteps):            
            # Select action, transition, update policy
            a = pi.select_action(s,epsilon)
            s_next,r,done = env.step(a)
            total_r += r
            rewards[t] = total_r
            pi.update(s, a, r, done, s_next,n_planning_updates=n_planning_updates)
            
            # Reset environment when terminated
            if done:
                s = env.reset()
            else:
                s = s_next
        cumulative_rewards += rewards
    cumulative_rewards /= n_repetitions
    
    # Apply additional smoothing
    learning_curve = smooth(cumulative_rewards,smoothing_window) # additional smoothing
    
    return learning_curve


def experiment():

    n_timesteps = 10000
    n_repetitions = 10
    smoothing_window = 101
    gamma = 0.99

    for policy in ["Dyna","Prioritized Sweeping"]:
    
        ##### Assignment a: effect of epsilon ######
        learning_rate = 0.5
        n_planning_updates = 5
        epsilons = [0.01,0.05,0.1,0.25]
        Plot = LearningCurvePlot(title = "{}: effect of $\epsilon$-greedy".format(policy))
        
        for epsilon in epsilons:
            learning_curve=run_repetitions(policy, n_repetitions, n_timesteps, smoothing_window, 
                                           learning_rate, gamma, epsilon, n_planning_updates)
            Plot.add_curve(learning_curve,label="$\epsilon$ = {}".format(epsilon))
        Plot.save("{}_egreedy.png".format(policy))
        
        ##### Assignment b: effect of n_planning_updates ######
        epsilon=0.1
        n_planning_updatess = [1,5,15]
        learning_rate = 0.5
        Plot = LearningCurvePlot(title = "{}: effect of number of planning updates per iteration".format(policy))

        for n_planning_updates in n_planning_updatess:
            learning_curve=run_repetitions(policy, n_repetitions, n_timesteps, smoothing_window, 
                                           learning_rate, gamma, epsilon, n_planning_updates)
            Plot.add_curve(learning_curve,label="Number of planning updates = {}".format(n_planning_updates))
        Plot.save("{}_n_planning_updates.png".format(policy))  
        
        ##### Assignment 1c: effect of learning_rate ######
        epsilon=0.1
        n_planning_updates = 5
        learning_rates = [0.1,0.5,1.0]
        Plot = LearningCurvePlot(title = "{}: effect of learning rate".format(policy))
    
        for learning_rate in learning_rates:
            learning_curve=run_repetitions(policy, n_repetitions, n_timesteps, smoothing_window, 
                                           learning_rate, gamma, epsilon, n_planning_updates)
            Plot.add_curve(learning_curve,label="Learning rate = {}".format(learning_rate))
        Plot.save("{}_learning_rate.png".format(policy)) 
    
if __name__ == "__main__":
    experiment()
    print(["123", "456"][0])
        