

import sys
import pandas as pd
import csv
import numpy as np

from task import Task
from agent import DDPG


from keras import layers, models, optimizers
from keras import backend as K

#from agents.agent import DDPG
#from task import Task

# Modify the values below to give the quadcopter a different starting position.
runtime = 5.                                     # time limit of the episode
init_pose = np.array([0., 0., 0.0, 0., 0., 0.])  # initial pose
init_velocities = np.array([0., 0., 0.0])         # initial velocities
init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities
file_output = 'rewards.txt'                         # file name for saved results

num_episodes = 1000
target_pos = np.array([0., 0., 10.])
task = Task(init_pose=init_pose,init_velocities=init_velocities, init_angle_velocities=init_angle_velocities, target_pos=target_pos)
agent = DDPG(task) 

labels = ['episod', 'avg_reward', 'total_reward']
results = {x : [] for x in labels}

#labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
#          'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
#          'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
#results = {x : [] for x in labels}

with open(file_output, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(labels)  
    best_total_reward = -1000
    for i_episode in range(1, num_episodes+1):
        state = agent.reset_episode() # start a new episode
        total_reward = 0
        rewards = []
        while True:
            
            # select action according to the learned policy and the exploration noise            
            action = agent.act(state) 
            # execute the action and observe the reward and the next state
            next_state, reward, done = task.step(action)
            
                
            # sample mini batch and learn
            agent.step(action, reward, next_state, done)

            # data tracking
            total_reward += reward
            rewards.append(reward)
            
            if total_reward > best_total_reward:
                best_total_reward = total_reward

            state = next_state
            if done:
                avg_reward = np.mean(np.array(rewards))
                #print(rewards)
                #to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(rotor_speeds)
                #for ii in range(len(labels)):
                #    results[labels[ii]].append(to_write[ii])
                #writer.writerow(to_write)
                
                to_write = [i_episode] + [avg_reward] + [total_reward]
                for ii in range(len(labels)):
                    results[labels[ii]].append(to_write[ii])
                writer.writerow(to_write)
                print("\rEpisode = {:4d}, total_reward = {:7.3f}, avg_reward={:7.3} (best = {:7.3f})".format(
                    i_episode, total_reward, avg_reward, best_total_reward), end="")  # [debug]
                break
        sys.stdout.flush()