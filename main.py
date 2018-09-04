

import sys
import pandas as pd
import csv
import numpy as np

from task import Task
from agent import DDPG

from keras import layers, models, optimizers
from keras import backend as K

import matplotlib.pyplot as plt




#from agents.agent import DDPG
#from task import Task

# Modify the values below to give the quadcopter a different starting position.


#labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
#          'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
#          'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
#results = {x : [] for x in labels}

def train():
        
    runtime = 5.                                     # time limit of the episode
    init_pose = np.array([0., 0., 4.0, 0., 0., 0.0])  # initial pose
    init_velocities = np.array([0., 0., 0.0])         # initial velocities
    init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities
    file_output = 'rewards.txt'                         # file name for saved results

    num_episodes = 10
    target_pos = np.array([0., 0., 40.])
    task = Task(init_pose=init_pose,init_velocities=init_velocities, init_angle_velocities=init_angle_velocities, target_pos=target_pos)
    agent = DDPG(task) 

    labels = ['episod', 'avg_reward', 'total_reward']
    results = {x : [] for x in labels}

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
                    print(task.sim.pose)
                    #to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(rotor_speeds)
                    #for ii in range(len(labels)):
                    #    results[labels[ii]].append(to_write[ii])
                    #writer.writerow(to_write)
                    
                    to_write = [i_episode] + [avg_reward] + [total_reward]
                    for ii in range(len(labels)):
                        results[labels[ii]].append(to_write[ii])                    
                    print("\rEpisode = {:4d}, total_reward = {:7.3f}, avg_reward={:7.3} (best = {:7.3f})".format(
                        i_episode, total_reward, avg_reward, best_total_reward), end="")  # [debug]
                    break
            sys.stdout.flush()
            


    return agent

def test(agent):
    # perform simulation     
    #from task import Task
    
    fig, sub1 = plt.subplots(1, 1)
    sub2 = sub1.twinx()
    
    def plt_dynamic(x, y1, y2, color_y1 = 'g', color_y2 = 'b'):                
        sub1.plot(x, y1, color_y1)
        sub1.plot(x, y2, color_y2)        
        plt.show()       
        

    time_limit = 5
    y1_lower = 0
    y1_upper = 60
    y2_lower = 0
    y2_upper = 120

    sub1.set_xlim(0, time_limit)  # this is typically time
    sub1.set_ylim(y1_lower, y1_upper)  # limits to your y1
    sub2.set_xlim(0, time_limit)  # time, again
    sub2.set_ylim(y2_lower, y2_upper)  # limits to your y2

    # set labels and colors for the axes
    sub1.set_xlabel('time (s)', color='k')
    sub1.tick_params(axis='x', colors='k')

    sub1.set_ylabel('z-height', color='g')
    sub1.tick_params(axis='y', colors="g")

    sub2.set_ylabel('total reward', color='b')
    sub2.tick_params(axis='y', colors='b')

    display_graph = True
    display_freq = 1

    # Modify the values below to give the quadcopter a different starting position.
    runtime = 5.                                     # time limit of the episode
    init_pose = np.array([0., 0., 1., 0., 0., 0.])  # initial pose
    init_velocities = np.array([0., 0., 0.])         # initial velocities
    init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities
    file_output = 'data_agent.txt'                         # file name for saved results

    # Setup
    task = Task(init_pose, init_velocities, init_angle_velocities, runtime)
    agent.task = task

    done = False
    labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
            'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
            'psi_velocity', 'rotor_speed']
    results_test = {x : [] for x in labels}

    task.init_pose = init_pose
    task.target_pos = np.array([0., 0., 40.])

    # Run the simulation, and save the results.
    with open(file_output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)
        state = agent.reset_episode() # start a new episode    
        x, y1, y2 = [], [], []

        total_reward = 0
        while True:
            action = agent.act_no_noise(state) 
            next_state, reward, done = task.step(action)
            state = next_state 
            
            total_reward += reward
            
            x.append(task.sim.time)  # x: time
            y1.append(task.sim.pose[2])  # y1: z-height
            y2.append(total_reward)  # y2: total reward

            print(reward)
            print(task.sim.pose[:3])
            to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(action)
            for ii in range(len(labels)):
                results_test[labels[ii]].append(to_write[ii])
            writer.writerow(to_write)
            if done:
                print('done')
                print(task.sim.pose[:3])
                break

        plt_dynamic(x, y1, y2)

if __name__ == '__main__':
    agent = train()
    test(agent)