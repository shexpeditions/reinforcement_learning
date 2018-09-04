import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0 # 0
        self.action_high = 800 # 900
        self.action_size = 1        

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 0.5
        # encourage positive difference between the z coordinates of 
        reward += np.tanh(self.sim.v[2])
        dz = self.target_pos[2] - self.sim.pose[2]
        reward += 0.5 * np.exp(-(dz**2)/ 25.0)
                
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []

        counter = 1
        #speed = 500.0
        #print([speed] * 4)
        speed_const = rotor_speeds * 4
        #print(rotor_speeds)
        for _ in range(self.action_repeat):            
            done = self.sim.next_timestep(speed_const) # update the sim pose and velocities
            reward += self.get_reward()
            if done == True:
                reward -= 0.2
                #print('crashed')

            if self.sim.pose[2] >= (self.target_pos[2] - 0.05):
                done = True
                reward += 0.1
                #print('target reached')
            #print(self.sim.pose)
            pose_all.append(self.sim.pose)
            counter += 1
        # clip reward function
        reward /= float(counter)        

        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state

    def sigmoid(self, value):
        return 1.0 / (1.0 + np.exp(-value))
