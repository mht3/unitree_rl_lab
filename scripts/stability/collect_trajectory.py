import numpy as np
import sys, os
import torch

class LyapunovDatasetCollector():

    def __init__(self, env, policy_path, N=500):
        self.env = env.unwrapped
        # numer of samples in dataset (total # of different trajectories)
        self.N = N
        self.policy = self.load_model(policy_path)

    def load_model(self):
        pass

    def get_action(self, x):
        '''
        Take in single observation x to get action 
        '''
        # this RL model needs the previous action as input, so we store a class variable for action
        self.action = self.rl_model.predict(x, yold=self.action)
        return self.action
    
    def get_next_state(self, x, pi):
        '''
        Generates next state given current state and current action
        x: current state
        pi: current action based on policy
        '''
        # take step in environment (0.1 seconds)
        # environment already knows about current state x so we don't need to use it here
        observation, reward, terminated, truncated, info = self.env.step(pi)

        return observation

    def build(self):

        initial_targets = self.load_initial_targets()

        # get 3 (s, pi(s), s') pairs
        # trajectory_size = self.trajectory_length * len(self.sta)
        trajectories = []
        for i in range(self.N):
            # numpy array for initial targets
            target = initial_targets[i, :]
            # reset environment for new trajectory
            self.env.reset(target_init=target)

            # get current state based on initialization
            x = self.env.get_state()
            # list of tuples of (s, pi(s), s') pairs
            trajectory = self.get_trajectory(x)
            trajectories.append(trajectory)

        return trajectories


    def get_trajectory(self, x):
        trajectory = []
        # get action for current state x
        pi = self.get_action(x)
        # get next state
        x_prime = self.get_next_state(x, pi)
        trajectory.append((x, pi, x_prime))


        for j in range(self.trajectory_length - 1):
            # perturb state and use as next trajectory
            if(j%5 == 0):
                x = x_prime + np.random.normal(0, self.scaled_sigma, len(x_prime))
            else:
                x = x_prime
            # get action
            pi = self.get_action(x)
            # get next state
            x_prime = self.get_next_state(x, pi)

            # add to trajectories
            trajectory.append((x, pi, x_prime))
            

        return trajectory

    def load_initial_targets(self):
        # X: Nx3 numpy array of initial states
        X = np.empty(shape=(self.N, 0))
        for i in range(len(self.env.low_target)):
            t_min = self.env.low_target[i]
            t_max = self.env.high_target[i]
            #x = np.random.uniform(t_min, t_max, size=(self.N, 1))
            # Changing initial states (targets) to have 20% buffer relative to low_target and high_target
            t_range = t_max - t_min
            x = np.random.uniform(t_min+0.2*t_range, t_max-0.2*t_range, size=(self.N, 1))
            X = np.concatenate([X, x], axis=1)
        return X

    def save(self, data, filename='trajectories.npz'):
        flattened_data = self.convert_data(data)
        np.savez_compressed(filename, *flattened_data)

    def convert_data(self, data):
        flattened = []
        for trajectory in data:
            for state,action,next_state in trajectory:
                flattened.append((state,action,next_state))
        all_states, all_actions, all_next_states = zip(*flattened)
        return np.array(all_states), np.array(all_actions), np.array(all_next_states)

def load_torch(filename):
    '''
    Loads trajectories.npz as a flattened torch tensor of (s, a, s')
    '''
    loaded = np.load(filename, allow_pickle=True)
    loaded_data = [loaded[key] for key in loaded]
    states, actions, next_states = loaded_data
    dataset = np.hstack([states, actions, next_states])
    data_tensor = torch.tensor(dataset)
    return data_tensor.to(torch.float32)

def load(filename, trajectory_length=40):
    '''
    Loads trajectories.npz
    '''
    loaded = np.load(filename, allow_pickle=True)
    loaded_data = [loaded[key] for key in loaded]
    states, actions, next_states = loaded_data
    trajectories = []
    for i in range(0, len(states), trajectory_length):
        trajectory = [(states[j], actions[j], next_states[j]) for j in range(i, i + trajectory_length)]
        trajectories.append(trajectory)
    return trajectories


if __name__ == '__main__':
    trajectory_length = 40
    dataset = KSTARLyapunovDataset(trajectory_length=trajectory_length, N=10, scaled_noise=True)
    trajectories = dataset.build()
    print('##### Trajectories ######')
    print(len(trajectories[0]))
    dataset.save(trajectories, filename='trajectories_sparsenoise2.npz')
    print('##### Loaded Data ######')
    loaded_data = load('trajectories_sparsenoise2.npz', trajectory_length)
    print(len(loaded_data[0]))
 