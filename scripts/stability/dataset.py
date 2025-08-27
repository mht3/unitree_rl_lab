import numpy as np
import sys, os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

import gymnasium as gym

from utils import *

'''
Trajectory Collection for the IsaacLab G1 Balance Environment.
History size of 5 is used (original observation is size 78)
+----------------------------------------------------------+
| Active Observation Terms in Group: 'policy' (shape: (390,)) |
+-----------+--------------------------------+-------------+
|   Index   | Name                           |    Shape    |
+-----------+--------------------------------+-------------+
|     0     | base_ang_vel                   |    (15,)    |
|     1     | projected_gravity              |    (15,)    |
|     2     | velocity_commands              |    (15,)    |
|     3     | joint_pos_rel                  |    (115,)   |
|     4     | joint_vel_rel                  |    (115,)   |
|     5     | last_action                    |    (115,)   |
+-----------+--------------------------------+-------------+
+----------------------------------------------------------+
| Active Observation Terms in Group: 'critic' (shape: (405,)) |
+-----------+--------------------------------+-------------+
|   Index   | Name                           |    Shape    |
+-----------+--------------------------------+-------------+
|     0     | base_lin_vel                   |    (15,)    |
|     1     | base_ang_vel                   |    (15,)    |
|     2     | projected_gravity              |    (15,)    |
|     3     | velocity_commands              |    (15,)    |
|     4     | joint_pos_rel                  |    (115,)   |
|     5     | joint_vel_rel                  |    (115,)   |
|     6     | last_action                    |    (115,)   |
+-----------+--------------------------------+-------------+
[INFO] Action Manager:  <ActionManager> contains 1 active terms.
+-----------------------------------------+
|     Active Action Terms (shape: 23)     |
+-------+---------------------+-----------+
| Index | Name                | Dimension |
+-------+---------------------+-----------+
|   0   | JointPositionAction |        23 |
+-------+---------------------+-----------+

'''
class TrajectoryCollector():

    def __init__(self, env, policy, N=500, state_action_only=True):
        self.env = env
        # numer of different trajectories
        self.N = N
        # trained torch policy. Assumes policy.predict method exists 
        self.policy = policy
        self.state_action_only = state_action_only

    def get_action(self, x):
        '''
        Take in single observation x to get action 
        '''
        # this RL model needs the previous action as input, so we store a class variable for action
        with torch.inference_mode():
            action = self.policy(x)
        return action

    def build(self):
        trajectories = []
        for i in tqdm(range(self.N)):
            # reset environment for new trajectory
            obs, _ = env.get_observations()
            # list of tuples of (s, pi(s), s') pairs
            trajectory = self.get_trajectory(obs)
            trajectories.append(trajectory)

        return trajectories

    def get_trajectory(self, x):
        terminate = False
        truncate = False

        trajectory = []
        while not truncate and not terminate:
            # get action for current state x
            a = self.get_action(x)
            # get next state
            x_prime, reward, terminate, info = self.env.step(a)
            if self.state_action_only:
                trajectory.append((x, a))
            else:
                trajectory.append((x, a, x_prime))
            # update x to be next state
            x = x_prime

        return trajectory

    def save(self, trajectories, filename='trajectories.npz'):
        flattened_data = self.flatten_trajectories(trajectories)
        np.savez_compressed(filename, *flattened_data)

    def flatten_trajectories(self, trajectories):
        flattened = []
        for trajectory in trajectories:
            for data in trajectory:
                flattened.append(data)
        if self.state_action_only:
            all_states, all_actions = zip(*flattened)
            return np.array(all_states), np.array(all_actions)
        else:
            all_states, all_actions, all_next_states = zip(*flattened)
            return np.array(all_states), np.array(all_actions), np.array(all_next_states)

class TrajectoryDataset(Dataset):
    def __init__(self, filename, state_action_only=True):
        self.state_action_only = state_action_only
        # dataset of flattened (s, a, s') or (s, a) pairs.
        if state_action_only:
            states, actions = self.load_dataset(filename)

        else:
            states, actions, next_states = self.load_dataset(filename)
            self.next_states = torch.tensor(next_states, dtype=torch.float32)

        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)

    def load_torch(self, filename):
        '''
        Loads trajectories.npz as a flattened torch tensor of (s, a, s') or (s, a)
        '''
        loaded = np.load(filename, allow_pickle=True)
        loaded_data = [loaded[key] for key in loaded]
        return loaded_data
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.state_action_only:
            return self.states[idx], self.actions[idx]
        else:
            return self.states[idx], self.actions[idx], self.next_states[idx]



if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    # parser = argparse.ArgumentParser(description="Collect a dataset.")
    # cli_args.add_rsl_rl_args(parser)
    # args_cli = parser.parse_args()
    env_id = 'Unitree-G1-23dof-Balance'
    policy_path = os.path.join(cur_dir, 'policies/2025-08-25_17-08-47/model_100.pt')
    
    print(f"Loading environment: {env_id}")
    print(f"Loading policy from: {policy_path}")
    
    env, policy = load_env_and_policy(env_id=env_id, policy_path=policy_path)

    print('##### Data Collection ######')
    # save path
    save_path = os.path.join(cur_dir, 'g1_balance_5_newton.npz')
    num_trajectories = 5
    state_action_only = True
    dataset = TrajectoryCollector(env, policy=policy, N=num_trajectories, state_action_only=state_action_only)
    print('Building Dataset...')
    trajectories = dataset.build()
    print('Saving flattened dataset of {} tracectories...'.format(len(trajectories)))
    dataset.save(trajectories, filename=save_path)

    print('##### Pytorch Dataset ######')
    load_path = save_path
    loaded_data = TrajectoryDataset(load_path, state_action_only=state_action_only)
    print('Loaded {} transitions.'.format(len(loaded_data)))
 