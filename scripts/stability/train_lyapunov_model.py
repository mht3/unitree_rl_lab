import sys, os

from matplotlib import pyplot as plt
import torch
import numpy as np
from lyapunov_trainer.models.neural_lyapunov_model import NeuralLyapunovModel
from lyapunov_trainer.loss import LyapunovRisk, CircleTuningLoss
from lyapunov_trainer.falsifier import Falsifier
from lyapunov_trainer.trainer import Trainer

import gymnasium as gym 

class G1BalanceLyapunovTrainer(Trainer):
    def __init__(self, model, policy, lr, optimizer, loss_fn, dt=0.02, circle_tuning_loss_fn=None, falsifier=None, loss_mode='approx_lie'):
        super().__init__(model, policy, lr, optimizer, loss_fn, dt, circle_tuning_loss_fn, falsifier, loss_mode)
        # use env to get s, a, s' pairs and use finite difference approximation
        self.env = gym.make('Pendulum-v1', g=9.81)

    def step(self, X, u):
        '''
        Generates all X_primes needed given current state and current action
        X: current angle and angular velocity
        u: input torque for inverted pendulum
        '''
        # take step in environment based upon current state and action
        N = X.shape[0]
        u_numpy = u.cpu().detach().numpy()
        X_prime = torch.empty_like(X)
        for i in range(N):
            x_i = X[i, :].detach().numpy()
            # set environment as x_i
            observation, info = self.env.reset()
            # get current action to take 
            u_i = u_numpy[i, :]
            self.env.unwrapped.state = x_i
            # take step in environment
            observation, reward, terminated, truncated, info = self.env.step(u_i)
            # add sample to X_prime
            X_prime[i, :] = torch.from_numpy(self.env.unwrapped.state)

        X_prime[:, 0] = InvertedPendulumTrainer.normalize_angle(X_prime[:, 0])

        return X_prime
    
def plot_loss(true_loss, filename):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(range(len(true_loss)), true_loss, label='True Loss')

    plt.ylabel('Lyapunov Risk', size=16)
    plt.xlabel('Epochs', size=16)
    plt.grid()
    plt.legend()
    plt.savefig(filename)

def load_model():

    d_in, n_hidden = ?
    controller = NeuralLyapunovModel(d_in, n_hidden)
    return controller


def load_state(state_min, state_max, N=500):
    # X: Nxlen(state_min) tensor of initial states
    X = torch.empty(N, 0)
    for i in range(len(state_min)):
        s_min = state_min[i]
        s_max = state_max[i]
        x = torch.Tensor(N, 1).uniform_(s_min, s_max)
        X = torch.cat([X, x], dim=1)
    return X

if __name__ == '__main__':
    torch.random.manual_seed(42)

    ### Generate random training data ###
    # number of samples
    N = 500
    '''
    bounds for position, velocity, angle, and angular velocity
    position: -2.4 to 2.4
    velocity: -2 to 2
    theta: -12 to 12 degrees
    thata_dot: -2 to 2
    '''
    # make samples closer to equilibrium
    state_min = [-np.pi/4, -np.pi/4]
    state_max = [np.pi/4, np.pi/4]
    # load 500 length 2 vectors of the state at random
    X = load_state(state_min, state_max, N=500)
    # stable conditions (used for V(x_0) = 0)
    theta_eq, theta_dot_eq = 0., 0.
    X_0 = torch.Tensor([theta_eq, theta_dot_eq])

    ### Start training process ##
    lr = 0.01
    ### Load falsifier
    falsifier = Falsifier(state_min, state_max, epsilon=0., scale=0.05, frequency=100, num_samples=5)
    ### Start training process ##
    loss_fn = LyapunovRisk(lyapunov_factor=1., lie_factor=1., equilibrium_factor=1.)
    circle_tuning_loss_fn = CircleTuningLoss(state_max=np.mean(state_max), tuning_factor=0.1)
    ### load model and training pipeline with initialized LQR weights ###
    model_1 = load_model()
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=lr)
    trainer_1 = InvertedPendulumTrainer(model_1, lr, optimizer_1, loss_fn, circle_tuning_loss_fn=circle_tuning_loss_fn,
                        falsifier=falsifier, loss_mode='true')
    true_loss = trainer_1.train(X, X_0, epochs=1250, verbose=True)
    # save model corresponding to true loss
    torch.save(model_1.state_dict(), 'examples/inverted_pendulum/models/pendulum_lyapunov_model_1.pt')

    plot_loss(true_loss, 'examples/inverted_pendulum/results/true_loss.png')