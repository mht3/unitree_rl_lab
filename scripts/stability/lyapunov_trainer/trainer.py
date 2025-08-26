'''
Generalized training class. Each specific domain will need to override this class.
'''
import torch

class Trainer():
    def __init__(self, model, policy, lr, optimizer, loss_fn, dt, circle_tuning_loss_fn=None, falsifier=None, loss_mode='true'):
        # neural lyapunov model
        self.model = model
        # trained policy (obs -> actions)
        self.policy = policy
        self.lr = lr
        self.optimizer = optimizer
        self.lyapunov_loss = loss_fn
        self.circle_tuning_loss = circle_tuning_loss_fn
        self.loss_mode = loss_mode
        self.falsifier = falsifier
        self.dt = dt
    
    def get_lie_derivative(self, X, V_candidate, f):
        '''
        Calculates L_V = ∑∂V/∂xᵢ*fᵢ
        '''
        w1 = self.model.layer1.weight
        b1 = self.model.layer1.bias
        w2 = self.model.layer2.weight
        b2 = self.model.layer2.bias
        # running through model again 
        z1 = X @ w1.t() + b1
        a1 = torch.tanh(z1)
        z2 = a1 @ w2.t() + b2
        d_z2 = 1. - V_candidate**2
        partial_z2_a1 = w2
        partial_a1_z1 = 1 - torch.tanh(z1)**2
        partial_z1_x = w1

        d_a1 = (d_z2 @ partial_z2_a1)
        d_z1 = d_a1 * partial_a1_z1

        # gets final ∂V/∂x
        d_x = d_z1 @ partial_z1_x

        lie_derivative = torch.diagonal((d_x @ f.t()), 0)
        return lie_derivative
    
    def get_approx_lie_derivative(self, V_candidate, V_candidate_next):
        '''
        Calculates L_V = ∑∂V/∂xᵢ*fᵢ by forward finite difference
                    L_V = (V' - V) / dt
        '''
        return (V_candidate_next - V_candidate) / self.dt

    def approx_f_value(self, X, X_prime):
        # Approximate f value with S, a, S'
        y = (X_prime - X) / self.dt
        return y

    def adjust_learning_rate(self, decay_rate=.9):
        # new_lr = lr * decay_rate
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] * decay_rate
    
    def reset_learning_rate(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def f_value(self, X, u):
        raise NotImplementedError("Override this function for your environment's dynamics.")

    def f_value_linearized(self, X, u):
        raise NotImplementedError("Override this function for your environment's linearized dynamics.")

    def step(self, X, u):
        raise NotImplementedError("Override this function to step through environment to get to the next state.")

    def train(self, X, x_0, alpha=0.7, epochs=1000, verbose=False, every_n_epochs=10, step_size=100, decay_rate=1.):
        self.model.train()
        no_counterexamples_ct = 0
        loss_list = []
        original_size = len(X)
        for epoch in range(1, epochs+1):
            # lr scheduler
            if (epoch + 1) % step_size == 0:
                self.adjust_learning_rate(decay_rate)
            # zero gradients
            self.optimizer.zero_grad()

            # get lyapunov function from model
            V_candidate = self.model(X)
            # get lyapunov function evaluated at equilibrium point
            V_X0 = self.model(x_0)

            # get input from trained NN policy
            u, _ = self.policy.predict(X)
            # get loss
            if self.loss_mode == 'true':
                # Compute lie derivative of V : L_V = ∑∂V/∂xᵢ*fᵢ
                f = self.f_value(X, u)
                L_V = self.get_lie_derivative(X, V_candidate, f)
            
            elif self.loss_mode == 'approx_dynamics':
                # compute the approximate dynamics lie derivative
                # balance between linearized dynamical system and from actual trajectory rollout
                X_prime = self.step(X, u)
                f_approx_linearized = self.f_value_linearized(X, u)
                f_approx_finite_difference = self.approx_f_value(X, X_prime)
                f_approx = alpha * f_approx_linearized + (1 - alpha) * f_approx_finite_difference
                L_V = self.get_lie_derivative(X, V_candidate, f_approx)
            elif self.loss_mode == 'approx_lie':
                # compute approximate f_dot and compare to true f
                X_prime = self.step(X, u)
                V_candidate_prime= self.model(X_prime)

                # compute lie derivative using finite difference methods
                L_V = self.get_approx_lie_derivative(V_candidate, V_candidate_prime).squeeze()
            
            loss = self.lyapunov_loss(V_candidate, L_V, V_X0)
            if self.circle_tuning_loss is not None:
                loss += self.circle_tuning_loss(X, V_candidate)
        
            loss_list.append(loss.item())
            loss.backward()
            self.optimizer.step() 
            if verbose and (epoch % every_n_epochs == 0):
                print('Epoch:\t{}\tLyapunov Risk: {:.4f}'.format(epoch, loss.item()))

            #### FALSIFIER ####
            # run falsifier every falsifier_frequency epochs
            if (self.falsifier is not None) and epoch % (self.falsifier.get_frequency())== 0:
                counterexamples = self.falsifier.check_lyapunov(X, V_candidate, L_V)
                if (not (counterexamples is None)): 
                    if verbose:
                        print("Not a Lyapunov function. Found {} counterexamples.".format(counterexamples.shape[0]))
                    # add new counterexamples sampled from old ones
                    if self.falsifier.counterexamples_added + len(counterexamples) > original_size:
                        if verbose:
                            print("Too many previous counterexamples. Pruning...")
                        # keep 1/5 of random elements of counterexamples
                        num_keep = original_size // 5
                        counterexample_keep_idx = torch.randperm(len(counterexamples))[:num_keep]
                        cur_counterexamples = X[counterexample_keep_idx]
                        X = X[:original_size]
                        X = torch.cat([X, cur_counterexamples], dim=0)
                        # reset counter
                        self.falsifier.counterexamples_added = num_keep

                    X = self.falsifier.add_counterexamples(X, counterexamples)
                    if verbose:
                        print('First counterexample: ')
                        print(counterexamples[0])
                else:  
                    if verbose:
                        print('No counterexamples found!')
                    no_counterexamples_ct += 1
                    # end training early if no counterexamples are found 2 separate times
                    if no_counterexamples_ct == 2:
                        break
        return loss_list