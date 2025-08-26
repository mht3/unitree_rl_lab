import torch

class Falsifier():
    def __init__(self, lower_bound, upper_bound, epsilon=0., scale=0.02, frequency=200, num_samples=10, env=None):
        self.epsilon = epsilon
        self.counterexamples_added = 0
        self.lower_bound = torch.Tensor(lower_bound)
        self.upper_bound = torch.Tensor(upper_bound)
        self.scale = scale
        self.frequency = frequency
        self.num_samples = num_samples
        # AI Gym environment. Used to get min and max state values and optionally get next_state
        self.env = env


    def get_frequency(self):
        return self.frequency

    def set_frequency(self, freq):
        self.frequency = freq

    @torch.no_grad
    def check_lyapunov(self, X, V_candidate, L_V):    
        '''
        Checks if the lyapunov conditions are violated for any sample. 
        Data points that are unsatisfiable will be sampled and added 
        '''
        N = X.shape[0]

        # Ensure lyapunov function and lie derivative are 1D tensors
        if V_candidate.dim() != 1:
            V_candidate = V_candidate.reshape(N)
        if L_V.dim() != 1:
            L_V = L_V.reshape(N)
        
        lyapunov_mask = (V_candidate < 0.)
        lie_mask = (L_V > self.epsilon)

        # bitwise or for falsification conditions
        union = lyapunov_mask.logical_or(lie_mask)

        # get batch indices that violate the lyapunov conditions
        indices = torch.nonzero(union).squeeze()
        # check num elements > 0
        if indices.numel() > 0:
            return  X[indices].reshape(indices.numel(), self.lower_bound.shape[0])
        else:
            return None
    
    @torch.no_grad
    def add_counterexamples(self, X, counterexamples):
        '''
        Take counter examples and sample points around them.
        X: current training data
        counterexamples: all new examples from training data that don't satisfy the lyapunov conditions
        '''        
        self.counterexamples_added += len(counterexamples) * self.num_samples
        for i in range(counterexamples.shape[0]):
            samples = torch.empty(self.num_samples, 0)
            counterexample = counterexamples[i]
            for j in range(self.upper_bound.shape[0]):
                lb = self.lower_bound[j]
                ub = self.upper_bound[j]
                value = counterexample[j]
                # Determine the min and max values for each feature in the chosen counterexamples
                min_value = torch.max(lb, value - self.scale * abs(value))
                max_value = torch.min(ub, value + self.scale * abs(value))
                
                sample = torch.Tensor(self.num_samples, 1).uniform_(min_value, max_value)
                samples = torch.cat([samples, sample], dim=1)
            X = torch.cat((X, samples), dim=0)
        return X
    
if __name__ == '__main__':
    # small test for falsifier
    num_samples=2
    falsifier = Falsifier(lower_bound=[-1., -0.5], upper_bound=[1., 0.5], num_samples=num_samples)
    x = torch.Tensor(5, 1).uniform_(-1., 1.)
    x = torch.cat((x, torch.Tensor(5, 1).uniform_(-0.5, 0.5)), dim=1)

    # Fake lyapunov funcions
    V = torch.ones(size=(5,1))
    V[3, 0] = -0.5
    V[2, 0] = -0.1

    L_V = -torch.ones(size=(5,1))
    L_V[1, 0] = 2.5

    # test that we find 3 unique counterexamples and add num_samples*3 to dataset
    counterexamples = falsifier.check_lyapunov(x, V, L_V)
    assert(counterexamples.size(0) == 3)
    x_new = falsifier.add_counterexamples(x, counterexamples)
    assert(x_new.size(0) == num_samples*counterexamples.size(0) + x.size(0))
    print(x)
    print(x_new)