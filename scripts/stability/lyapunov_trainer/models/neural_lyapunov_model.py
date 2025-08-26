import torch

class NeuralLyapunovModel(torch.nn.Module):
    
    def __init__(self, n_input, n_hidden):
        super(NeuralLyapunovModel, self).__init__()

        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, 1)
        self.activation = torch.nn.Tanh()

    def forward(self, x):
        h_1 = self.activation(self.layer1(x))
        # V is potential lyapunov function
        V = self.activation(self.layer2(h_1))
        return V

if __name__ == '__main__':
    # test out function

    # number of samples
    N = 1
    # inputs 
    D_in = 4
    H1 = 6

    x = torch.Tensor(N, D_in).uniform_(-6, 6)       

    controller = NeuralLyapunovModel(D_in, H1)
    V = controller(x)

    print(V)
