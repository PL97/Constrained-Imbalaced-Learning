import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=30, num_layers=3):
        super().__init__()
        bias = False
        layers = [nn.Linear(input_dim, hidden_dim, bias=bias)]
        for _ in range(num_layers-2):
            layers.append(nn.BatchNorm1d(num_features = hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        layers.append(nn.Linear(hidden_dim, 1, bias=bias))
        layers.append(nn.Sigmoid())
        
        # layers = [nn.Linear(input_dim, 1, bias=bias)]
        # layers.append(nn.Sigmoid())
        
        
        self.net = nn.Sequential(*layers)
    def forward(self, X):
        return self.net(X)

# def MLP(input_dim, hidden_dim=30, num_layers=3):
#     net =  nn.Sequential(nn.Linear(input_dim, 1, bias=True),
#                         nn.Sigmoid()
#                         )
    
#     for name, param in net.named_parameters():
#         print(name, param.size())
    
#     return net