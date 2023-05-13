import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=30, num_layers=3, output_dim=1):
        super().__init__()
        bias = False
        layers = [nn.Linear(input_dim, hidden_dim, bias=bias)]
        for _ in range(num_layers-2):
            layers.append(nn.BatchNorm1d(num_features = hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        self.features = nn.Sequential(*layers)

        if output_dim == 1:
            self.fc_layers = nn.Sequential(
                        nn.Linear(hidden_dim, output_dim, bias=bias),
                        nn.Sigmoid())
        else:
            self.fc_layers = nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, X):
        x = self.features(X)
        return self.fc_layers(x)