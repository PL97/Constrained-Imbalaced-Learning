import torch
import torch.nn as nn
import numpy as np

class shared(nn.Module):
    def __init__(self):
        super(shared, self).__init__()
    
    def init_clf(self, input_shape, num_classes, mlp=False,  dropout=0):
        ## get output features
        with torch.no_grad():
            test_input = torch.randn(input_shape)
            test_output = self.features(test_input)
            output_flatten_size = np.prod(test_output.shape)

        ## init clf
        # classifier
        self.fc_layers = nn.Sequential(
            # nn.Dropout(0.8),
            nn.Linear(output_flatten_size, 2048),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.8),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        ) if mlp else \
            nn.Linear(output_flatten_size, num_classes, bias=False)
        # nn.utils.weight_norm(nn.Linear(output_flatten_size, num_classes, bias=False))
        
        self.dropout = nn.Dropout(dropout) if (dropout > 0 and dropout < 1) else nn.Identity()
        
    def forward(self, x):
        conv_features = self.features(x)
        flatten = conv_features.view(conv_features.size(0), -1)
        flatten = self.dropout(flatten)
        fc = self.fc_layers(flatten)
        return fc
