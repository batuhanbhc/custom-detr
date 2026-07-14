import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        ):
        super().__init__()

        dims = [input_dim] + [hidden_dim] * (num_layers-1) + [output_dim]
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i+1]) for i in range(num_layers)]
        )
        self.act = nn.SiLU()
    
    def forward(self, x):
        for i in range(self.num_layers-1):
            x = self.act(self.layers[i](x))
        return self.layers[-1](x)