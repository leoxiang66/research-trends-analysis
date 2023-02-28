import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self,input_dim:int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layerNorm = nn.LayerNorm(input_dim)
        self.down_proj = nn.Linear(input_dim,hidden_dim,False)
        self.up_proj = nn.Linear(hidden_dim,input_dim,False)

    def forward(self,x):
        '''

        :param x: N,L,D
        :return:  N,L,D
        '''
        output = x
        x = self.layerNorm(x)
        x = self.down_proj(x)
        x = nn.functional.relu(x)
        x = self.up_proj(x)
        output = output + x # residual connection
        return output
