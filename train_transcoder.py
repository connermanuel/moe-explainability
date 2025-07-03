import torch
import torch.nn as nn


class TopKActivation(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, indices = torch.topk(x, self.k, dim=-1)
        output = torch.zeros_like(x)
        output[indices] = values

        return output


class Transcoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, k: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.k = k

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.topk = TopKActivation(k)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.topk(x)
        x = self.linear2(x)
        return x
