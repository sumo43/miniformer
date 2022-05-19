from torch.nn import Module
import torch


class SDPAttention(Module):
    def __init__(self):
        super().__init__()

        d_k = 0
        d_v = 0
        d_model = 0

        self.matmul1 = torch.nn.Linear(n_d, ...)
        self.matmul2 = torch.nn.Linear(n_d, ...)

    def scale(self, mat):
        return mat /

    def forward(self, Q, K, V):


class MultiHeadAttention(Module):
