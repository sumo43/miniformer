from torch.nn import Module
from utils.general import VERY_SMOL_NUM
import torch


class SDPAttention(Module):
    def __init__(self, mp, masked=False):
        super().__init__()

        self.masked = masked
        self.d_model = mp.d_model
        self.d_v = mp.d_v
        self.d_k = mp.d_k

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, Q, K, V):

        assert Q.shape[-1] == self.d_k
        assert K.shape[-1] == self.d_k
        assert V.shape[-1] == self.d_v

        x = torch.matmul(Q, K.mT)
        x = self.scale(x)

        if self.masked == True:
            x = self.mask(x)

        x = self.mask(x)
        # no mask
        x = self.softmax(x)

        x = torch.matmul(x, V)

        assert x.shape[-1] == self.d_k

        return x

    def scale(self, mat):
        return mat / torch.sqrt(torch.tensor(self.d_k))
    
    def mask(self, x):
        tri_mask = torch.triu(torch.ones(size=x.shape, dtype=torch.float), 1) * VERY_SMOL_NUM
        x = x + tri_mask
        return x 


class MultiHeadAttention(Module):
    def __init__(self, mp, masked=False):

        super().__init__()

        d_k = 0
        d_v = 0
        d_model = 0

        self.d_model = mp.d_model
        self.d_v = mp.d_v
        self.d_k = mp.d_k
        self.h = mp.h

        # TODO make these each 2 weight
        self.w_Q = [torch.nn.Linear(self.d_model, self.d_k) for i in range(self.h)]
        self.w_K = [torch.nn.Linear(self.d_model, self.d_k) for i in range(self.h)]
        self.w_V = [torch.nn.Linear(self.d_model, self.d_v) for i in range(self.h)]
        self.w_O = torch.nn.Linear(self.d_model, self.d_k * self.h)

        self.SDPA = SDPAttention(mp, masked)

    def forward(self, V, K, Q):

        assert Q.shape[-1] == self.d_model
        assert K.shape[-1] == self.d_model
        assert V.shape[-1] == self.d_model

        heads = []

        for i in range(self.h):

            q_i = self.w_Q[i](Q)
            k_i = self.w_K[i](K)
            v_i = self.w_V[i](V)

            head = self.SDPA(q_i, k_i, v_i)

            heads.append(head)

        x = self.w_O(torch.concat(heads, -1))

        assert x.shape[-1] == self.d_model

        return x
