from torch.nn import Module
import torch


class SDPAttention(Module):
    def __init__(self, mp):
        super().__init__()

        self.d_model = mp.d_model
        self.d_v = mp.d_v
        self.d_k = mp.d_k

        self.softmax = torch.nn.Softmax()

    def forward(self, Q, K, V):

        assert Q.shape[0] == self.d_k
        assert K.shape[0] == self.d_k
        assert V.shape[0] == self.d_v

        x = torch.matmul(Q, K.T)
        x = self.scale(x)
        # no mask
        x = self.softmax(x)
        x = torch.matmul(x, V)

        assert x.shape[0] == self.d_k

        return x

    def scale(self, mat):
        return mat / torch.sqrt(self.d_k)


class MultiHeadAttention(Module):
    def __init__(self, mp):
        super().__init__()

        d_k = 0
        d_v = 0
        d_model = 0

        self.d_model = mp.d_model
        self.d_v = mp.d_v
        self.d_k = mp.d_k
        self.h = mp.h

        self.w_Q = [torch.nn.Linear(d_model, d_k) for i in range(h)]
        self.w_K = [torch.nn.Linear(d_model, d_k) for i in range(h)]
        self.w_V = [torch.nn.Linear(d_model, d_v) for i in range(h)]
        self.w_O = torch.nn.Linear(d_model, d_k * h)

        self.SDPA = SDPAttention(mp)

    def forward(self, Q, K, V):

        assert Q.shape[0] == self.d_model
        assert K.shape[0] == self.d_model
        assert V.shape[0] == self.d_model

        heads = []

        for i in range(h):

            q_i = self.w_Q[i](Q)
            k_i = self.w_K[i](K)
            v_i = self.w_v[i](V)

            head = self.SDPA(q_i, k_i, v_i)

            heads.append(head)

        x = self.w_O(torch.concat(heads, 0))

        assert x.shape[0] == self.d_model

        return x
