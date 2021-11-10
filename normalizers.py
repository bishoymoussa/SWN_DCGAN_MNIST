import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def forward(self, w, power_iterations=1):
        w = w.data.shape[0]
        height = w.data.shape[0]
        for _ in range(power_iterations):
            v = l2normalize(np.matmul(np.transpose(w.view(height,-1).data), u.data))
            u = l2normalize(np.matmul(w.view(height,-1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        return w / sigma.expand_as(w)

