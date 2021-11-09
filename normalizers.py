from matplotlib.colors import hsv_to_rgb
import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter
import numba # To Accelerate the L2Normalizer Loop

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)
    
@numba.jit(nopython=True, parallel=True)
def u_v_l2normalizer(u, w, height, power_iter):
    for _ in range(power_iter):
        mat_v_u = np.matmul(np.transpose(w.view(height,-1).data), u) # torch.t(w.view(height,-1).data).dot(u)
        v = mat_v_u / (mat_v_u.norm() + 1e-12)
        mat_u_v = np.matmul(w.view(height,-1).data, v) # w.view(height,-1).data.dot(v)
        u = mat_u_v / (mat_u_v.norm() + 1e-12)
    return u, v

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations

    def _update_u_v(self):
        if not self._made_params():
            self._make_params()
        w = getattr(self.module, self.name)
        u = getattr(self.module, self.name + "_u")

        height = w.data.shape[0]
        u, v = u_v_l2normalizer(u, w, height, self.power_iterations)

        setattr(self.module, self.name + "_u", u)
        w.data = w.data / u.dot(np.matmul(w.view(height,-1).data, v))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = l2normalize(w.data.new(height).normal_(0, 1))

        self.module.register_buffer(self.name + "_u", u)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)    

