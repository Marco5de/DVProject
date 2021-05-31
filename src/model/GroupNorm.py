import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm(nn.Module):
    """
    Implementation of the GroupNorm layer - using the implementation provided in the original paper
    """

    def __init__(self, channels, groups=16, eps=1e-5, last_residual=False):
        super(GroupNorm, self).__init__()

        self.gamma = nn.Parameter(torch.zeros([1, channels, 1, 1])) if last_residual else nn.Parameter(
            torch.ones([1, channels, 1, 1]))
        self.beta = nn.Parameter(torch.zeros([1, channels, 1, 1]))
        self.num_groups = groups
        self.eps = eps

    def forward(self, t: torch.Tensor):
        bs, C, H, W = t.shape
        t = t.view(bs * self.num_groups, -1)

        mean = t.mean(dim=1, keepdim=True)
        var = t.var(dim=1, keepdim=True)
        t = (t - mean) / torch.sqrt(var + self.eps)

        t = t.view([bs, C, H, W])

        return self.gamma * t + self.beta