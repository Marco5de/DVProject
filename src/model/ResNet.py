import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.GroupNorm import GroupNorm

class BasicResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, projection_shortcut=False, group_norm=False, groups=16):
        super(BasicResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=(2 if projection_shortcut else 1),
                               padding=1)
        self.norm1 = GroupNorm(out_channels, groups=groups) if group_norm else nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = GroupNorm(out_channels, last_residual=True, groups=groups) if group_norm else nn.BatchNorm2d(out_channels)

        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2,
                                   padding=0) if projection_shortcut else nn.Identity()

    def forward(self, t: torch.Tensor):
        residual_skip = t

        t = F.relu(self.norm1(self.conv1(t)))
        t = self.norm2(self.conv2(t))

        residual_skip = self.skip_conv(residual_skip)

        return F.relu(t + residual_skip)


class ResNet34(nn.Module):
    def __init__(self, input_size=(32, 32), num_classes=10, group_norm=False, input_channels=3, groups=16):
        super(ResNet34, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.norm1 = GroupNorm(64, groups=groups) if group_norm else nn.BatchNorm2d(64)

        # Stacking BasicResidualBlocks
        self.layer01 = nn.Sequential(
            BasicResidualBlock(64, 64, group_norm=group_norm, groups=groups),
            BasicResidualBlock(64, 64, group_norm=group_norm, groups=groups),
            BasicResidualBlock(64, 64, group_norm=group_norm, groups=groups)
        )

        self.layer02 = nn.Sequential(
            BasicResidualBlock(64, 128, group_norm=group_norm, projection_shortcut=True, groups=groups),
            BasicResidualBlock(128, 128, group_norm=group_norm, groups=groups),
            BasicResidualBlock(128, 128, group_norm=group_norm, groups=groups),
            BasicResidualBlock(128, 128, group_norm=group_norm, groups=groups)
        )

        self.layer03 = nn.Sequential(
            BasicResidualBlock(128, 256, group_norm=group_norm, projection_shortcut=True, groups=groups),
            BasicResidualBlock(256, 256, group_norm=group_norm, groups=groups),
            BasicResidualBlock(256, 256, group_norm=group_norm, groups=groups),
            BasicResidualBlock(256, 256, group_norm=group_norm, groups=groups),
            BasicResidualBlock(256, 256, group_norm=group_norm, groups=groups),
            BasicResidualBlock(256, 256, group_norm=group_norm, groups=groups)
        )

        self.layer04 = nn.Sequential(
            BasicResidualBlock(256, 512, group_norm=group_norm, projection_shortcut=True, groups=groups),
            BasicResidualBlock(512, 512, group_norm=group_norm, groups=groups),
            BasicResidualBlock(512, 512, group_norm=group_norm, groups=groups)
        )

        self.pool2 = nn.AvgPool2d(kernel_size=4)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, t: torch.Tensor):
        t = F.relu(self.norm1(self.conv1(t)))

        t = self.layer01(t)
        t = self.layer02(t)
        t = self.layer03(t)
        t = self.layer04(t)

        t = self.pool2(t)
        t = t.reshape([t.shape[0], t.shape[1]])
        t = self.fc(t)

        return t