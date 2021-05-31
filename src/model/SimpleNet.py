import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.GroupNorm import GroupNorm


class SimpleNet(nn.Module):

    def __init__(self, input_size=(32, 32), num_classes=10, group_norm=False, input_channels=3):
        super(SimpleNet, self).__init__()

        self.cnet = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 1, 1),
            GroupNorm(64, groups=16) if group_norm else nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 256, 3, 1, 1),
            GroupNorm(256, groups=16) if group_norm else nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 512, 3, 1, 1),
            GroupNorm(512, groups=16) if group_norm else nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Flatten(),
            nn.Linear(512 * input_size[0] * input_size[1] // 16, 512),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, t: torch.Tensor):
        return self.cnet(t)



class SimpleNetv2(nn.Module):
    # todo dominik hat norm nach relu
    num_channels = [32, 64, 128, 1024]


    def __init__(self, input_size=(32, 32), num_classes=10, group_norm=False, input_channels=3, groups=16):
        super(SimpleNetv2, self).__init__()

        self.cnet = nn.Sequential(
            nn.Conv2d(input_channels, self.num_channels[0], 3, 1, 1),
            GroupNorm(self.num_channels[0], groups=groups) if group_norm else nn.BatchNorm2d(self.num_channels[0]),
            nn.ReLU(),
            nn.Conv2d(self.num_channels[0], self.num_channels[0], 3, 1, 1),
            GroupNorm(self.num_channels[0], groups=groups) if group_norm else nn.BatchNorm2d(self.num_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(self.num_channels[0], self.num_channels[1], 3, 1, 1),
            GroupNorm(self.num_channels[1], groups=groups) if group_norm else nn.BatchNorm2d(self.num_channels[1]),
            nn.ReLU(),
            nn.Conv2d(self.num_channels[1], self.num_channels[1], 3, 1, 1),
            GroupNorm(self.num_channels[1], groups=groups) if group_norm else nn.BatchNorm2d(self.num_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(self.num_channels[1], self.num_channels[2], 3, 1, 1),
            GroupNorm(self.num_channels[2], groups=groups) if group_norm else nn.BatchNorm2d(self.num_channels[2]),
            nn.ReLU(),
            nn.Conv2d(self.num_channels[2], self.num_channels[2], 3, 1, 1),
            GroupNorm(self.num_channels[2], groups=groups) if group_norm else nn.BatchNorm2d(self.num_channels[2]),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(self.num_channels[2] * input_size[0] * input_size[1] // 16, self.num_channels[3]),
            nn.Dropout(p=0.2),
            nn.Linear(self.num_channels[3], num_classes)
        )

    def forward(self, t: torch.Tensor):
        return self.cnet(t)



def __main__():
    t = torch.randn((8, 1, 28, 28))
    model = SimpleNet(input_size=(28, 28), input_channels=1)
    out = model(t)
    print(out.shape)

    model = SimpleNetv2(input_size=(28, 28), input_channels=1)
    out = model(t)
    print(out.shape)

    from src.util.DataLoader import count_parameters
    print(count_parameters(model))



if __name__ == "__main__":
    __main__()
