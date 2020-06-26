import torch.nn as nn
import torch


class Flatten(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.view(x.size(0), -1)


class Lenet(nn.Module):

    def __init__(self):
        super().__init__()

        # data info
        self.num_channels = 1
        self.image_size = (28, 28)
        self.num_classes = 10

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(1024, 200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)

        return x
