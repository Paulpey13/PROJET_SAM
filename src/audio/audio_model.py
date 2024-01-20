import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 6)) # pooling adaptatif pour obtenir une taille fixe

        self.fc1 = nn.Linear(32 * 1 * 6, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.adaptive_pool(x)

        x = x.view(-1, 32 * 1 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

