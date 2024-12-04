import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Initial layers
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # First block
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        
        # Second block
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        
        # Final layers
        self.conv6 = nn.Conv2d(32, 10, 1)  # 1x1 convolution
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.15)

    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        
        # First residual block
        identity1 = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = F.relu(x + identity1)  # Skip connection
        
        # Second block (no residual due to channel change)
        x = self.pool(x)
        x = self.dropout2(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Final layers
        x = self.conv6(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)
