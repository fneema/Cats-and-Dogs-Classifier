import torch.nn as nn 
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, input_size, n_feature, output_size):
        super(ConvNet, self).__init__()
        self.n_feature = n_feature
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_feature, kernel_size=5)
        self.conv2 = nn.Conv2d(n_feature, 2*n_feature, kernel_size=5)
        self.conv3 = nn.Conv2d(2*n_feature, 2**2*n_feature, kernel_size=5)
        self.conv4 = nn.Conv2d(2**2*n_feature, n_feature, kernel_size=5)
        self.fc1 = nn.Linear(n_feature*10*10, 50)
        self.fc2 = nn.Linear(50, 2)
        
    def forward(self, x, verbose=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, self.n_feature*10*10)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


