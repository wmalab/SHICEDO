import torch.nn as nn
import torch.nn.functional as F
import torch
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 128, 3, padding='same')
        self.norm_1=torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.99)
        self.se_1 = SELayer(128,reduction=256)
        
        self.conv2 = nn.Conv2d(128, 128, 3, padding='same')
        self.norm_2=torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.99)
        self.se_2 = SELayer(128,reduction=256)
        
        self.conv4 = nn.Conv2d(128, 256, 3, padding='same')
        self.norm_4=torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.99)
        self.se_4 = SELayer(256,reduction=256)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.t_conv1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.norm_t1=torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.99)
        self.t_se_1 = SELayer(128,reduction=256)

        self.t_conv3 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.norm_t3=torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.99)
        self.t_se_3 = SELayer(128,reduction=256)
        
        self.t_conv4 = nn.ConvTranspose2d(128, 1, 2, stride=2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.norm_1(x)
        x = self.se_1(x)
        x = F.relu(x)
        x1 = self.pool(x)

        x = self.conv2(x1)
        x = self.norm_2(x)
        x = self.se_2(x)
        x = F.relu(x)
        x2 = self.pool(x)  # compressed representation
        
        x = self.conv4(x2)
        x = self.norm_4(x)
        x = self.se_4(x)
        x = F.relu(x)
        x4 = self.pool(x)  # compressed representation
            
        ## decode ##
        # add transpose conv layers, with relu activation function      
        x = self.t_conv1(x4)
        x = self.norm_t1(x)
        x = self.t_se_1(x)
        tx3 = F.relu(x)
        
        tx3 = tx3 + x2
        x = self.t_conv3(tx3)
        x = self.norm_t3(x)
        x = self.t_se_3(x)
        tx4 = F.relu(x)
        
        tx4 = tx4 + x1
        x = self.t_conv4(tx4)
        x = F.relu(x)
        x = self.Normal_layer(x)
        return x
# initialize the NN
model = ConvAutoencoder()
