import torch
from torchvision.models.resnet import ResNet, BasicBlock
from torch import nn, optim
import torch.nn as nn
import torch.nn.functional as F

# def preprocess(x):
#     return x.view(-1, 1, 28, 28)

# def conv(in_size, out_size, pad=1): 
#     return nn.Conv2d(in_size, out_size, kernel_size=3, stride=2, padding=pad)

# class ResBlock(nn.Module):
    
#     def __init__(self, in_size:int, hidden_size:int, out_size:int, pad:int):
#         super().__init__()
#         self.conv1 = conv(in_size, hidden_size, pad)
#         self.conv2 = conv(hidden_size, out_size, pad)
#         self.batchnorm1 = nn.BatchNorm2d(hidden_size)
#         self.batchnorm2 = nn.BatchNorm2d(out_size)
    
#     def convblock(self, x):
#         x = F.relu(self.batchnorm1(self.conv1(x)))
#         x = F.relu(self.batchnorm2(self.conv2(x)))
#         return x
    
#     def forward(self, x): return x + self.convblock(x) # skip connection
    
# class MnistResNet(nn.Module):
    
#     def __init__(self, n_classes=10):
#         super().__init__()
#         self.res1 = ResBlock(1, 8, 16, 15)
#         self.res2 = ResBlock(16, 32, 16, 15)
#         self.conv = conv(16, n_classes)
#         self.batchnorm = nn.BatchNorm2d(n_classes)
#         self.maxpool = nn.AdaptiveMaxPool2d(1)
        
#     def forward(self, x):
#         x = preprocess(x)
#         x = self.res1(x)
#         x = self.res2(x) 
#         x = self.maxpool(self.batchnorm(self.conv(x)))
#         return x.view(x.size(0), -1)
# import torch
# from torchvision.models.resnet import ResNet, BasicBlock
# class MnistResNet(ResNet):
#     def __init__(self):
#         super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
#         self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
#     def forward(self, x):
#         return torch.softmax(super(MnistResNet, self).forward(x), dim=-1)

class Residual(nn.Module):
    def __init__(self,in_channel,num_channel,use_conv1x1=False,strides=1):
        super(Residual,self).__init__()
        self.relu=nn.ReLU()
        self.bn1=nn.BatchNorm2d(in_channel,eps=1e-3)
        self.conv1=nn.Conv2d(in_channels =in_channel,out_channels=num_channel,kernel_size=3,padding=1,stride=strides)
        self.bn2=nn.BatchNorm2d(num_channel,eps=1e-3)
        self.conv2=nn.Conv2d(in_channels=num_channel,out_channels=num_channel,kernel_size=3,padding=1)
        if use_conv1x1:
            self.conv3=nn.Conv2d(in_channels=in_channel,out_channels=num_channel,kernel_size=1,stride=strides)
        else:
            self.conv3=None
 
 
    def forward(self, x):
        y=self.conv1(self.relu(self.bn1(x)))
        y=self.conv2(self.relu(self.bn2(y)))
        # print (y.shape)
        if self.conv3:
            x=self.conv3(x)
        # print (x.shape)
        z=y+x
        return z

# blk = Residual(3,3,True)
# X = Variable(torch.zeros(4, 3, 96, 96))
# out=blk(X)
 
def ResNet_block(in_channels,num_channels,num_residuals,first_block=False):
    layers=[]
    for i in range(num_residuals):
        if i==0 and not first_block:
            layers+=[Residual(in_channels,num_channels,use_conv1x1=True,strides=2)]
        elif i>0 and not first_block:
            layers+=[Residual(num_channels,num_channels)]
        else:
            layers += [Residual(in_channels, num_channels)]
    blk=nn.Sequential(*layers)
    return blk

class MnistResNet(nn.Module):
    def __init__(self,in_channel=1,num_classes=10):
        super(MnistResNet,self).__init__()
        self.block1=nn.Sequential(nn.Conv2d(in_channels=in_channel,out_channels=64,kernel_size=7,stride=2,padding=3),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
        self.block2=nn.Sequential(ResNet_block(64,64,2,True),
                                  ResNet_block(64,128,2),
                                  ResNet_block(128,256,2),
                                  ResNet_block(256,512,2))
        self.block3=nn.Sequential(nn.AvgPool2d(kernel_size=3))
        self.Dense=nn.Linear(512,10)
 
 
    def forward(self,x):
        y=self.block1(x)
        y=self.block2(y)
        y=self.block3(y)
        y=y.view(-1,512)
        y=self.Dense(y)
        return y
 
