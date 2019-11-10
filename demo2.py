# -*- coding: utf-8 -*-
# @Time     :2019/10/25 19:52
# @Author   :XiaoMa
# @File     :demo2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
#定义卷积神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1,6,3)
        self.fc1=nn.Linear(1350,10)
    def forward(self, x):
        print('x.size:',x.size())
        x=self.conv1(x)
        x=F.relu(x)
        print('x.size:',x.size())
        x=F.max_pool2d(x,(2,2))
        x=F.relu(x)
        print('x.size:',x.size())
        x=x.view(x.size()[0],-1)
        print('x.size:',x.size())
        x=self.fc1(x)
        return x
net=Net()
# for parameters in net.parameters():
#     print('------',parameters)

# for name,parameters in net.named_parameters():
#     print(name,'-',parameters.size())

input=torch.randn(1,1,32,32)
out=net(input)
# print(out.size())
# net.zero_grad()
# out.backward(torch.ones(1,10))

y=torch.arange(0,10).view(1,10).float()
criterion=nn.MSELoss()
loss=criterion(out,y)
# print(loss.item())
import torch.optim
optimizer=torch.optim.SGD(net.parameters(),lr=0.01)
optimizer.zero_grad()
loss.backward()

optimizer.step()
