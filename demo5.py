# -*- coding: utf-8 -*-
# @Time     :2019/11/14 19:35
# @Author   :XiaoMa
# @File     :demo5.py
import torch.nn as nn
import torch.nn.functional as F
import torch


class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 3)
        self.fc1 = nn.Linear(1350, 10)

    # 正向传播
    def forward(self, x):
        print('01:',x.size())  # 结果：[1, 1, 32, 32]
        # 卷积 -> 激活 -> 池化
        x = self.conv1(x)
        x = F.relu(x)
        print('02:',x.size())  # 结果：[1, 6, 30, 30]
        x = F.max_pool2d(x, (2, 2))  # 我们使用池化层，计算结果是15
        x = F.relu(x)
        print('03:',x.size())  # 结果：[1, 6, 15, 15]
        # reshape，‘-1’表示自适应
        # 这里做的就是压扁的操作 就是把后面的[1, 6, 15, 15]压扁，变为 [1, 1350]
        x = x.view(x.size()[0], -1)
        print('04:',x.size())
        x = self.fc1(x)
        return x


net=NNet()
input=torch.randn(1,1,32,32)    #这是一个1x1x32x32的4维张量，元素遵循正太分布
# print(input)
out=net(input)
print(out.size())
for name,parameters in net.named_parameters():
    print(name,':',parameters)

# net.zero_grad()
# out.backward(torch.ones(1,10))
#
y=torch.arange(0,10).view(1,10).float()
criterion=nn.MSELoss()
loss=criterion(out,y)
# print(loss.item())
optimizer=torch.optim.SGD(net.parameters(),lr=0.01)
optimizer.zero_grad()   #效果与net.zero_grad()一样
loss.backward()

optimizer.step()





