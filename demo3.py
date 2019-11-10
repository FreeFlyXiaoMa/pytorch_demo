# -*- coding: utf-8 -*-
# @Time     :2019/11/10 19:49
# @Author   :XiaoMa
# @File     :demo3.py

import torch
x=torch.randn(1)

if torch.cuda.is_available():
    device=torch.device('cuda')
    y=torch.ones_like(x,device=device)  #将张量x转为张量y--复用x的属性
    x=x.to(device)

    z=x+y
    print(z)
    print(z.to('cpu',torch.double))
