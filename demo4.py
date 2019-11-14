# -*- coding: utf-8 -*-
# @Time     :2019/11/14 10:00
# @Author   :XiaoMa
# @File     :demo4.py
import torch
#
# x=torch.rand(5,5,requires_grad=True)
# y=torch.rand(5,5,requires_grad=True)
# z=x**2+y**3
# z.backward(torch.ones_like(x))  #复用张量x的属性
# print(x)
# print(x.grad)
# print(x.grad/x)

from torch.autograd.function import Function

class Separate(Function):
    @staticmethod
    def forward(ctx,tensor,constant):
        ctx.constant=constant
        return tensor*constant
    @staticmethod
    def backward(ctx,grad_output):
        return grad_output,None

x=torch.rand(2,2,requires_grad=True)
y=Separate.apply(x,5)
y.backward(torch.ones_like(x))
print(x.grad)