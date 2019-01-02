from __future__ import print_function
from torch.autograd import Variable
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv=nn.Conv2d(1,1,2)

    def forward(self,x):
        x=2*x
        tmp=self.conv(x).sum()

        #two options
        tmp.backward()
        #torch.autograd.grad(tmp,x,create_graph=True)

model=Model()
x=Variable(torch.Tensor([[1,2],[3,4]]),requires_grad=True).reshape(1,1,2,2)
loss=model(x)
for parameter in model.parameters():
    print("GRAD",parameter.size(),parameter.grad)
