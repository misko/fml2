from __future__ import print_function
from torch.autograd import Variable
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv=torch.nn.Conv2d(1,1,1)
        self.conv.weight.data.fill_(1)
        self.conv.bias.data.fill_(0)
        self.linear=torch.nn.Linear(1,1)
        self.linear.weight.data.fill_(1)
        self.linear.bias.data.fill_(0)

    def forward(self,xo):
        x=self.conv(xo.reshape(1,1,1,1)**2).sum().reshape(1,1) # required otherwise second deriviative is a scalar not depending on other variable
        x+=self.linear(xo.reshape(1,1)**2)
        l=torch.autograd.grad(outputs=x,inputs=xo,create_graph=True)[0]
        y=(l**2).sum()
        model.zero_grad()

        #use grad of backward
        #force=torch.autograd.grad(outputs=y,inputs=xo,create_graph=True,allow_unused=True)[0]
        y.backward()

model=Model()
x=Variable(torch.tensor([[1.0]]),requires_grad=True)
loss=model(x)
for name,parameter in model.named_parameters():
    print("GRAD",name,parameter.size(),parameter.grad)
