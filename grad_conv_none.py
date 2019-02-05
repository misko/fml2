from __future__ import print_function
from torch.autograd import Variable
import torch

from torchviz import make_dot, make_dot_from_trace

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear=torch.nn.Linear(1,1)
        self.linear.weight.data.fill_(1)
        self.linear.bias.data.fill_(1)
        self.sq = torch.nn.Sequential()
        self.sq.add_module("linear",self.linear)

    def forward(self,xo):
        x=self.sq(xo.reshape(1,1)**2)
        print(self.sq.named_parameters())
        dot=make_dot(x, params=dict(self.sq.named_parameters()))
        print("WTF")
        print(dot)
        dot.render(view=True)
        return x

model = torch.nn.Sequential()
linear=torch.nn.Linear(1,1)
model.add_module("linear",linear)

#model=Model()

x=Variable(torch.tensor([[1.0]]),requires_grad=True)
#model(x)
#loss=model(x.reshape(1,1)**2)
loss=model(x)
print(loss)
dot=make_dot(model(x), params=dict(model.named_parameters()))
#dot=make_dot(loss, params=dict(model.sq.named_parameters()))
#dot.render(view=True)
#for name,parameter in model.named_parameters():
#    print("GRAD",name,parameter.size(),parameter.grad)

