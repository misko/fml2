import torch
from torch import nn
from torchviz import make_dot, make_dot_from_trace

model = nn.Sequential()

#if its just conv, then there is no gradient?
conv = nn.Conv2d(1,1,1)
conv.weight.data.fill_(2)
conv.bias.data.fill_(0)
model.add_module('C0', conv)

#if its just linear, the equivalent of the conv? without bias? 
#model.add_module('W0', nn.Linear(1,1))

x = torch.randn(1,1,1,1).requires_grad_(True)

def double_backprop(inputs, net):
    y = net(x).mean()
    #print(inputs,y)
    #return y
    grad,  = torch.autograd.grad(y, x, create_graph=True, retain_graph=True)
    return grad.mean()

dot=make_dot(double_backprop(x, model), params=dict(list(model.named_parameters()) + [('x', x)]))

dot.render(view=True) 



 
