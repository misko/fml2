from __future__ import print_function
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from fml2 import Dist3


d=Dist3(3)
point=Variable(torch.Tensor([[0,1,0]]),requires_grad=True)
output=d.forward(point).sum()
d_output_to_point = torch.autograd.grad(output,point,create_graph=True)[0]
print(d_output_to_point)
