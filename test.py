import torch
from torch import Tensor
from torch.autograd import Variable
from torch.autograd import grad
from torch import nn

# some toy data
x = Variable(Tensor([4., 2.]), requires_grad=True)
y = Variable(Tensor([1.]), requires_grad=False)

# linear model and squared difference loss
model = nn.Linear(2, 1)
#loss = torch.sum(model(x)**2) # (y - model(x))**2)
loss = torch.sum(model(x)) # (y - model(x))**2)
print(loss)

# instead of using loss.backward(), use torch.autograd.grad() to compute gradients
loss_grads = grad(loss, model.parameters(), create_graph=True,allow_unused=True,retain_graph=True)
print(loss_grads)

# compute the squared norm of the loss gradient
gn2 = loss_grads[0].sum() #sum([grd.norm()**2 for grd in loss_grads])
print(gn2)
model.zero_grad()
gn2.backward()
