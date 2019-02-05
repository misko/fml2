import torch               
                                                                      
x = torch.randn(1,1,1,1).requires_grad_(True)                         

c=torch.nn.Conv2d(1,1,1)
print(c.weight.size(),c.bias.size())
c.weight.retain_grad()
y = c(x).mean()**2
y.retain_grad()
conv_grad,  = torch.autograd.grad(y, x, create_graph=True, retain_graph=True)   
print("Conv grad",conv_grad)

l = torch.nn.Linear(1,1)
print(l.weight.size(),l.bias.size())
y = torch.nn.Linear(1,1)(x).mean()**2
linear_grad,  = torch.autograd.grad(y, x, create_graph=True, retain_graph=True)  
print("Linear grad",linear_grad)

