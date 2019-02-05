import torch                                                          
from torch import nn                                                  
from torchviz import make_dot, make_dot_from_trace                    
                                                                      
model = nn.Sequential()                                               

#if its only one conv module in the model
c=nn.Conv2d(1,1,1)
#model.add_module('C0', c)  #either uncomment this line                   
#print("WTF",c.weight)
#c.weight.register_hook(print)


#if its just one linear module in the model
l=nn.Linear(1,1)
model.add_module('W0', l)     #or uncomment this line
l.weight.register_hook(print)
                                                                      
x = torch.randn(1,1,1,1).requires_grad_(True)                         
                                                                      
def double_backprop(inputs, net):                                     
    y = net(x).mean() #**2                                                
    grad,  = torch.autograd.grad(y, x, create_graph=True, retain_graph=True)   
    print("X",x,"G",grad)
    return grad.mean()                                                
    
#dot=make_dot(double_backprop(x, model), params=dict(list(model.named_parameters()) + [('x', x)]))
dot=make_dot(double_backprop(x, model), params=dict(list(model.named_parameters())))

dot.render(view=True) 
