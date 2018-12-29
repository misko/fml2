import torch

def make_grid(gz=5):
    t=torch.linspace(-(gz/2),gz/2,gz)
    a=t.reshape(gz,1,1,1).expand(gz,gz,gz,1)
    b=t.reshape(1,gz,1,1).expand_as(a)
    c=t.reshape(1,1,gz,1).expand_as(a)
    return torch.cat((a,b,c),dim=3)

def grid_to_points(g):
    gz=g.size()[0]
    return g.reshape(gz*gz*gz,3)

def points_to_grid(ps,n=1):
    gz=int(torch.pow(ps.size()[0]/n,torch.tensor([1.0/3.0])))
    return ps.resize(gz,gz,gz,n)

def make_grid_points(gz=5):
    g=make_grid(gz)
    return grid_to_points(g)

def dist_euc(x,ys):
    x=x.expand_as(ys)
    return torch.norm(x.expand_as(ys)-ys,3,1)

ps=make_grid_points()
ps=dist_euc(torch.tensor([1,1,1]).float(),ps)

