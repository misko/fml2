
class Dist3(nn.Module):
    def make_grid(self,n=1):
        t=torch.linspace(-(self.gz/2),self.gz/2,self.gz)
        a=t.view(1,self.gz,1,1,1).clone().expand(1,self.gz,self.gz,self.gz,1)
        b=t.view(1,1,self.gz,1,1).clone().expand_as(a)
        c=t.view(1,1,1,self.gz,1).clone().expand_as(a)
        grid = torch.cat((a,b,c),dim=4).view(1,self.gz,self.gz,self.gz,3)
        grid = grid.expand(n,self.gz,self.gz,self.gz,3)
        return grid

    def grid_to_points(self,g,n=1):
        return g.contiguous().view(n*self.gz*self.gz*self.gz,3) 

    def values_to_grid(self,g,n=1,c=1):
        return g.view(n,c,self.gz,self.gz,self.gz) 

    def __init__(self,gz,norm="inv"):
        super(Dist3, self).__init__()
        self.gz = int(gz)
        self.norm  = norm

    def forward(self, x):
        #process the input
        n = x.size()[0]
        x = x.view(n,1,3)
        x = x.expand(n,self.gz*self.gz*self.gz,3)
        x = x.contiguous().view(n*self.gz*self.gz*self.gz,3)

        #make a grid
        grid = self.make_grid(n) # gz,gz,gz,3
        grid_points = self.grid_to_points(grid,n)
        #distance type
        norms = None
        if self.norm=="r3":
            norms = torch.norm(x-grid_points,3,1)
        elif self.norm=="lin":
            norms = torch.sum(x-grid_points,1)
        elif self.norm=="abs":
            norms = torch.sum(torch.abs(x-grid_points),1)
        elif self.norm=="inv":
            #norms = torch.sum(1.0/(torch.abs(x-grid_points)+1),1)
            norms = 1.0/(torch.norm(x-grid_points,3,1)+1)
        else:
            print("invalid distance type?")

        #norms = (self.values_to_grid(norms,n)).max(0,True)[0]
        norms = self.values_to_grid(norms,n)
        norms = norms.sum(0,True)
        return norms
