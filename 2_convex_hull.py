from __future__ import print_function
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
from fml2 import Disty,DistyNorm,Agg,AggMax
import gravity_generator
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self,gz,dim):
        super(Model, self).__init__()
        self.gz = gz
        self.conv_model = nn.Sequential()
        self.act = nn.ELU
        self.conv = nn.Conv2d
        self.bn = nn.BatchNorm2d
        if dim==3:
            self.conv = nn.Conv3d
            self.bn = nn.BatchNorm3d
        bn=True
        self.conv_model.add_module("conv1",self.conv(4, 32, 5, stride=2,padding=2))
        if bn:
            self.conv_model.add_module("bn1",self.bn(32))
        self.conv_model.add_module("act1",self.act())
        for x in range(16):
            self.conv_model.add_module("conv%d" % (x+2),self.conv(32, 32, 5, stride=1,padding=2))
            if bn:
                self.conv_model.add_module("bn%d" % (x+2),self.bn(32))
            self.conv_model.add_module("act%d" % (x+2),self.act())

        print(self.conv_model)

        #create the distnaces / norm generator
        self.distyNorm=DistyNorm(dim,norm="euc")
        self.disty=Disty(dim,gz,self.distyNorm)
        #create the aggregator 
        self.aggMax=AggMax()
        self.agg=Agg(dim,gz,self.aggMax)
        #create the renger function
        self.R=torch.nn.Sequential(torch.nn.Linear(2,16),torch.nn.ELU(),torch.nn.Linear(16,16),torch.nn.ELU(),torch.nn.Linear(16,4))

    def forward(self, goals,inputs):
        inputs_grid=self.agg.forward(self.R.forward(self.disty.forward(inputs)),inputs)
        scenes=inputs_grid #torch.cat((inputs_grid,goals_grid),1)

        output = self.conv_model(scenes)

        #inputs_tensor = torch.cat([ input['points'] for input in inputs ],0) #TODO WHY CANT I DO THIS?!?!?!
        #d_output_to_input = torch.cat(torch.autograd.grad(output,inputs_tensor,create_graph=True),0)


        #iterate...
        dpoints=[]
        dattrs=[]
        if True:
            cost=0
            g_sum=0
            g_abs_sum=0
            for idx in range(len(inputs)):
                if False:
                    d_output_to_input = torch.autograd.grad(output[idx].mean(),inputs[idx]['attrs'],create_graph=True)[0]
                    doutputs.append(d_output_to_input.data.numpy().reshape(1,-1))
                    cost += ((d_output_to_input-goals[idx]['attrs'])**2).mean()
                else:
                    d_output_to_points,d_output_to_attrs = torch.autograd.grad(output[idx].mean(),[inputs[idx]['points'],inputs[idx]['attrs']],create_graph=True)
                    dpoints.append(d_output_to_points.data.numpy())
                    dattrs.append(d_output_to_attrs.data.numpy())
                    cost += ((d_output_to_points-goals[idx]['force'])**2).mean()+((d_output_to_attrs-goals[idx]['attrs'])**2).mean()
                g_sum += d_output_to_points.sum()+d_output_to_attrs.sum()
                g_abs_sum += d_output_to_points.abs().sum()+d_output_to_attrs.abs().sum()
            cost/=len(inputs)
            g_sum/=len(inputs)
            g_abs_sum/=len(inputs)


        #do it all at once
        #goals_tensor = torch.cat([ goal['points'] for goal in goals ],0)
        #d_output_to_input = torch.autograd.grad(output.mean(),inputs['points'],create_graph=True)
        #d_output_to_input = torch.cat(torch.autograd.grad(output.mean(),inputs['points'],create_graph=True),1)
        #cost = ((d_output_to_input-goals_tensor)**2).mean()
        #g_sum = d_output_to_input.sum()
        #g_abs_sum = d_output_to_input.abs().sum()

        return cost,g_sum,g_abs_sum,dpoints,dattrs #np.concatenate(doutputs,axis=1)




def order_simplices(points,simplices):
    #find top most point
    mx_pnt_idx=0
    for x in range(len(points)):
        if points[mx_pnt_idx][1]<points[x][1]:
            mx_pnt_idx=x
    #make dict for simplices
    d={}
    for x,y in simplices:
        if x not in d:
            d[x]=[]
        if y not in d:
            d[y]=[]
        d[x].append(y)
        d[y].append(x)
    #find the sequence
    seq=[mx_pnt_idx]  
    next_point_idx=d[mx_pnt_idx][0]
    if points[next_point_idx][0]<points[d[mx_pnt_idx][1]][0]:
        next_point_idx=d[mx_pnt_idx][1]
    seq.append(next_point_idx)
    while True:
        next_point=d[seq[-1]][0]
        if next_point==seq[-2]:
            next_point=d[seq[-1]][1]
        if next_point==seq[0]:
            break
        seq.append(next_point)
    #seq to vector
    vecs=np.zeros(points.shape)
    for x in range(len(seq)):
        vecs[seq[x]]=points[seq[x-1]]-points[seq[x]]
    return vecs

def plot_points(goal,input,out,attr):
    plt.clf()
    points_np=input['points'].data.numpy()
    forces_np=goal['force'].data.numpy()
    #plot the points
    plt.plot(points_np[attr>0.5,0], points_np[attr>0.5,1], 'go') 
    plt.plot(points_np[attr<=0.5,0], points_np[attr<=0.5,1], 'bo') 
    #plot the vecs
    for x in range(points_np.shape[0]):
        xs=[points_np[x,0],points_np[x,0]+forces_np[x,0]]
        ys=[points_np[x,1],points_np[x,1]+forces_np[x,1]]
        plt.plot(xs,ys, 'k-')
    for x in range(points_np.shape[0]):
        xs=[points_np[x,0],points_np[x,0]+out[x,0]]
        ys=[points_np[x,1],points_np[x,1]+out[x,1]]
        plt.plot(xs,ys, 'r-')
    plt.draw()
    plt.pause(0.001)

def get_examples(n=1,scale=7):
    goals=[]
    inputs=[]
    for x in range(n):
        num_points=30
        points = (2*(np.random.rand(num_points, dim)-0.5))*scale
        hull = ConvexHull(points)
        in_hull = np.zeros(num_points)
        for v in hull.vertices:
            in_hull[v]=1.0
        vecs=order_simplices(points,hull.simplices)
        goals.append({'force':Variable(torch.Tensor(vecs),requires_grad=False),'attrs':torch.Tensor( in_hull )})
        inputs.append({'points':Variable(torch.Tensor(points),requires_grad = True) , 'attrs':Variable(torch.Tensor(np.ones( num_points )),requires_grad=True)})
        #plot_points(inputs[-1],goals[-1])
        #sys.exit(1)
        #inputs.append({'points':Variable(torch.Tensor(train_example['xyzs']),requires_grad = True) })
    return goals,inputs


dim=2
gz=15.0

test_set = get_examples(128)

model = Model(gz,dim)

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(50000):
    mmbz=16

    #evaluate on test set, and get difference to mean (0)
    model.eval()

    #goals_test,inputs_test=get_examples(max(1,mmbz/10))
    test_loss = 0
    mean_loss = 0
    n=0
    n2=0
    for x in range(len(test_set[0])):
        goals_test,inputs_test=test_set[0][x:x+1],test_set[1][x:x+1]
        test_out,_,_,out_points,out_attrs = model(goals_test,inputs_test)
	#n+=float((out>0.5).sum())
        #n2+=float((out<=0.5).sum())
        test_loss += test_out.item()
        mean_loss += (torch.cat([goal['attrs'] for goal in goals_test ])**2).mean().item()
        if x==0:
            plot_points(test_set[0][0],test_set[1][0],out_points[0],out_attrs[0])
    mean_loss/=len(test_set[0])
    test_loss/=len(test_set[0])
    n/=len(test_set[0])
    n2/=len(test_set[0])

    #run a training round
    model.train()
    optimizer.zero_grad()

    goals,inputs=get_examples(mmbz)
    model_out,s,a,out_points,out_attrs = model(goals,inputs)
    train_loss = model_out.item()
    ga = a.item()
    gs = s.item()

    print("PARAM",sum([ x.sum() for x in model.parameters() ]),sum([ np.prod(x.size()) for x in model.parameters() ]))
    model_out.backward()
    optimizer.step()
    
    print("TRAIN",train_loss,"TEST",test_loss,"TEST_MEAN",mean_loss,"DIFF",mean_loss-test_loss,ga,gs,"N",n,n2)
