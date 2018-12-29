from __future__ import print_function
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
from fml2 import Disty,DistyNorm,Agg,AggMax
import gravity_generator

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
        self.conv_model.add_module("conv1",self.conv(4, 8, 5, stride=2,padding=2))
        if bn:
            self.conv_model.add_module("bn1",self.bn(8))
        self.conv_model.add_module("act1",self.act())
        for x in range(6):
            self.conv_model.add_module("conv%d" % (x+2),self.conv(8, 8, 5, stride=1,padding=2))
            if bn:
                self.conv_model.add_module("bn%d" % (x+2),self.bn(8))
            self.conv_model.add_module("act%d" % (x+2),self.act())

        print(self.conv_model)

        #create the distnaces / norm generator
        self.distyNorm=DistyNorm(dim,norm="euc")
        self.disty=Disty(dim,gz,self.distyNorm)
        #create the aggregator 
        self.aggMax=AggMax()
        self.agg=Agg(dim,gz,self.aggMax)
        #create the renger function
        self.R=torch.nn.Sequential(torch.nn.Linear(2,4),torch.nn.ELU(),torch.nn.Linear(4,4),torch.nn.ELU(),torch.nn.Linear(4,2))

    def forward(self, goals,inputs):
        inputs_grid=self.agg.forward(self.R.forward(self.disty.forward(inputs)),inputs)
        goals_grid=self.agg.forward(self.R.forward(self.disty.forward(goals)),goals)
        #inputs_grid=self.agg.forward(self.disty.forward(inputs),inputs)
        #goals_grid=self.agg.forward(self.disty.forward(goals),goals)
        scenes=torch.cat((inputs_grid,goals_grid),1)

        output = self.conv_model(scenes)

        #inputs_tensor = torch.cat([ input['points'] for input in inputs ],0) #TODO WHY CANT I DO THIS?!?!?!
        #d_output_to_input = torch.cat(torch.autograd.grad(output,inputs_tensor,create_graph=True),0)


        #iterate...
        if True:
            cost=0
            g_sum=0
            g_abs_sum=0
            for idx in range(len(inputs)):
                d_output_to_input = torch.autograd.grad(output[idx].mean(),inputs[idx]['points'],create_graph=True)[0]
                cost += ((d_output_to_input-goals[idx]['points'])**2).mean()
                g_sum += d_output_to_input.sum()
                g_abs_sum += d_output_to_input.abs().sum()
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

        return cost,g_sum,g_abs_sum

def get_examples(n=1):
    goals=[]
    inputs=[]
    for x in range(n):
        train_example=gravity_generator.get_examples(min_mass=1,max_mass=10,n=1)[0]
        goals.append({'points':Variable(1e7*torch.Tensor(train_example['force']),requires_grad=False) , 'attrs':torch.Tensor(train_example['mass'])})
        inputs.append({'points':Variable(torch.Tensor(train_example['xyzs']),requires_grad = True) , 'attrs':torch.Tensor(train_example['mass'])})
        #inputs.append({'points':Variable(torch.Tensor(train_example['xyzs']),requires_grad = True) })
    return goals,inputs

test_set = get_examples(128)

dim=3
gz=15.0

model = Model(gz,dim)

learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(50000):
    mmbz=16

    #evaluate on test set, and get difference to mean (0)
    model.eval()

    #goals_test,inputs_test=get_examples(max(1,mmbz/10))
    test_loss = 0
    mean_loss = 0
    for x in range(len(test_set[0])):
        goals_test,inputs_test=test_set[0][x:x+1],test_set[1][x:x+1]
        test_out,_,_ = model(goals_test,inputs_test)
        test_loss += test_out.item()
        mean_loss += (torch.cat([goal['points'] for goal in goals_test ])**2).mean().item()

    #run a training round
    model.train()
    optimizer.zero_grad()

    goals,inputs=get_examples(mmbz)
    model_out,s,a = model(goals,inputs)
    train_loss = model_out.item()
    ga = a.item()
    gs = s.item()

    print("PARAM",sum([ x.sum() for x in model.parameters() ]),sum([ np.prod(x.size()) for x in model.parameters() ]))
    model_out.backward()
    optimizer.step()
    
    print("TRAIN",train_loss,"TEST",test_loss,"TEST_MEAN",mean_loss,"DIFF",mean_loss-test_loss,ga,gs)
