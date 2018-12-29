from __future__ import print_function
from torch.autograd import Variable
import torch
import torch.nn as nn
from fml2 import Disty,DistyNorm,Agg,AggMax

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
        self.conv_model.add_module("conv1",self.conv(2, 1, 5, stride=1,padding=2))
        if bn:
            self.conv_model.add_module("bn1",self.bn(1))
        self.conv_model.add_module("act1",self.act())
        for x in range(2):
            self.conv_model.add_module("conv%d" % (x+2),self.conv(1, 1, 5, stride=1,padding=2))
            if bn:
                self.conv_model.add_module("bn%d" % (x+2),self.bn(1))
            self.conv_model.add_module("act%d" % (x+2),self.act())

        print(self.conv_model)

        #create the distnaces / norm generator
        self.distyNorm=DistyNorm(dim,norm="euc")
        self.disty=Disty(dim,gz,self.distyNorm)
        #create the aggregator 
        self.aggMax=AggMax()
        self.agg=Agg(dim,gz,self.aggMax)
        #create the renger function
        self.R=torch.nn.Linear(1,1)

    def forward(self, goals,inputs):
        inputs_grid=self.agg.forward(self.R.forward(self.disty.forward(inputs)),inputs)
        goals_grid=self.agg.forward(self.R.forward(self.disty.forward(goals)),goals)
        #inputs_grid=self.agg.forward(self.disty.forward(inputs),inputs)
        #goals_grid=self.agg.forward(self.disty.forward(goals),goals)
        scenes=torch.cat((inputs_grid,goals_grid),1)

        output = self.conv_model(scenes)

        #inputs_tensor = torch.cat([ input['points'] for input in inputs ],0) #TODO WHY CANT I DO THIS?!?!?!
        #d_output_to_input = torch.cat(torch.autograd.grad(output,inputs_tensor,create_graph=True),0)

        #goals_tensor = torch.cat([ goal['points'] for goal in goals ],0)

        cost=0
        g_sum=0
        g_abs_sum=0
        for idx in range(len(inputs)):
            d_output_to_input = torch.autograd.grad(output[idx].mean(),inputs[idx]['points'],create_graph=True)[0]
            diff=Variable(goals[idx]['points']-inputs[idx]['points'],requires_grad=False)
            #cost+=((d_output_to_input-goals[idx]['points'])**2).mean()
            cost+=((d_output_to_input-diff)**2).mean()
            g_sum += d_output_to_input.sum()
            g_abs_sum += d_output_to_input.abs().sum()
        cost/=len(inputs)
        g_sum/=len(inputs)
        g_abs_sum/=len(inputs)

        return cost,g_sum,g_abs_sum


def get_examples(n=1):
    goals=[]
    inputs=[]
    for x in range(n):
        goals.append({'points':Variable(((torch.rand(1,dim)-0.5)*2)*(gz/4),requires_grad=False) })
        inputs.append({'points':Variable(((torch.rand(1,dim)-0.5)*2)*(gz/4),requires_grad = True) })
    return goals,inputs


dim=2
n=1
gz=5.0

model = Model(gz,dim)

learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(50000):
    mmbz=100

    #evaluate on test set, and get difference to mean (0)
    model.eval()

    goals_test,inputs_test=get_examples(int(max(1,mmbz/10)))
    test_out,_,_ = model(goals_test,inputs_test)
    test_loss = test_out.item()
    mean_loss = (torch.cat([goal['points'] for goal in goals_test ])**2).mean().item()

    #run a training round
    model.train()
    optimizer.zero_grad()

    goals,inputs=get_examples(mmbz)
    model_out,s,a = model(goals,inputs)
    train_loss = model_out.item()
    ga = a.item()
    gs = s.item()

    print("PARAM",sum([ x.sum() for x in model.parameters() ]),sum([ len(x) for x in model.parameters() ]))
    model_out.backward()
    optimizer.step()
    
    print("TRAIN",train_loss,"TEST",test_loss,"TEST_MEAN",mean_loss,"DIFF",mean_loss-test_loss,ga,gs)
