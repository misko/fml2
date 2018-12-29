from __future__ import print_function
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
from fml2 import Disty,DistyNorm,Agg,AggMax, PDB
import random
import cPickle as pickle

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
        self.conv_model.add_module("conv1",self.conv(12, 8, 5, stride=2,padding=2))
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
        self.R=torch.nn.Sequential(torch.nn.Linear(12,16),torch.nn.ELU(),torch.nn.Linear(16,16),torch.nn.ELU(),torch.nn.Linear(16,4))

    def forward(self, goals,inputs,backgrounds):
        inputs_grid=self.agg.forward(self.R.forward(self.disty.forward(inputs)),inputs)
        goals_grid=self.agg.forward(self.R.forward(self.disty.forward(goals)),goals)
        backgrounds_grid=self.agg.forward(self.R.forward(self.disty.forward(backgrounds)),backgrounds)
        print("I",inputs_grid.sum(),"G",goals_grid.sum(),"B",backgrounds_grid.sum())
        #inputs_grid=self.agg.forward(self.disty.forward(inputs),inputs)
        #goals_grid=self.agg.forward(self.disty.forward(goals),goals)
        scenes=torch.cat((inputs_grid,goals_grid,backgrounds_grid),1)

        output = self.conv_model(scenes)
        print("O",output.sum())
        #inputs_tensor = torch.cat([ input['points'] for input in inputs ],0) #TODO WHY CANT I DO THIS?!?!?!
        #d_output_to_input = torch.cat(torch.autograd.grad(output,inputs_tensor,create_graph=True),0)


        #iterate...
        if True:
            cost=0
            g_sum=0
            g_abs_sum=0
            for idx in range(len(inputs)):
                d_output_to_input = torch.autograd.grad(output[idx].mean(),inputs[idx]['points'],create_graph=True)[0]
                print("D",d_output_to_input.sum())
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

p=PDB()

def get_examples(l,n=1):
    goals=[]
    inputs=[]
    backgrounds=[]
    idx=0
    while idx<n:
        e = None
        while e==None or e['ligand_atoms'].shape[0]==0:
            fn=random.choice(l)
            e = pickle.load( open( fn, "rb" ) )

        if e['receptor_atoms'].shape[0]==0 or e['ligand_atoms'].shape[0]==0:
            continue

        #randomly rotate
        h=p.get_random_rotation()
        e['ligand_atoms']=p.rotate(e['ligand_atoms'],h)
        e['receptor_atoms']=p.rotate(e['receptor_atoms'],h)

        #center on ligand and shift
        ligand_center=p.center(e['ligand_atoms'])
        e['ligand_atoms']=p.shift(e['ligand_atoms'],-ligand_center)
        e['receptor_atoms']=p.shift(e['receptor_atoms'],-ligand_center)
        t=p.get_random_translation()
        e['ligand_atoms']=p.shift(e['ligand_atoms'],t)
        e['receptor_atoms']=p.shift(e['receptor_atoms'],t)

        #trim
        e['receptor_atoms'],e['receptor_attrs']=p.trim_square(21,e['receptor_atoms'],e['receptor_attrs'])
        if e['receptor_atoms'].shape[0]==0:
            continue


        backgrounds.append({'points':Variable(torch.Tensor(e['receptor_atoms']),requires_grad=False), 'attrs':torch.Tensor(e['receptor_attrs'])})
        goals.append({'points':Variable(torch.Tensor(e['ligand_atoms']),requires_grad=False), 'attrs':torch.Tensor(e['ligand_attrs'])})

        #rotate and shift ligand
        #h=p.get_random_rotation()
        #e['ligand_atoms']=p.rotate(e['ligand_atoms'],h)

        ligand_center=p.center(e['ligand_atoms'])
        e['ligand_atoms']=p.shift(e['ligand_atoms'],-ligand_center)

        inputs.append({'points':Variable(torch.Tensor(e['ligand_atoms']),requires_grad=True), 'attrs':torch.Tensor(e['ligand_attrs'])})
        
        #train_example=gravity_generator.get_examples(min_mass=1,max_mass=10,n=1)[0]
        #goals.append({'points':Variable(,requires_grad=False) , 'attrs':torch.Tensor(train_example['mass'])})
        #inputs.append({'points':Variable(torch.Tensor(train_example['xyzs']),requires_grad = True) , 'attrs':torch.Tensor(train_example['mass'])})
        ##inputs.append({'points':Variable(torch.Tensor(train_example['xyzs']),requires_grad = True) })
        idx+=1
    return goals,inputs,backgrounds


def read_file(fn):
    f=open(fn)
    return [ line.strip() for line in f ] 

test_fns=read_file("test.list")
train_fns=read_file("train.list")

test_set = get_examples(test_fns,6)

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
        goals_test,inputs_test,backgrounds_test=test_set[0][x:x+1],test_set[1][x:x+1],test_set[2][x:x+1]
        test_out,_,_ = model(goals_test,inputs_test,backgrounds_test)
        test_loss += test_out.item()
        mean_loss += (torch.cat([goal['points'] for goal in goals_test ])**2).mean().item()

    #run a training round
    model.train()
    optimizer.zero_grad()

    goals,inputs,backgrounds=get_examples(train_fns,mmbz)
    model_out,s,a = model(goals,inputs,backgrounds)
    train_loss = model_out.item()
    ga = a.item()
    gs = s.item()

    print("PARAM",sum([ x.sum() for x in model.parameters() ]),sum([ np.prod(x.size()) for x in model.parameters() ]))
    model_out.backward()
    optimizer.step()
    
    print("TRAIN",train_loss,"TEST",test_loss,"TEST_MEAN",mean_loss,"DIFF",mean_loss-test_loss,ga,gs)
