from __future__ import print_function
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
from fml2 import Disty,DistyNorm,Agg,AggMax,AggSum,AggProd
from gravity_generator import GravityDataset,show_scene
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Gravity example')
parser.add_argument('--load', type=str, help='load checkpoint')
parser.add_argument('--dim', type=int, help='dim is 2 or 3', default=2)
parser.add_argument('--grid-size', type=int, help='grid size (NxN) 2D or (NxNxN) 3D', default=9)
parser.add_argument('--learning-rate', type=float, help='learning rate', default=1e-3)
args = parser.parse_args()

class Model(nn.Module):
    def __init__(self,gz,dim):
        super(Model, self).__init__()
        self.gz = gz
        self.conv_model = nn.Sequential()
        self.act = nn.ELU # nn.Softplus # nn.ELU
        self.conv = nn.Conv2d
        self.bn = nn.BatchNorm2d
        if dim==3:
            self.conv = nn.Conv3d
            self.bn = nn.BatchNorm3d
        bn=False
       
        #self.conv_model.add_module("conv1",self.conv(4, 8, 5, stride=2,padding=2))
        self.conv_model.add_module("conv1",self.conv(2, 8, 5, stride=2,padding=2))
        if bn:
            self.conv_model.add_module("bn1",self.bn(8))
        self.conv_model.add_module("act1",self.act())
        for x in range(3):
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
        self.aggSum=AggSum()
        self.agg=Agg(dim,gz,self.aggSum)
        #create the renger function
        self.R=torch.nn.Sequential(torch.nn.Linear(2,4),torch.nn.ELU(),torch.nn.Linear(4,4),torch.nn.ELU(),torch.nn.Linear(4,2))

    def forward(self, goals,inputs):
        tmp=self.disty.forward(inputs)
        R_tmp=self.R.forward(tmp)
        #tmp=tmp.prod(1,True)
        inputs_grid=self.agg.forward(R_tmp,inputs)
        output = self.conv_model(inputs_grid)
        energy = output.mean(3,False).mean(2,False).mean(1,False)

        #iterate...
        loss=0
        for idx in range(len(inputs)):
            dE_dx, = torch.autograd.grad(output.sum(), inputs[idx]['points'],create_graph=True)
            inputs[idx]['pred']=dE_dx
            loss += (((dE_dx - goals[idx]['points'])/inputs[idx]['points'].size()[0])**2).sum()

        return loss,0,0 #g_sum,g_abs_sum


def collate_gravity(l):
    goals=[]
    inputs=[]
    for train_example in l:
        #goals.append({'points':Variable(1e7*torch.Tensor(train_example['force']),requires_grad=False)})
        goals.append({'points':Variable(torch.Tensor(train_example['force'])/10000,requires_grad=False)})
        inputs.append({'points':Variable(torch.Tensor(train_example['points']),requires_grad = True) , 'attrs':torch.Tensor(train_example['mass'])})
    return goals,inputs


train_loader = DataLoader(GravityDataset(size=256*32,dim=args.dim), batch_size=32,collate_fn=collate_gravity)
test_loader = DataLoader(GravityDataset(size=1000,dim=args.dim), batch_size=64,collate_fn=collate_gravity)


model = Model(args.grid_size,args.dim)

def save_model(model,optimizer,save_fn,epoch):
    torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			}, save_fn)

def load_model(save_fn,model,optimizer):
    d=torch.load(save_fn)
    model.load_state_dict(d['model_state_dict'])
    optimizer.load_state_dict(d['optimizer_state_dict'])

  
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

if args.load:
    load_model(args.load,model,optimizer)

for epoch in range(10000):
    train_loss=0
    sz=0
    epoch_loss=0
    epoch_size=0
    for i,train_batch in enumerate(train_loader,0):
        epoch_size += len(train_batch[0])

        if i%1000==0:
            save_model(model,optimizer,"save_%d.t7" % epoch,epoch)
            #test
            model.eval()
            test_sz=0
            avg_test_loss=0
            test_inputs=None
            for j,test_batch in enumerate(test_loader,0):
                test_sz+=len(test_batch[0])
                test_goals,test_inputs=test_batch
                test_loss,_,_ = model(test_goals,test_inputs)
                avg_test_loss+=test_loss.item()
                #print(test_loss.item()/len(test_batch[0]))
                #mean_loss += (torch.cat([goal['points'] for goal in goals_test ])**2).mean().item()
            show_scene(test_inputs[0]['points'].detach().numpy(),test_inputs[0]['attrs'].detach().numpy(),test_goals[0]['points'].detach().numpy(),pred=test_inputs[0]['pred'].detach().numpy())
            print("*TEST LOSS",avg_test_loss/test_sz)
        
        #run a training round
        model.train()
        
        train_goals,train_inputs=train_batch
        model_out,s,a = model(train_goals,train_inputs)
        
        epoch_loss += model_out.item()
   
        optimizer.zero_grad()

        model_out.backward()
        optimizer.step()
        force=train_goals[0]['points'].detach().numpy()
        pred=train_inputs[0]['pred'].detach().numpy()
        #print(force,pred)
        
        #print("TRAIN",train_loss,"TEST",test_loss,"TEST_MEAN",mean_loss,"DIFF",mean_loss-test_loss,ga,gs)
        #print(" TRAIN LOSS",train_loss/sz)
    print("*TRAIN LOSS",epoch_loss/epoch_size)
            
