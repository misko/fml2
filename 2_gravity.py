from __future__ import print_function
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
from fml2 import Disty,DistyNorm,Agg,AggMax,AggSum,AggProd
from gravity_generator import GravityDataset,show_scene
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

#https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    print(named_parameters)
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        print(n)
        if(p.requires_grad):
            print(n)
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    #plt.legend([Line2D([0], [0], color="c", lw=4),
    #            Line2D([0], [0], color="b", lw=4),
    #            Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.pause(5)

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
       
        conv=True
        if conv:
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
        else:
            #self.conv_model.add_module("lin1",nn.Linear(15*15*2,15*15*2))
            self.conv_model.add_module("lin1",nn.Linear(9*2,9*16))
            self.conv_model.add_module("act1",self.act())
            self.conv_model.add_module("lin2",nn.Linear(9*16,9*16))
            self.conv_model.add_module("act3",self.act())
            self.conv_model.add_module("lin3",nn.Linear(9*16,9*16))
            self.conv_model.add_module("act4",self.act())
            self.conv_model.add_module("lin4",nn.Linear(9*16,18))
            self.conv_model.add_module("act5",self.act())
            self.conv_model.add_module("lin5",nn.Linear(9*2,1))
            #self.conv_model.add_module("lin2",nn.Linear(15*15*2,15*15*2))
            #self.conv_model.add_module("bn2",nn.BatchNorm1d(15*15*2))
            #self.conv_model.add_module("act2",self.act())
            #self.conv_model.add_module("lin3",nn.Linear(15*15*2,15*15*2))
            #self.conv_model.add_module("bn3",nn.BatchNorm1d(15*15*2))
            #self.conv_model.add_module("act3",self.act())
            #self.conv_model.add_module("lin4",nn.Linear(15*15*2,15*15*2))

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
        energy=None
        if True:
            #goals_grid=self.agg.forward(self.R.forward(self.disty.forward(goals)),goals)
            #inputs_grid=self.agg.forward(self.disty.forward(inputs),inputs)
            #goals_grid=self.agg.forward(self.disty.forward(goals),goals)
            scenes=inputs_grid #torch.cat((inputs_grid,goals_grid),1)
            output = self.conv_model(scenes)
            energy = output.mean(3,False).mean(2,False).mean(1,False)
        else:
            scenes=inputs_grid.view(len(inputs),-1)
            output = self.conv_model(scenes)
            #energy = output.mean(1,False)
            energy = output.sum(1,False)
        #print("FWD SUM",output.sum())
        
        #inputs_tensor = torch.cat([ input['points'] for input in inputs ],0) #TODO WHY CANT I DO THIS?!?!?!
        #d_output_to_input = torch.autograd.grad(output.sum(),inputs_tensor,create_graph=True)
        #print(d_output_to_input.size())
        #sys.exit(1)


        #iterate...
        loss=0
        if False: #broken? TODO
            cost=0
            g_sum=0
            g_abs_sum=0
            for idx in range(len(inputs)):
                d_output_to_input, = torch.autograd.grad(output[idx].mean(),inputs[idx]['points'],create_graph=True,retain_graph=True)
                inputs[idx]['pred']=d_output_to_input
                cost += ((d_output_to_input-goals[idx]['points'])**2).mean()
                #cost += ((d_output_to_input-goals[idx]['points']).abs()).mean()
                g_sum += d_output_to_input.sum()
                g_abs_sum += d_output_to_input.abs().sum()
            cost/=len(inputs)
            g_sum/=len(inputs)
            g_abs_sum/=len(inputs)

        #do it all at once
        #calculate F=dE/dx
        #idx=0
        #torch.autograd.grad(energy[idx], inputs[idx]['points'], create_graph=True)
        #cost=0
        #dE_dx = [ torch.autograd.grad(energy[idx], inputs[idx]['points'],create_graph=True) for idx in range(len(inputs)) ]
        #input_tensors = [ input['points'] for input in inputs ]
        if True:
            for idx in range(len(inputs)):
                #dE_dx, = torch.autograd.grad(energy[idx], inputs[idx]['points'],create_graph=True)
                dE_dx, = torch.autograd.grad(output.sum(), inputs[idx]['points'],create_graph=True)
                inputs[idx]['pred']=dE_dx
                #if idx==1:
                #    print("PRED",dE_dx)
                #    print("GOAL",goals[idx]['points'])
                #sys.exit(1)
                #print(dE_dx.size(),goals[idx]['points'].size(),scenes.size())
                #sys.exit(1)
                loss += (((dE_dx - goals[idx]['points'])/inputs[idx]['points'].size()[0])**2).sum()
        #loss.backward()
        #print(loss)
        #dF_dE = torch.autograd.grad(loss,energy[0],create_graph=True)
        #print(dF_dE)
        #for parameter in model.parameters():
        #    print("GRAD",parameter.size(),parameter.grad)
        #    torch.autograd.grad(loss,parameter,create_graph=True,allow_unused=True)
        #print(scales)
        #pred_force=torch.autograd.grad(energy,input_tensors[0],create_graph=True)
        #goals_tensor = torch.cat([ goal['points'] for goal in goals ],0)
        #print(output.mean(3,False).mean(2,False).mean(1,False).size())
        #print(input_tensor.size())
        #print(len(inputs))
        #print(scales)
        #model.zero_grad()
        #d_output_to_input = torch.autograd.grad(output.mean(),inputs['points'],create_graph=True)
        #d_output_to_input = torch.cat(torch.autograd.grad(output.mean(),inputs['points'],create_graph=True),1)
        #cost = ((d_output_to_input-goals_tensor)**2).mean()
        #g_sum = d_output_to_input.sum()
        #g_abs_sum = d_output_to_input.abs().sum()

        return loss,0,0 #g_sum,g_abs_sum


def collate_gravity(l):
    goals=[]
    inputs=[]
    for train_example in l:
        #goals.append({'points':Variable(1e7*torch.Tensor(train_example['force']),requires_grad=False)})
        goals.append({'points':Variable(torch.Tensor(train_example['force'])/10000,requires_grad=False)})
        inputs.append({'points':Variable(torch.Tensor(train_example['points']),requires_grad = True) , 'attrs':torch.Tensor(train_example['mass'])})
    return goals,inputs

dim=2
gz=9.0

train_loader = DataLoader(GravityDataset(size=256*8,dim=dim), batch_size=64,collate_fn=collate_gravity)
test_loader = DataLoader(GravityDataset(size=1000,dim=dim), batch_size=64,collate_fn=collate_gravity)

model = Model(gz,dim)

learning_rate = 5e-4 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for epoch in range(10000):
    train_loss=0
    sz=0
    epoch_loss=0
    epoch_size=0
    for i,train_batch in enumerate(train_loader,0):
        sz+=len(train_batch[0])
        if False and i%10==0:
            #test
            model.eval()
            for j,test_batch in enumerate(test_loader,0):
                test_goals,test_inputs=test_batch
                test_loss,_,_ = model(test_goals,test_inputs)
                #test_loss += test_out.item()
                #mean_loss += (torch.cat([goal['points'] for goal in goals_test ])**2).mean().item()
        
        #run a training round
        model.train()
        
        train_goals,train_inputs=train_batch
        model_out,s,a = model(train_goals,train_inputs)
        
        train_loss += model_out.item()
        epoch_loss += train_loss
        epoch_size += sz
        #ga = a.item()
        #gs = s.item()
   
        optimizer.zero_grad()
        #print("PARAM",sum([ x.sum() for x in model.parameters() ]),sum([ np.prod(x.size()) for x in model.parameters() ]))
        #for parameter in model.parameters():
        #    print("PARAMS",parameter.size())
        #    #print("GRAD",parameter.size(),parameter.grad)
        model_out.backward()
        #plot_grad_flow(model.conv_model.named_parameters())
        optimizer.step()
        force=train_goals[0]['points'].detach().numpy()
        pred=train_inputs[0]['pred'].detach().numpy()
        #print(force,pred)
        show_scene(train_inputs[0]['points'].detach().numpy(),train_inputs[0]['attrs'].detach().numpy(),force,pred=pred)
        
        #print("TRAIN",train_loss,"TEST",test_loss,"TEST_MEAN",mean_loss,"DIFF",mean_loss-test_loss,ga,gs)
        #print(" TRAIN LOSS",train_loss/sz)
    print("TRAIN LOSS",epoch_loss/epoch_size)
            
