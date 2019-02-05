from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
import pulp

class Netlp(nn.Module):
    def __init__(self):
        super(Netlp, self).__init__()
        self.fc1 = nn.Linear(28*28, 10)
        #self.fc1.bias.data.fill_(0)

    def forward(self, x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        return F.log_softmax(x, dim=1)

    def solve_lp(self,x,y):
        y=y.detach().numpy()
        x = x.view(-1,28*28)
        #xp = F.relu(self.fc1(x))
        #xp.sum().backward() #make sure grad is there
        #xp= xp.detach().numpy()
        #xp = self.fc1(x).detach().numpy()
        #print(xp.shape) # 64 x 10
        w=self.fc1.weight.detach().numpy()
        b=self.fc1.bias.detach().numpy()
        xpp = np.matmul(x,w.T)+b
        lp_vars_w = []
        for ii in range(10):
            lp_vars_w.append([])
            for i in range(28*28):
                lp_vars_w[-1].append(pulp.LpVariable('w%d_%d' % (ii,i), lowBound=w[ii][i]-1,upBound=w[ii][i]+1, cat='Continuous'))
        lp_vars_b = []
        for i in range(10):
            lp_vars_b.append(pulp.LpVariable('b%d' % i, lowBound=b[i]-1,upBound=b[1]+1,cat='Continuous'))
        problem=[]
        constraints=[]
        for target_idx in range(len(y)):
            target=y[target_idx]
            for idx in range(10):
                if xpp[target_idx][idx]<0:
                    #S w_i * x_i + b  <= 0
                    #constraints.append(sum([ lp_vars_w[j] * x[target_idx][j].item() for j in range(28*28)]) <= 0)
                    constraints.append(sum([ lp_vars_w[idx][j] * x[target_idx][j].item()   for j in range(28*28)] + lp_vars_b[idx]) <= 0.1)
                else:
                    #S w_i * x_i + b  <= 0
                    #constraints.append(sum([ lp_vars_w[j] * x[target_idx][j].item() for j in range(28*28)]) >= 0)
                    constraints.append(sum([ lp_vars_w[idx][j] * x[target_idx][j].item()  for j in range(28*28)] + lp_vars_b[idx]) >= -0.1)
                if target==idx:
                    problem.append(sum([ lp_vars_w[idx][j] * x[target_idx][j].item() for j in range(28*28)] + lp_vars_b[idx]))
                    #problem.append(sum([ lp_vars_w[j] for j in range(28*28)]))
                else:
                    problem.append(sum([ - lp_vars_w[idx][j] * x[target_idx][j].item() for j in range(28*28)] + lp_vars_b[idx]))
                    #problem.append(sum([ - lp_vars_w[j]  for j in range(28*28)]))
        lp_max = pulp.LpProblem("nn", pulp.LpMaximize)
        lp_max += sum(problem) , "Z"
        for constraint in constraints:
            lp_max += constraint
        lp_max.solve()
        print(pulp.LpStatus[lp_max.status])
        #zero the grad
        #self.fc1.bias.grad.data.fill_(0)
        #self.fc1.weight.grad.data.fill_(1)
        for variable in lp_max.variables():
            if variable.name[0]=='w':
                ii,i=map(int,variable.name[1:].split('_'))
                self.fc1.weight[ii][i]=float(variable.varValue)
            elif variable.name[0]=='b':
                i=int(variable.name[1:])
                self.fc1.bias[i]=float(variable.varValue)
            #if variable.varValue!=0.0:
            #    print("{} = {}".format(variable.name, variable.varValue))


class Netfc(nn.Module):
    def __init__(self):
        super(Netfc, self).__init__()
        self.fc1 = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        return F.log_softmax(x, dim=1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        model.solve_lp(data,target)
        loss = F.nll_loss(output, target)
        #loss.backward()
        #optimizer.step()
        if 1==1 or batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Netlp().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()
