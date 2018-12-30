#!/usr/bin/python
import sys
import numpy as np
import argparse
import scipy.spatial.distance
import pickle
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

G=6.674e-11

class GravityDataset(Dataset):
    """Face Landmarks dataset."""

    def show_scene(self,scene):
        plt.clf()
        plt.axes().set_aspect('equal', 'datalim')
        for idx in range(scene['xyzs'].shape[0]):
            plt.scatter(scene['xyzs'][idx, 0], scene['xyzs'][idx, 1], s=scene['mass'][idx], marker='.', c='r')
            plt.quiver(scene['xyzs'][idx, 0], scene['xyzs'][idx, 1], scene['force'][idx,0]/G, scene['force'][idx,1]/G,color='b')
            if 'pred' in scene:
                plt.quiver(scene['xyzs'][idx, 0], scene['xyzs'][idx, 1], scene['force'][idx,0]/G, scene['force'][idx,1]/G,color='g')
        plt.pause(1)

    def __init__(self, dim, size, seed=0, max_bodies=10,min_bodies=2,max_mass=1000,min_mass=1,box_size=10):
        self.r = np.random.RandomState(seed)
        self.size = size
        self.dim = dim
        self.dataset = self.get_examples(n=size,max_bodies=max_bodies,min_bodies=min_bodies,max_mass=max_mass,min_mass=min_mass,box_size=box_size,dim=dim)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_examples(self,n=10,max_bodies=10,min_bodies=2,max_mass=1000,min_mass=1,box_size=10,dim=3):
        #number of planets per example
        planets=np.floor(np.random.rand(n)*(max_bodies-min_bodies+1)+min_bodies).astype(np.int)
        #xyz coord of each planet for all examples
        xyzs=np.random.rand(planets.sum(),dim)*box_size-box_size/2
        #mass of each planet
        ms=np.random.rand(planets.sum(),1)*(max_mass-min_mass)+min_mass
    
        idx=0
        out=[]
        for x in planets: # the next x planets are interacting
            current_xyzs=xyzs[idx:idx+x]
            current_masses=ms[idx:idx+x]
            distances=scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(current_xyzs))
            m=scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(current_masses, lambda u, v: u*v))*G
            f=np.zeros((x,dim))
            E=np.zeros(x)
            ru=np.zeros((x,x,dim))
            for i in range(x):
                for j in range(x):
                    if i!=j:
                        rv=current_xyzs[i]-current_xyzs[j]
                        r=np.linalg.norm(rv)
                        ru[i,j,:]=(current_xyzs[j]-current_xyzs[i])/pow(r,dim)
                        f[i]+=m[i,j]*ru[i,j,:]
                        E[i]+=m[i,j]/r
            out.append({'distances':distances,'mass':current_masses,'energy':sum(E),'force':f,'xyzs':current_xyzs})
            idx+=x
        return out

if __name__=='__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-o','--output', type=str, required=False)
    argparser.add_argument('-n','--examples', type=int, default=10)
    argparser.add_argument('-maxb','--maxbodies', type=int, default=10)
    argparser.add_argument('-minb','--minbodies', type=int, default=2)
    argparser.add_argument('-maxm','--maxmass', type=int, default=1000)
    argparser.add_argument('-minw','--minmass', type=int, default=1)
    argparser.add_argument('-gz','--boxsize', type=int, default=10)
    args = argparser.parse_args()
   
    g_dataset = GravityDataset(2, 100, seed=0, max_bodies=2,min_bodies=2,max_mass=1000,min_mass=1,box_size=10)
    for x in range(3):
        example = g_dataset[x]
        print(example)
        g_dataset.show_scene(example)
        exit
    #out = get_examples(args.examples,args.maxbodies,args.minbodies,args.maxmass,args.minmass,args.boxsize)

    #pickle.dump(out,open(args.output,'wb')) 
