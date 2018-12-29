#!/usr/bin/python
import sys
import numpy as np
import argparse
import scipy.spatial.distance
import pickle

G=6.674e-11

def get_examples(n=10,max_bodies=10,min_bodies=2,max_mass=1000,min_mass=1,box_size=10):
    planets=np.floor(np.random.rand(n)*(max_bodies-min_bodies+1)+min_bodies).astype(np.int)
    xyzs=np.random.rand(planets.sum(),3)*box_size-box_size/2
    ms=np.random.rand(planets.sum(),1)*(max_mass-min_mass)+min_mass

    idx=0
    out=[]
    for x in planets: # the next x planets are interacting
        current_xyzs=xyzs[idx:idx+x]
        current_masses=ms[idx:idx+x]
        distances=scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(current_xyzs))
        m=scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(current_masses, lambda u, v: u*v))*G
        f=np.zeros((x,3))
        E=np.zeros(x)
        ru=np.zeros((x,x,3))
        for i in range(x):
            for j in range(x):
                if i!=j:
                    rv=current_xyzs[i]-current_xyzs[j]
                    r=np.linalg.norm(rv)
                    ru[i,j,:]=(current_xyzs[i]-current_xyzs[j])/pow(r,3)
                    f[i]+=m[i,j]*ru[i,j,:]
                    E[i]+=m[i,j]/r
        out.append({'distances':distances,'mass':current_masses,'energy':sum(E),'force':f,'xyzs':current_xyzs})
        idx+=x
    return out

if __name__=='__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-o','--output', type=str, required=True)
    argparser.add_argument('-n','--examples', type=int, default=10)
    argparser.add_argument('-maxb','--maxbodies', type=int, default=10)
    argparser.add_argument('-minb','--minbodies', type=int, default=2)
    argparser.add_argument('-maxm','--maxmass', type=int, default=1000)
    argparser.add_argument('-minw','--minmass', type=int, default=1)
    argparser.add_argument('-gz','--boxsize', type=int, default=10)
    args = argparser.parse_args()
    
    out = get_examples(args.examples,args.maxbodies,args.minbodies,args.maxmass,args.minmass,args.boxsize)

    pickle.dump(out,open(args.output,'wb')) 
