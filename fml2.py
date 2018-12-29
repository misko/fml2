from __future__ import print_function
#import cPickle as pickle
import pickle
import torch
import torch.nn as nn
import sys
import gzip
import numpy as np
import gzip

class AggMax(nn.Module):
    def __init__(self):
        super(AggMax, self).__init__()

    def forward(self,input):
        #input n x gz^dim x r_out
        #output should be 1 x r_out x gz^dim
        return input.max(0,True)[0].transpose(2,1)

class Agg(nn.Module):
    def grid_to_points(self,g,n=1):
        return g.contiguous().view(n*pow(self.gz,self.dim),self.dim) 

    def __init__(self,dim,gz,agg):
        super(Agg, self).__init__()
        self.dim = int(dim)
        self.gz = int(gz)
        self.agg  = agg

    def forward(self, distances_with_attrs, inputs):
        #input is (NxG)xRout , need to shape to NxGxRout, then break out and agg
        n=int(distances_with_attrs.shape[0]/pow(self.gz,self.dim))

        #all the cubes, each point has a cube
        all_hs=distances_with_attrs.view(n,pow(self.gz,self.dim),-1)

        #lets aggregate the point cubes into example cubes
        aggd_hs=[]
        idx=0
        for input in inputs:
            aggd_hs.append( self.agg.forward( all_hs[idx:idx+len(input['points'])] ).view((1,-1)+(self.gz,)*self.dim))
            idx+=len(input['points'])

        return torch.cat(aggd_hs,0)

class DistyNorm(nn.Module):
    def __init__(self,dim,norm="inv"):
        super(DistyNorm, self).__init__()
        self.dim = int(dim)
        self.norm  = norm
    
    def forward(self,inputs):
        #distance type
        distances = None
        if self.norm=="euc":
            distances = torch.norm(inputs,self.dim,1)
        elif self.norm=="lin":
            distances = torch.sum(inputs,1)
        elif self.norm=="abs":
            distances = torch.sum(torch.abs(inputs),1)
        elif self.norm=="inv":
            #distances = torch.sum(1.0/(torch.abs(x-grid_points)+1),1)
            distances = 1.0/(torch.norm(inputs,self.dim,1)+1)
        else:
            print("invalid distance type?")
        return distances

class Disty(nn.Module):
    def make_grid(self,n=1):
        t=torch.linspace(-(self.gz/2),self.gz/2,self.gz)
        l=[t.view((1,)*(x+1)+(self.gz,)+(1,)*(self.dim-x)).clone().expand((1,)+(self.gz,)*self.dim + (1,)) for x in range(self.dim)]
        grid = torch.cat(l,dim=self.dim+1).view((1,) + (self.gz,)*self.dim + (self.dim,))
        grid = grid.expand((n,) + (self.gz,)*self.dim + (self.dim,))
        return grid

    def __init__(self,dim,gz,norm):
        super(Disty, self).__init__()
        if gz%2==0:
            print("Grid size must be odd!")
            sys.exit(1)
        self.dim = int(dim)
        self.gz = int(gz)
        self.norm  = norm

    def forward(self, inputs):
        #firs tlets get input into an array
        points=torch.cat([ input['points'] for input in inputs ])
        n=points.shape[0]
        #now lets repeat each point by grid shape...
        points=points.view(-1,1,self.dim).expand(-1,pow(self.gz,self.dim),self.dim).contiguous().view(-1,self.dim)

        #now get the distance from grid to these points
        grid_points=self.make_grid(n).contiguous().view(n*pow(self.gz,self.dim),self.dim)

        distances = self.norm.forward(points-grid_points)

        if 'attrs' in inputs[0]:
            attrs=torch.cat([ input['attrs'] for input in inputs ]).view((n,1,-1)).expand(n,pow(self.gz,self.dim),-1).contiguous().view(n*pow(self.gz,self.dim),-1)
            distances_with_attrs=torch.cat((distances.view(-1,1),attrs),dim=1)
            return distances_with_attrs
        else:
            return distances.view(-1,1)

class PDB():
    def __init__(self):
        #self.filename  = filename
        #  58 CA
        #  59 MN
        # 140 ZN
        # 157 SE
        # 400 P
        #4232 S
        #112183 H
        #146834 N
        #194916 O
        #540784 C
        self.ty_to_idx = {'C':0,'O':1,'N':2,'H':3,'S':4,'P':5,'SE':6,'ZN':7,'MN':8,'CA':9,'-':10}
        self.idx_to_ty = {}
        for key in self.ty_to_idx:
            self.idx_to_ty[self.ty_to_idx[key]]=key

    def load(self,filename):
        f=None
        atoms_receptor=[]
        atoms_ligand=[]
        if filename[-2:]=='gz':
            f=gzip.open(filename)
        else:
            f=open(filename)
        for line in f:
            if line[:4]=='ATOM':
                #receptor side
                atoms_receptor.append(self.read_atom_line(line))
            elif line[:6]=='HETATM':
                #ligand side
                #make sure only chain A
                if line[21]=='A':
                    #remove any waters 
                    if line[17:17+3]=='HOH':
                        continue
                    atoms_ligand.append(self.read_atom_line(line))
            #ATOM    321  C5'   G B  16      35.018   9.987  -2.036  1.00 17.32           C 
            # 0       1    2    3 4  5       6        7      8       9    10              11
        d={}
        d['receptor_atoms'] = np.array([ r['p'] for r in atoms_receptor ])
        d['receptor_attrs'] = np.array([ self.ty_to_vec(r['ty']) for r in atoms_receptor ])
        d['ligand_atoms'] = np.array([ r['p'] for r in atoms_ligand ])
        d['ligand_attrs'] = np.array([ self.ty_to_vec(r['ty']) for r in atoms_ligand ])
        return d
        #self.center_ligand()
        #self.trim_square()
        #self.save('wtf_ligand_out.xyz.gz',self.atoms_ligand)
       
    	#r=self.generate_3d() #  rotate
        #self.receptor_points_np = np.dot(self.receptor_points_np,r)
        #self.ligand_points_np = np.dot(self.ligand_points_np,r)
        #self.from_np()
        #self.save('wtf_receptor.xyz.gz',self.atoms_receptor) # scene/receptor goal
        #self.save('wtf_ligand_out.xyz.gz' , self.atoms_ligand) # ligand goal
        #t=np.random.rand(3) # translate 
        #self.ligand_points_np += t
        #self.from_np()
        #self.save('wtf_ligand_in.xyz.gz' , self.atoms_ligand) # ligand input
            
    def read_atom_line(self,line):
        #ATOM     10  N1    C A   1      34.130  -2.299   2.970  1.00 18.82           N 
        line=line[:12]+" "+line[26:]
        line=line[:43]+" "+line[53:]
        line=line.strip().split()
        str_ty=line[5]
        ty=0
        if str_ty in self.ty_to_idx:
            ty=self.ty_to_idx[str_ty]
        else:
            ty=self.ty_to_idx['-']
        x,y,z=[float(x) for x in line[2:2+3] ] 
        n=int(line[1])
        return {'ty':ty,'str_ty':str_ty,'p':(x,y,z) , 'n':n}

    def center(self,atoms):
        return atoms.mean(axis=0)

    def shift(self,atoms,t):
        atoms+=t
        return atoms

    def rotate(self,atoms,h):
        return np.dot(atoms,h)

    def get_random_rotation(self):
        return self.generate_3d()

    def get_random_translation(self,scale=1):
        return np.random.randn(1,3)*scale

    def trim_square(self,gz,atoms,attrs):
        new_atoms=[]
        new_attrs=[]
        for x in range(atoms.shape[0]):
            atom=atoms[x]
            attr=attrs[x]
            if abs(atom).max()>(2*gz)/3:
                continue
            new_atoms.append(atom)
            new_attrs.append(attr)
        if len(new_atoms)==0:
            return np.array([]),np.array([])
        return np.concatenate(new_atoms,axis=0), np.concatenate(new_attrs,axis=0)

  
    def center_ligand(self):
        center = self.ligand_points_np.mean(axis=0)
        for atom in self.atoms_receptor:
            atom['p']-=center
        for atom in self.atoms_ligand:
            atom['p']-=center
        self.to_np()

    def ty_to_vec(self,ty):
        v=np.zeros((1,len(self.ty_to_idx)))
        v[0,ty]=1
        return v

    def vec_to_ty(self,v):
        return self.idx_to_ty[np.argmax(v)]

    def to_np(self):
        self.receptor_points_np = np.array([ r['p'] for r in self.atoms_receptor ])
        self.receptor_attrs_np = np.array([ self.ty_to_vec(r['ty']) for r in self.atoms_receptor ])
        self.ligand_points_np = np.array([ r['p'] for r in self.atoms_ligand ])
        self.ligand_attrs_np = np.array([ self.ty_to_vec(r['ty']) for r in self.atoms_ligand ])

    def from_np(self):
        for x in range(len(self.atoms_receptor)):
            atom=self.atoms_receptor[x]
            atom['p']=( self.receptor_points_np[x,0],self.receptor_points_np[x,1],self.receptor_points_np[x,2] )
        for x in range(len(self.atoms_ligand)):
            atom=self.atoms_ligand[x]
            atom['p']=( self.ligand_points_np[x,0],self.ligand_points_np[x,1],self.ligand_points_np[x,2] )
  
    def save(self,output_filename,atoms,attrs):
        f=None
        if output_filename[-3:]==".gz":
            f=gzip.open(output_filename,'w')
        else:
            f=open(output_filename,'w')
        for x in range(atoms.shape[0]):
            atom=atoms[x]
            attr=attrs[x]
            f.write("%s\t%8.4f\t%8.4f\t%8.4f\n" % (self.vec_to_ty(attr),atom[0,0],atom[0,1],atom[0,2]))

    #https://github.com/qobilidop/randrot/
    def generate_2d(self):
        """Generate a 2D random rotation matrix.
        Returns:
            np.matrix: A 2D rotation matrix.
        """
        x = np.random.random()
        M = np.matrix([[np.cos(2 * np.pi * x), -np.sin(2 * np.pi * x)],
                       [np.sin(2 * np.pi * x), np.cos(2 * np.pi * x)]])
        return M
    
    #httpssssssssss://github.com/qobilidop/randrot/
    def generate_3d(self):
        """Generate a 3D random rotation matrix.
        Returns:
            np.matrix: A 3D rotation matrix.
        """
        x1, x2, x3 = np.random.rand(3)
        R = np.matrix([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
                       [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
                       [0, 0, 1]])
        v = np.matrix([[np.cos(2 * np.pi * x2) * np.sqrt(x3)],
                       [np.sin(2 * np.pi * x2) * np.sqrt(x3)],
                       [np.sqrt(1 - x3)]])
        H = np.eye(3) - 2 * v * v.T
        M = -H * R
        return M


if __name__=='__main__':
    if len(sys.argv)!=3:
        print("%s pdb out" % sys.argv[0])
        sys.exit(1)

    fn_in=sys.argv[1]
    fn_out=sys.argv[2]

    p=PDB()
    e=p.load(fn_in)

    #center on ligand
    ligand_center=p.center(e['ligand_atoms'])
    e['ligand_atoms']=p.shift(e['ligand_atoms'],-ligand_center)
    e['receptor_atoms']=p.shift(e['receptor_atoms'],-ligand_center)

    e['receptor_atoms'],e['receptor_attrs']=p.trim_square(51,e['receptor_atoms'],e['receptor_attrs'])

    pickle.dump( e, gzip.open( fn_out, "wb" ) )

    sys.exit(1)
    #randomly rotate
    h=p.get_random_rotation()
    e['ligand_atoms']=p.rotate(e['ligand_atoms'],h)
    e['receptor_atoms']=p.rotate(e['receptor_atoms'],h)

    #center on ligand
    ligand_center=p.center(e['ligand_atoms'])
    e['ligand_atoms']=p.shift(e['ligand_atoms'],-ligand_center)
    e['receptor_atoms']=p.shift(e['receptor_atoms'],-ligand_center)

    #trim
    before=len(e['receptor_atoms'])
    e['receptor_atoms'],e['receptor_attrs']=p.trim_square(21,e['receptor_atoms'],e['receptor_attrs'])
    after=len(e['receptor_atoms'])

    p.save('wtf_ligand_out.xyz.gz' , e['ligand_atoms'],e['ligand_attrs']) # ligand input
    p.save('wtf_receptor_out.xyz.gz' , e['receptor_atoms'],e['receptor_attrs']) # ligand input

    #rotate and shift ligand
    h=p.get_random_rotation()
    e['ligand_atoms']=p.rotate(e['ligand_atoms'],h)
    t=p.get_random_translation()
    e['ligand_atoms']=p.shift(e['ligand_atoms'],t)

    #save as input
    p.save('wtf_ligand_in.xyz.gz' , e['ligand_atoms'],e['ligand_attrs']) # ligand input
        
