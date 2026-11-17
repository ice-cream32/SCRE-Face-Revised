from pathlib import Path
import random,hashlib
import numpy as np
import torch
from torch.nn import functional as F
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)
def image(a):return torch.from_numpy(a.reshape(-1,19,19).transpose(0,2,1).copy()).float()[:,None]
def blur(x,s):
    k=int(np.ceil(s*3))*2+1;t=torch.arange(k,device=x.device,dtype=x.dtype)-k//2;v=torch.exp(-.5*(t/s)**2);v=v/v.sum();w=(v[:,None]*v[None,:])[None,None]
    return F.conv2d(F.pad(x,(k//2,)*4,mode='reflect'),w)
def down(x,n):
    # Exact area/adaptive-average weights, without unsupported MPS adaptive pooling.
    ii=torch.arange(n,device=x.device)[:,None];jj=torch.arange(19,device=x.device)[None,:]
    start=ii*19//n;end=((ii+1)*19+n-1)//n;w=((jj>=start)&(jj<end)).to(x.dtype)/(end-start)
    return F.interpolate(w@x@w.T,size=(19,19),mode='bilinear',align_corners=False)
def motion(x,kind=0):
    w=x.new_zeros(1,1,5,5)
    if kind==0:w[0,0,2,:]=.2
    elif kind==1:w[0,0,:,2]=.2
    else:w[0,0]=torch.eye(5,device=x.device)*.2
    return F.conv2d(F.pad(x,(2,)*4,mode='reflect'),w)
def weak(x):
    n=len(x);ang=(torch.rand(n,device=x.device)*2-1)*.12;t=x.new_zeros(n,2,3);t[:,0,0]=ang.cos();t[:,1,1]=ang.cos();t[:,0,1]=-ang.sin();t[:,1,0]=ang.sin();t[:,:,2]=(torch.rand(n,2,device=x.device)*2-1)*1.6/19
    x=F.grid_sample(x,F.affine_grid(t,x.shape,align_corners=False),align_corners=False,padding_mode='border');return torch.where(torch.rand(n,1,1,1,device=x.device)<.5,x.flip(-1),x)
