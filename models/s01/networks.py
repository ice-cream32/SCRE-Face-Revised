import torch
from torch import nn
from torch.nn import functional as F
def conv(a,b,batch=False):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b) if batch else nn.GroupNorm(4,b),nn.ReLU())
class Residual(nn.Module):
    def __init__(self,a,b,stride):
        super().__init__();self.main=nn.Sequential(nn.Conv2d(a,b,3,stride,padding=1,bias=False),nn.GroupNorm(4,b),nn.ReLU(),nn.Conv2d(b,b,3,padding=1,bias=False),nn.GroupNorm(4,b));self.short=nn.Identity() if a==b and stride==1 else nn.Conv2d(a,b,1,stride,bias=False)
    def forward(self,x):return F.relu(self.main(x)+self.short(x))
class Res8(nn.Module):
    def __init__(self):
        super().__init__();self.stem=conv(1,16);self.blocks=nn.Sequential(Residual(16,16,1),Residual(16,32,2),Residual(32,64,2));self.head=nn.Sequential(nn.Flatten(),nn.Dropout(.2),nn.Linear(64*5*5,1))
    def forward(self,x):return self.head(self.blocks(self.stem(x))).reshape(-1)
def prep(x,kind):
    if kind!='densenet121':x=((x-x.mean((2,3),keepdim=True))/x.std((2,3),correction=0,keepdim=True).clamp_min(.01)).clamp(-3,3)
    if kind in ['resnet18','densenet121']:
        if kind=='resnet18':x=x/6+.5
        x=F.interpolate(x,(128,128),mode='bilinear',align_corners=False).repeat(1,3,1,1)
        x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
    return x
def augment(x):
    n=len(x);theta=x.new_zeros(n,2,3);a=(torch.rand(n,device=x.device)*2-1)*.12
    theta[:,0,0]=a.cos();theta[:,1,1]=a.cos();theta[:,0,1]=-a.sin();theta[:,1,0]=a.sin();theta[:,:,2]=(torch.rand(n,2,device=x.device)*2-1)*(.8*2/19)
    x=F.grid_sample(x,F.affine_grid(theta,x.shape,align_corners=False),align_corners=False,padding_mode='border');x=torch.where(torch.rand(n,1,1,1,device=x.device)<.5,x.flip(-1),x)
    return (x*(.5+torch.rand(n,1,1,1,device=x.device))+(torch.rand(n,1,1,1,device=x.device)-.5)*.2+torch.randn_like(x)*.015).clamp(0,1)
class DilatedBlock(nn.Module):
    def __init__(self,ch,d):
        super().__init__();self.main=nn.Sequential(nn.Conv2d(ch,ch,3,padding=d,dilation=d,bias=False),nn.GroupNorm(4,ch),nn.ReLU(),nn.Conv2d(ch,ch,3,padding=d,dilation=d,bias=False),nn.GroupNorm(4,ch))
    def forward(self,x):return F.relu(x+self.main(x))
class HiRes19(nn.Module):
    def __init__(self):
        super().__init__();self.features=nn.Sequential(conv(1,32),DilatedBlock(32,1),DilatedBlock(32,2),DilatedBlock(32,3));self.head=nn.Sequential(nn.Linear(32*10,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,1))
    def forward(self,x):
        h=self.features(x);height,width=h.shape[-2:]
        # Exact adaptive-average bins; avoids MPS's non-divisible 19->3 limitation.
        spatial=torch.stack([h[:,:,i*height//3:((i+1)*height+2)//3,j*width//3:((j+1)*width+2)//3].mean((2,3)) for i in range(3) for j in range(3)],dim=-1).flatten(1)
        z=torch.cat([h.mean((2,3)),spatial],dim=1);return self.head(z).reshape(-1)
