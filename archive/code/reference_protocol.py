from pathlib import Path
import io,random,hashlib,re
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from sklearn.metrics import roc_auc_score
from PIL import Image
import networks as n
import utils as u
VIEWS=['clean','blur1','blur15','blur2','down12','down7','motion5','contrast03','noise025','blur1_noise025','blur15_noise025','down9_noise02','blur1_contrast035_noise015','diagonal5','jpeg45','shift1']
VW=np.array([.5]+[.5/15]*15)
def jpeg(x,quality):
    ys=[]
    for a in x:
        buf=io.BytesIO();Image.fromarray((a[0].numpy().clip(0,1)*255).round().astype('uint8')).save(buf,format='JPEG',quality=quality);buf.seek(0)
        with Image.open(buf) as im:ys.append(torch.from_numpy(np.array(im,dtype=np.float32)/255)[None])
    return torch.stack(ys)
def validation(x):
    noise=torch.randn(x.shape,generator=torch.Generator().manual_seed(807123));b1=u.blur(x,1);b15=u.blur(x,1.5)
    return torch.cat([x,b1,b15,u.blur(x,2),u.down(x,12),u.down(x,7),u.motion(x),x*.3+.35,(x+.025*noise).clamp(0,1),(b1+.025*noise).clamp(0,1),(b15+.025*noise).clamp(0,1),(u.down(x,9)+.02*noise).clamp(0,1),(b1*.35+.325+.015*noise).clamp(0,1),u.motion(x,2),jpeg(x,45),F.pad(x,(1,0,0,0),mode='replicate')[:,:,:,:19]])
def legacy(x):
    out=x.clone();apply=torch.rand(len(x))<.5;types=torch.randint(3,(len(x),))
    for t in range(3):
        ids=torch.where(apply&(types==t))[0]
        if not len(ids):continue
        if t==0:out[ids]=u.blur(out[ids],random.choice([.5,1,1.5]))
        elif t==1:out[ids]=F.interpolate(F.interpolate(out[ids],size=(random.choice([9,12,15]),)*2,mode='bilinear',align_corners=False),size=(19,19),mode='bilinear',align_corners=False)
        else:
            z=u.blur(out[ids],random.choice([.5,1]));out[ids]=F.interpolate(F.interpolate(z,size=(random.choice([12,15]),)*2,mode='bilinear',align_corners=False),size=(19,19),mode='bilinear',align_corners=False)
    return out
def bce(p,y,views=True):
    q=p.reshape(-1,len(y)).clip(1e-6,1-1e-6);t=(y==1)[None,:];v=-(t*np.log(q)+(1-t)*np.log(1-q)).mean(1);return float(v@VW if views else v[0])
def metrics(p,y):
    q=p.reshape(len(VIEWS),len(y));t=(y==1);d=q>=.37;acc=(d==t).mean(1);rec=d[:,t].mean(1);sp=(~d[:,~t]).mean(1)
    return {'weighted_accuracy':float(VW@acc),'worst_view_accuracy':float(acc.min()),'clean_accuracy':float(acc[0]),'weighted_face_recall':float(VW@rec),'weighted_specificity':float(VW@sp),'views':[{'view':name,'accuracy':float(a),'recall':float(r),'specificity':float(s),'auc':float(roc_auc_score(t,pp))} for name,a,r,s,pp in zip(VIEWS,acc,rec,sp,q)]}
