"""Seven fixed networks, source-only family specialization and regularized fusion.
All task checkpoints refit within folds. Target prediction is separate and locked.
"""
from pathlib import Path
import argparse,json,sys,io,random,hashlib,platform,re
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
from scipy.optimize import minimize
from PIL import Image
import networks as n
import utils as u
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent.parent
CACHE=ROOT/'work/torch_cache/hub/checkpoints/densenet121-a639ec97.pth'
CACHE_SHA='a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395'
torch.set_num_threads(4);torch.hub.set_dir(str(ROOT/'work/torch_cache'));DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'
FAMILIES=['resnet8','densenet121','hires19'];SEEDS=[42,7,2026]
RECIPES={'A':{'resnet8':'base','densenet121':'base','hires19':'base'},'B':{'resnet8':'base','densenet121':'base','hires19':'special'},'C':{'resnet8':'special','densenet121':'base','hires19':'base'},'D':{'resnet8':'special','densenet121':'base','hires19':'special'}}
NAMES=list(RECIPES)+[f'{k}_weighted' for k in RECIPES]
VIEWS=['clean','blur1','blur15','blur2','down12','down7','motion5','contrast03','noise025','blur1_noise025','blur15_noise025','down9_noise02','blur1_contrast035_noise015','diagonal5','jpeg45','shift1']
VW=np.array([.5]+[.5/15]*15)
PLAN={'families':FAMILIES,'seeds':SEEDS,'recipes':RECIPES,'candidates':NAMES,'threshold':.37,'fixed_network_count':7,
 'outer_folds':3,'outer_seed':20261112,'inner_fraction':.2,'inner_seed_base':871,'meta_and_selection':'stratify inner107 into meta53 and selection54; early-stop uses inner107; outer remains untouched',
 'stage1_epochs':{'resnet8':80,'hires19':80,'densenet121':30},'stage2_epochs':{'resnet8':40,'hires19':40,'densenet121':20},
 'stage1_lr':{'resnet8':.001,'hires19':.001,'densenet121':.0001},'stage2_lr':{'resnet8':.0001,'hires19':.0001,'densenet121':.00003},
 'batches':{'scratch':64,'dense':32},'weight_decay':.001,'checkpoint':'phase1 inner clean BCE; phase2 all16 inner weighted BCE; each phase separately best',
 'base_policy':'original brightness/noise augment then legacy50% Gaussian/downsample/combined degradation',
 'special_policy':'geometric -> curriculum blur/downsample -> contrast/brightness -> noise -> optional JPEG; paired clean/degraded losses0.5/0.5',
 'special_hard_loss_weight':.15,'hard_fraction_per_class':.25,'hard_source':'previous-epoch loss per fit-source ID, classwise capped; start after5 epochs; no validation IDs',
 'resnet_special_types':[.25,.20,.25,.15,.15],'hires_special_types':[.35,.10,.20,.20,.15],
 'special_gaussian_max':1.8,'special_down_sizes':[8,10,12,15],'clean_preserved':True,'new_teacher_or_stage3':False,
 'fusion':'3 global nonnegative sum1 weights; BCE plus0.1 distance-to-equal penalty; minimum each0.05; fixed0.37 threshold',
 'fold_fusion_training':'inner meta53 only, score on inner selection54, never outer labels',
 'final_fusion_training':'all800 outer-OOF family scores after procedure assessment; no target labels',
 'selection':'mean inner-selection weighted accuracy, worst-view accuracy, clean accuracy, candidate order',
 'final_refit':'source-only select, median best epochs per kind/policy, refit all800 from initialization; no target early stopping',
 'validation_views':VIEWS,'validation_weights':VW.tolist(),'densenet_init_sha256':CACHE_SHA,'old_task_weights_used':False,'test_labels_used':False}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,v):(OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def guard(stage):
    attempts=[];allowed={ROOT/'work/data/train_data.mat',ROOT/'work/data/train_label.mat',CACHE}
    if stage=='predict':allowed.add(ROOT/'work/data/test_data.mat')
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        p=Path(args[0]).resolve();bad=p.is_relative_to(ROOT/'outputs') and not p.is_relative_to(OUT)
        bad|=p.is_relative_to(ROOT/'work') and p.suffix in ['.mat','.json','.npz','.txt','.csv','.pt','.pth','.pkl'] and p not in allowed and not p.is_relative_to(ROOT/'work/cnn_env')
        if bad:attempts.append(str(p));raise RuntimeError('Forbidden historical/target access '+str(p))
    sys.addaudithook(hook);return attempts
def build(kind,initial=False):
    if kind=='resnet8':return n.Res8()
    if kind=='hires19':return n.HiRes19()
    if initial:assert sha(CACHE)==CACHE_SHA
    net=models.densenet121(weights=None)
    if initial:
        # Same legacy key conversion as torchvision.densenet._load_state_dict;
        # read the explicitly hashed cache, without hub path lookup/download.
        state=torch.load(CACHE,map_location='cpu',weights_only=True)
        pattern=re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state):
            match=pattern.match(key)
            if match:state[match.group(1)+match.group(2)]=state.pop(key)
        net.load_state_dict(state)
    for p in net.parameters():p.requires_grad=False
    for p in net.features.denseblock4.parameters():p.requires_grad=True
    net.classifier=nn.Linear(net.classifier.in_features,1);return net
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
def geometric(x,hires):
    if not hires:return u.weak(x)
    count=len(x);a=(torch.rand(count)*2-1)*.12;theta=x.new_zeros(count,2,3);theta[:,0,0]=a.cos();theta[:,1,1]=a.cos();theta[:,0,1]=-a.sin();theta[:,1,0]=a.sin();theta[:,:,2]=(torch.rand(count,2)*2-1)*2.4/19
    x=F.grid_sample(x,F.affine_grid(theta,x.shape,align_corners=False),align_corners=False,padding_mode='border');return torch.where(torch.rand(count,1,1,1)<.5,x.flip(-1),x)
def specialized(x,kind,progress):
    clean=geometric(x,kind=='hires19');out=clean.clone();probs=PLAN['hires_special_types' if kind=='hires19' else 'resnet_special_types'];types=torch.multinomial(torch.tensor(probs),len(x),replacement=True);strength=.5+.5*min(1,progress*1.5)
    for t in range(4):
        ids=torch.where(types==t)[0]
        if not len(ids):continue
        if t==0:out[ids]=u.blur(out[ids],random.choice([.5,1,1.4,1.8])*strength)
        elif t==1:out[ids]=u.motion(out[ids],random.choice([0,1,2])) if progress>.25 else u.blur(out[ids],.6)
        elif t==2:out[ids]=u.down(out[ids],random.choice([8,10,12,15]) if progress>.3 else random.choice([12,15]))
        else:out[ids]=u.down(u.blur(out[ids],random.choice([.5,.8,1.])*strength),random.choice([10,12,15]))
    contrast=torch.tensor([1.,.7,.4])[torch.randint(3,(len(x),))][:,None,None,None];ns=torch.tensor([0.,.01,.02,.025])[torch.randint(4,(len(x),))][:,None,None,None]*strength
    out=((out-.5)*contrast+.5+(torch.rand(len(x),1,1,1)-.5)*.1+torch.randn_like(out)*ns).clamp(0,1)
    if random.random()<.15:out=jpeg(out,random.choice([45,65,85]))
    return clean,out
@torch.no_grad()
def predict(net,x,kind):
    net.eval();return np.concatenate([net(n.prep(b.to(DEVICE),kind)).flatten().sigmoid().cpu().numpy() for b in x.split(64)])
def bce(p,y,views=True):
    q=p.reshape(-1,len(y)).clip(1e-6,1-1e-6);t=(y==1)[None,:];v=-(t*np.log(q)+(1-t)*np.log(1-q)).mean(1);return float(v@VW if views else v[0])
def metrics(p,y):
    q=p.reshape(len(VIEWS),len(y));t=(y==1);d=q>=.37;acc=(d==t).mean(1);rec=d[:,t].mean(1);sp=(~d[:,~t]).mean(1)
    return {'weighted_accuracy':float(VW@acc),'worst_view_accuracy':float(acc.min()),'clean_accuracy':float(acc[0]),'weighted_face_recall':float(VW@rec),'weighted_specificity':float(VW@sp),'views':[{'view':name,'accuracy':float(a),'recall':float(r),'specificity':float(s),'auc':float(roc_auc_score(t,pp))} for name,a,r,s,pp in zip(VIEWS,acc,rec,sp,q)]}
def fit(kind,seed,phase,policy,x,y,fitids,valids=None,epochs=None,parent=None,tag=''):
    u.seed(seed+7000*phase);net=build(kind,initial=phase==1)
    if parent:net.load_state_dict(torch.load(parent,map_location='cpu',weights_only=True)['state_dict'])
    net.to(DEVICE);epochs=epochs or PLAN[f'stage{phase}_epochs'][kind];opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=PLAN[f'stage{phase}_lr'][kind],weight_decay=.001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    ty=torch.from_numpy((y==1).astype('float32'));rng=np.random.default_rng(seed+phase*113);best=float('inf');bs=None;be=epochs;history=[];hardness=np.ones(800);hardweight=np.ones(800);vx=(x[valids] if phase==1 else validation(x[valids])) if valids is not None else None;batchsize=32 if kind=='densenet121' else 64;hardstats=[]
    for epoch in range(1,epochs+1):
        net.train()
        if kind=='densenet121':
            for layer in net.modules():
                if isinstance(layer,nn.BatchNorm2d):layer.eval()
        if policy=='special' and epoch>5:
            hardweight[:]=1
            for cls in [-1,1]:
                ids=fitids[y[fitids]==cls];count=int(np.ceil(len(ids)*.25));chosen=ids[np.argsort(hardness[ids])[-count:]];hardweight[ids]=.85;hardweight[chosen]+=.15*len(ids)/len(chosen)
            hardstats.append({'epoch':epoch,'hard_positive_count':int(np.sum(hardweight[fitids[y[fitids]==1]]>1)),'hard_negative_count':int(np.sum(hardweight[fitids[y[fitids]==-1]]>1)),'max_weight':float(hardweight[fitids].max())})
        for bi,ids in enumerate(np.array_split(rng.permutation(fitids),int(np.ceil(len(fitids)/batchsize)))):
            random.seed(seed+phase*100000+epoch*1000+bi);torch.manual_seed(seed+phase*100000+epoch*1000+bi);t=ty[ids].to(DEVICE)
            if phase==2 and policy=='special':
                clean,corrupt=specialized(x[ids],kind,epoch/epochs);zc=net(n.prep(clean.to(DEVICE),kind)).flatten();zd=net(n.prep(corrupt.to(DEVICE),kind)).flatten();per=.5*F.binary_cross_entropy_with_logits(zc,t,reduction='none')+.5*F.binary_cross_entropy_with_logits(zd,t,reduction='none');hardness[ids]=.7*hardness[ids]+.3*per.detach().cpu().numpy();weights=torch.tensor(hardweight[ids],dtype=torch.float32,device=DEVICE);ll=(per*weights).mean()
            else:
                batch=n.augment(x[ids]);batch=legacy(batch) if phase==2 else batch;z=net(n.prep(batch.to(DEVICE),kind)).flatten();ll=F.binary_cross_entropy_with_logits(z,t)
            opt.zero_grad();ll.backward();opt.step()
        sched.step()
        if valids is not None and (epoch%2==0 or epoch==1 or epoch==epochs):
            vl=bce(predict(net,vx,kind),y[valids],phase==2);history.append({'epoch':epoch,'loss':vl})
            if vl<best:best=vl;be=epoch;bs={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    if valids is not None:net.load_state_dict(bs)
    record={'kind':kind,'seed':seed,'phase':phase,'policy':policy,'epoch':be,'history':history,'fit_ids':(fitids+1).tolist(),'validation_ids':[] if valids is None else (valids+1).tolist(),'parent_sha256':sha(parent) if parent else None,'init_sha256':CACHE_SHA if kind=='densenet121' else None,'hard_mining_records':hardstats,'hard_ids_source':'fit_ids only'}
    path=OUT/'models'/f'{tag}_{kind}_{seed}_{policy}_stage{phase}.pt';torch.save({**record,'state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()}},path);print('FIT',tag,kind,seed,phase,policy,be,round(best,5) if valids is not None else 'full800',flush=True);return net,path,record
def family_variants(groups):
    return {label:np.stack([groups[f'{k}_{pol}'] for k,pol in recipe.items()]) for label,recipe in RECIPES.items()}
def fit_weights(fam,y):
    p=fam.reshape(3,len(VIEWS),len(y));t=(y==1)[None,:]
    def loss(w):
        score=np.einsum('f,fvi->vi',w,p).clip(1e-6,1-1e-6);return float((-(t*np.log(score)+(1-t)*np.log(1-score)).mean(1))@VW+.1*np.sum((w-1/3)**2))
    r=minimize(loss,np.ones(3)/3,method='SLSQP',bounds=[(.05,.9)]*3,constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'maxiter':100,'ftol':1e-10});assert r.success and abs(r.x.sum()-1)<1e-6;return r.x
def rank(rows):return sorted(rows,key=lambda r:(-round(r['weighted_accuracy'],12),-round(r['worst_view_accuracy'],12),-round(r['clean_accuracy'],12),NAMES.index(r['candidate'])))
def train():
    attempts=guard('train');assert sha(CACHE)==CACHE_SHA;(OUT/'models').mkdir(exist_ok=True);dump('运行前固定方案.json',PLAN)
    X=loadmat(ROOT/'work/data/train_data.mat')['train_data'];y=loadmat(ROOT/'work/data/train_label.mat')['train_label'].ravel();assert X.shape==(800,361);x=u.image(X);folds=[];records=[];innerrows=[];outerrows=[];procedure=[];allscore={};oof={k:np.zeros((3,len(VIEWS),800)) for k in RECIPES};epochs={};fold_weights=[]
    for fold,(outertr,outerva) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261112).split(X,y),1):
        ids,va=train_test_split(outertr,test_size=.2,stratify=y[outertr],random_state=871+fold);meta,sel=train_test_split(np.arange(len(va)),test_size=.5,stratify=y[va],random_state=513+fold)
        assert not set(ids)&set(va) and not set(outerva)&set(ids) and not set(outerva)&set(va);folds.append({'fold':fold,'fit_ids':(ids+1).tolist(),'inner_ids':(va+1).tolist(),'meta_ids':(va[meta]+1).tolist(),'selection_ids':(va[sel]+1).tolist(),'outer_ids':(outerva+1).tolist()});vi=validation(x[va]);vo=validation(x[outerva]);inner={};outer={}
        for kind in FAMILIES:
            seeds=[42] if kind=='densenet121' else SEEDS;ps_inner={'base':[],'special':[]};ps_outer={'base':[],'special':[]}
            for seed in seeds:
                _,parent,r=fit(kind,seed,1,'base',x,y,ids,va,tag=f'fold{fold}');records.append(r);epochs.setdefault(f'{kind}_stage1',[]).append(r['epoch'])
                for policy in (['base'] if kind=='densenet121' else ['base','special']):
                    net,path,r=fit(kind,seed,2,policy,x,y,ids,va,parent=parent,tag=f'fold{fold}');records.append(r);epochs.setdefault(f'{kind}_{policy}_stage2',[]).append(r['epoch']);ps_inner[policy].append(predict(net,vi,kind));ps_outer[policy].append(predict(net,vo,kind));del net
            for pol in ['base'] if kind=='densenet121' else ['base','special']:
                inner[f'{kind}_{pol}']=np.mean(ps_inner[pol],0);outer[f'{kind}_{pol}']=np.mean(ps_outer[pol],0)
            dump('阶段训练记录_进行中.json',records)
        infam=family_variants(inner);outfam=family_variants(outer);ins={};outs={};fw={}
        for label,pp in infam.items():
            pp=pp.reshape(3,len(VIEWS),len(va));w=fit_weights(pp[:,:,meta],y[va[meta]]);fw[label]=w.tolist();ins[label]=pp[:,:,sel].mean(0).reshape(-1);ins[label+'_weighted']=np.einsum('f,fvi->vi',w,pp[:,:,sel]).reshape(-1)
            oo=outfam[label].reshape(3,len(VIEWS),len(outerva));oof[label][:,:,outerva]=oo;outs[label]=oo.mean(0).reshape(-1);outs[label+'_weighted']=np.einsum('f,fvi->vi',w,oo).reshape(-1)
            allscore[f'fold{fold}_{label}_inner_families']=pp;allscore[f'fold{fold}_{label}_outer_families']=oo
        fold_weights.append({'fold':fold,'meta_weights':fw});fr=[{'fold':fold,'candidate':k,**metrics(p,y[va[sel]])} for k,p in ins.items()];innerrows+=fr;outerrows +=[{'fold':fold,'candidate':k,**metrics(p,y[outerva])} for k,p in outs.items()];winner=rank(fr)[0]['candidate'];procedure.append({'fold':fold,'inner_winner':winner,**metrics(outs[winner],y[outerva])});dump('交叉验证进度.json',{'folds':folds,'inner':innerrows,'outer':outerrows,'procedure':procedure,'fold_weights':fold_weights});print('OUTER',fold,winner,procedure[-1]['weighted_accuracy'],flush=True)
    summaries=[]
    for name in NAMES:
        ii=[r for r in innerrows if r['candidate']==name];oo=[r for r in outerrows if r['candidate']==name];summaries.append({'candidate':name,**{k:float(np.mean([r[k] for r in ii])) for k in ['weighted_accuracy','worst_view_accuracy','clean_accuracy']},'outer_mean':float(np.mean([r['weighted_accuracy'] for r in oo])),'outer_std':float(np.std([r['weighted_accuracy'] for r in oo],ddof=1))})
    ranking=rank(summaries);winner=ranking[0]['candidate'];full_weights={k:fit_weights(v,y).tolist() for k,v in oof.items()};dump('仅源数据选型.json',{'primary':winner,'ranking':ranking,'full_fusion_weights_outer_OOF':full_weights,'outer_procedure':procedure,'test_data_read':False});dump('源图分折.json',folds);full_epochs={k:int(np.floor(np.median(v)+.5)) for k,v in epochs.items()};final=[]
    for kind in FAMILIES:
        for seed in ([42] if kind=='densenet121' else SEEDS):
            _,parent,r=fit(kind,seed,1,'base',x,y,np.arange(800),epochs=full_epochs[f'{kind}_stage1'],tag='full');records.append(r)
            for policy in ['base'] if kind=='densenet121' else ['base','special']:
                net,path,r=fit(kind,seed,2,policy,x,y,np.arange(800),epochs=full_epochs[f'{kind}_{policy}_stage2'],parent=parent,tag='full');records.append(r);del net;final.append({'file':str(path.relative_to(OUT)),'sha256':sha(path)})
    dump('全部阶段训练记录.json',records);np.savez(OUT/'仅源交叉验证分数.npz',**allscore,**{f'oof_{k}':v for k,v in oof.items()})
    import torchvision,sklearn,scipy,PIL
    dump('运行环境.json',{'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,'Pillow':PIL.__version__,'device':DEVICE})
    lock={'primary':winner,'ranking':ranking,'threshold':.37,'recipes':RECIPES,'family_order':FAMILIES,'full_fusion_weights':full_weights,'epochs':full_epochs,'models':final,'code_sha256':sha(__file__),'helper_sha256':{n:sha(OUT/n) for n in ['networks.py','utils.py']},'plan_sha256':sha(OUT/'运行前固定方案.json'),'split_sha256':sha(OUT/'源图分折.json'),'source_sha256':{n:sha(ROOT/'work/data'/n) for n in ['train_data.mat','train_label.mat']},'dense_init_sha256':sha(CACHE),'test_data_read':False,'test_labels_used':False,'forbidden_attempts':attempts};dump('模型锁定.json',lock);print('LOCKED',winner,json.dumps(ranking,ensure_ascii=False),flush=True)
def infer():
    guard('predict');lock=json.loads((OUT/'模型锁定.json').read_text());before=sha(OUT/'模型锁定.json');assert lock['code_sha256']==sha(__file__) and lock['threshold']==.37
    for n,h in lock['helper_sha256'].items():assert sha(OUT/n)==h
    for r in lock['models']:assert sha(OUT/r['file'])==r['sha256']
    x=u.image(loadmat(ROOT/'work/data/test_data.mat')['test_data']);assert len(x)==420;groups={}
    for kind in FAMILIES:
        for policy in ['base'] if kind=='densenet121' else ['base','special']:
            runs=[]
            for seed in [42] if kind=='densenet121' else SEEDS:
                net=build(kind,False);net.load_state_dict(torch.load(OUT/'models'/f'full_{kind}_{seed}_{policy}_stage2.pt',map_location='cpu',weights_only=True)['state_dict']);net.to(DEVICE);runs.append(predict(net,x,kind));del net
            groups[f'{kind}_{policy}']=np.mean(runs,0)
    fam=family_variants(groups);scores={};files=[]
    for label,v in fam.items():scores[label]=v.mean(0);scores[label+'_weighted']=np.einsum('f,fi->i',lock['full_fusion_weights'][label],v)
    for name,p in scores.items():
        path=OUT/f'{name}.txt';path.write_bytes(''.join(f'{i} {int(v)}\n' for i,v in enumerate(np.where(p>=.37,1,-1),1)).encode());files.append({'candidate':name,'file':path.name,'sha256':sha(path)})
    (OUT/'result_seven_optimized.txt').write_bytes((OUT/f"{lock['primary']}.txt").read_bytes());np.savez(OUT/'锁定后测试分数.npz',**scores,**groups);assert before==sha(OUT/'模型锁定.json');dump('预测核验.json',{'files':files,'primary':lock['primary'],'threshold':.37,'lock_unchanged':True,'manual_changes':False,'test_truth_used':False});print('PREDICTED',lock['primary'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['train','predict'],required=True);args=p.parse_args();train() if args.stage=='train' else infer()
