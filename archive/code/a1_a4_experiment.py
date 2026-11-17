"""Controlled second-stage experiments, source-only model selection."""
from pathlib import Path
import sys,json,random,hashlib,time,csv,argparse,importlib.util,math
import numpy as np
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from scipy.io import loadmat

ROOT=Path('${ORIGINAL_WORKSPACE}')
BASE=ROOT/'outputs/七网络原图通道与条件蒸馏';SRC=ROOT/'outputs/七网络定向鲁棒优化'
OUT=ROOT/'outputs/A1-A4模糊鲁棒性对照';OUT.mkdir(exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
sys.path.insert(0,str(SRC));sp=importlib.util.spec_from_file_location('original',SRC/'experiment.py');e=importlib.util.module_from_spec(sp);sp.loader.exec_module(e)
torch.set_num_threads(4);DEVICE=e.DEVICE
KINDS=['resnet8','hires19','densenet121'];SEEDS={'resnet8':[42,7,2026],'hires19':[42,7,2026],'densenet121':[42]}
VARIANTS=['A1','A2','A3','A4'];VIEW_NAMES=e.VIEWS+['held_disk15','held_jpeg25','held_impulse02','held_blur_speckle']
PLAN={'date':'2026-11-13','A0':'exact previous seven-network A','experts':7,'stage1':'reuse exact same-partition original stage1 weights; no extra expert stage',
 'A1':'mean supervised BCE on common clean/geometric/photometric view and 2 independent reordered corruption views',
 'A2':'A1 + 0.1 Jensen-Shannon over 3 binary distributions',
 'A3':'A1 + 0.1 conditional clean-teacher to 2 degraded-student Bernoulli KL; teacher true-class score >=0.9; no student difficulty filter',
 'A4':'chosen A1/A2/A3 objective + 0.1 auxiliary L1 lowpass clean reconstruction from degraded feature maps; decoder removed at inference',
 'A4_selection':'each outer fold chooses its base using only own inner107 A1-A3 scores; full refit uses mean inner107 results, never outer scores',
 'teacher_selection':'phase1 family with lowest inner107 clean BCE; average seeds within family; same fit IDs; full teacher uses mean inner clean BCE',
 'epochs_max':{'resnet8':40,'hires19':40,'densenet121':20},'learning_rate':{'resnet8':.0001,'hires19':.0001,'densenet121':.00003},
 'optimizer':'AdamW weight_decay0.001; cosine per run; same baseline trainable DenseNet block4+head; all DenseNet BN running statistics frozen',
 'batch':{'small':64,'dense':32},'validation':'first16 views; clean0.5 others0.5/15; BCE checkpoint at first/even/last epoch',
 'held_out_corruptions':'disk defocus, JPEG25, saltpepper, blur+speckle; assessed on outer original IDs only; no checkpoint/selection',
 'aux_target':'Gaussian sigma1 of same geometric and photometric clean view; both classes; decoder 1x1 ->16, bilinear19, conv3 ->1 sigmoid',
 'selection':'clean accuracy >= A0-0.01 and weighted specificity>=A0-0.005 preferred; then weighted accuracy desc, worst-view face recall desc, BCE asc, A1/A2/A3 order; no outer selection',
 'final_epochs':'median same-family inner-selected epochs; phase2 full800 starts original full800 stage1',
 'fusion':'probability mean within seeds; family mean 1/3','threshold':.37,'threshold_tuned':False,'seed_flow':'same seed/tag/epoch/batch inputs across A1/A2/A3/A4',
 'note':'A1 vs A0 changes corruption distribution and view-paired CE; A2/A3 isolate losses relative to A1',
 'log':'online objective and separate losses each epoch; all-fit clean and inner clean/robust eval first/even/last; all-fit fixed robust every10 and new best/last',
 'no_external_task_labels':True}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,a):(OUT/n).write_text(json.dumps(a,ensure_ascii=False,indent=2),encoding='utf-8')
def allowed_guard(stage):
    allowed={ROOT/'work/data/train_data.mat',ROOT/'work/data/train_label.mat',BASE/'源图分折.json',BASE/'仅源交叉验证分数.npz',BASE/'A.txt'}
    if stage=='predict':allowed.add(ROOT/'work/data/test_data.mat')
    attempts=[]
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        p=Path(args[0]).resolve()
        if not p.is_relative_to(ROOT) or p.suffix.lower() not in ['.mat','.npz','.pt','.pth','.json','.csv','.txt','.pkl']:return
        mode=args[1] if len(args)>1 else None
        if isinstance(mode,str) and any(k in mode for k in 'wax+'):return
        good=p in allowed or p.is_relative_to(OUT) or p.is_relative_to(BASE/'baseline_models') or p.is_relative_to(ROOT/'work/cnn_env')
        if not good:attempts.append(str(p));raise RuntimeError('Disallowed task-data access '+str(p))
    sys.addaudithook(hook);return attempts
def freeze_bn(net,kind):
    if kind=='densenet121':
        for p in net.modules():
            if isinstance(p,nn.BatchNorm2d):p.eval()
def build(kind,parent):
    net=e.build(kind);ck=torch.load(parent,map_location='cpu',weights_only=True);net.load_state_dict(ck['state_dict']);return net,ck
def phase1(tag,kind,seed):return BASE/f'baseline_models/{tag}_{kind}_{seed}_base_stage1.pt'
def path_for(tag,variant,kind,seed):return OUT/'models'/f'{tag}_{variant}_{kind}_{seed}.pt'

# Fixed eval views share the exact frozen DenseNet encoder. Reuse only its
# outputs (never trainable block4/head); bound host RAM and check logits.
ORIGINAL_PREDICT=e.predict
FEATURE_CACHE=OrderedDict();CACHE_BYTES=0;CACHE_CHECKS=[]
@torch.no_grad()
def efficient_predict(net,images,kind):
    global CACHE_BYTES
    if kind!='densenet121':return ORIGINAL_PREDICT(net,images,kind)
    net.eval()
    if not hasattr(net,'_frozen_cache_sha'):
        hh=hashlib.sha256()
        for name,v in net.features.state_dict().items():
            if not name.startswith(('denseblock4.','norm5.')):hh.update(name.encode());hh.update(v.detach().cpu().numpy().tobytes())
        net._frozen_cache_sha=hh.hexdigest()
    digest=hashlib.sha256(images.numpy().tobytes()).hexdigest();key=(net._frozen_cache_sha,digest,tuple(images.shape))
    if key in FEATURE_CACHE:
        encoded=FEATURE_CACHE.pop(key);FEATURE_CACHE[key]=encoded
    else:
        encoder=nn.Sequential(*list(net.features.children())[:-2]);encoder.eval()
        encoded=torch.cat([encoder(e.n.prep(b.to(DEVICE),kind)).cpu() for b in images.split(64)])
        size=encoded.numel()*encoded.element_size()
        while FEATURE_CACHE and CACHE_BYTES+size>320*1024**2:
            _,old=FEATURE_CACHE.popitem(last=False);CACHE_BYTES-=old.numel()*old.element_size()
        if size<=320*1024**2:FEATURE_CACHE[key]=encoded;CACHE_BYTES+=size
    logits=[]
    for h in encoded.split(64):
        a=net.features.norm5(net.features.denseblock4(h.to(DEVICE)));a=F.relu(a,inplace=True);a=F.adaptive_avg_pool2d(a,(1,1)).flatten(1);logits.append(net.classifier(a).flatten())
    out=torch.cat(logits)
    if not getattr(net,'_cache_verified',False):
        direct=net(e.n.prep(images[:min(64,len(images))].to(DEVICE),kind)).flatten();error=float((direct-out[:len(direct)]).abs().max().cpu());assert error<2e-5,error
        CACHE_CHECKS.append({'frozen_encoder_sha':net._frozen_cache_sha,'max_logit_difference':error});net._cache_verified=True
        dump('冻结特征缓存核验.json',CACHE_CHECKS)
    return out.sigmoid().cpu().numpy()
e.predict=efficient_predict

def base_view(x):return e.u.weak(x).mul(.8+torch.rand(len(x),1,1,1)*.4).add((torch.rand(len(x),1,1,1)-.5)*.10).clamp(0,1)
def motion(x,angle,length):
    r=length//2;k=torch.zeros(1,1,length,length);t=torch.linspace(-r,r,length*4+1);ys=(r+t*math.sin(angle)).round().long().clamp(0,length-1);xs=(r+t*math.cos(angle)).round().long().clamp(0,length-1)
    k[0,0,ys,xs]=1;k/=k.sum();return F.conv2d(F.pad(x,(r,)*4,mode='reflect'),k)
def corrupt(clean):
    out=clean.clone();typ=torch.randint(5,(len(out),));strength=random.uniform(.5,1.)
    for t in range(5):
        ids=torch.where(typ==t)[0]
        if not len(ids):continue
        a=out[ids]
        if t in [1,3]:a=e.u.blur(a,random.choice([.5,.8,1.2,1.6])*strength)
        if t==2:a=motion(a,random.uniform(0,math.pi),random.choice([3,5]))
        if t in [3,4]:
            sz=random.choice([8,10,12,15]);mode=random.choice(['bilinear','area'])
            a=F.interpolate(a,(sz,sz),mode=mode,**({'align_corners':False} if mode=='bilinear' else {}));a=F.interpolate(a,(19,19),mode='bilinear',align_corners=False)
        # Both noise-before-filter and filter-before-noise chains occur.
        sigma=torch.tensor([0.,.005,.01,.02,.03])[torch.randint(5,(len(ids),))][:,None,None,None]*strength
        contrast=torch.tensor([1.,.75,.5,.35])[torch.randint(4,(len(ids),))][:,None,None,None]
        if random.random()<.5:
            a=((a-.5)*contrast+.5+torch.randn_like(a)*sigma).clamp(0,1)
            if random.random()<.35:a=e.u.blur(a,random.choice([.4,.7]))
        else:
            if random.random()<.25:a=e.u.blur(a,random.choice([.4,.7]))
            a=((a-.5)*contrast+.5+torch.randn_like(a)*sigma).clamp(0,1)
        if random.random()<.2:a=e.jpeg(a,random.choice([45,65,85]))
        out[ids]=a
    return out.clamp(0,1)
def held_views(x):
    t=torch.arange(-2,3).float();disk=((t[:,None]**2+t[None,:]**2)<=1.5**2).float();disk/=disk.sum();disk=F.conv2d(F.pad(x,(2,)*4,mode='reflect'),disk[None,None])
    rng=torch.Generator().manual_seed(8813);uniform=torch.rand(x.shape,generator=rng);imp=x.clone();imp[uniform<.01]=0;imp[uniform>.99]=1
    speckle=(e.u.blur(x,1.2)*(1+.10*torch.randn(x.shape,generator=rng))).clamp(0,1)
    return torch.cat([disk,e.jpeg(x,25),imp,speckle])
def metrics(p,y):
    q=p.reshape(-1,len(y));n=len(q);d=q>=.37;t=y==1;a=(d==t).mean(1);r=d[:,t].mean(1);s=(~d[:,~t]).mean(1)
    pp=q.clip(1e-6,1-1e-6);b=-(t*np.log(pp)+(1-t)*np.log(1-pp)).mean(1);w=e.VW if n==16 else np.ones(n)/n
    return {'accuracy':float(w@a),'recall':float(w@r),'specificity':float(w@s),'clean_accuracy':float(a[0]),'worst_accuracy':float(a.min()),'worst_recall':float(r.min()),'bce':float(w@b),'views':[{'name':VIEW_NAMES[i] if n==20 else (e.VIEWS[i] if n==16 else VIEW_NAMES[16+i]),'accuracy':float(aa),'recall':float(rr),'specificity':float(ss)} for i,(aa,rr,ss) in enumerate(zip(a,r,s))]}
def metrics20(p,y):return {'standard':metrics(p[:16],y),'held':metrics(p[16:],y)}
def js(probs):
    p=probs.clamp(1e-5,1-1e-5);dis=torch.stack([1-p,p],-1);mean=dis.mean(0,keepdim=True)
    return (dis*(dis.log()-mean.log())).sum(-1).mean()
class Aux(nn.Module):
    def __init__(self,ch):super().__init__();self.proj=nn.Conv2d(ch,16,1);self.last=nn.Conv2d(16,1,3,padding=1)
    def forward(self,h):return self.last(F.interpolate(F.relu(self.proj(h)),(19,19),mode='bilinear',align_corners=False)).sigmoid()

def choose(rows,baseline):
    acceptable=[r for r in rows if r['clean_accuracy']>=baseline['clean_accuracy']-.01 and r['specificity']>=baseline['specificity']-.005]
    pool=acceptable or rows
    return sorted(pool,key=lambda r:(-round(r['accuracy'],12),-round(r['worst_recall'],12),r['bce'],r['variant']))[0],bool(acceptable)
def save_csv(rows):
    if not rows:return
    with (OUT/'逐轮训练验证.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

ALL_LOGS=[];TEACHER_CACHE={}
def fit(variant,kind,seed,tag,x,y,fitids,valids,teacher_kind,base_objective=None,epochs=None):
    path=path_for(tag,variant,kind,seed)
    if path.exists():
        ck=torch.load(path,map_location='cpu',weights_only=True);assert ck['fit_ids']==(fitids+1).tolist() and ck['validation_ids']==([] if valids is None else (valids+1).tolist());print('REUSE',path.name,flush=True);return ck['record']
    objective=base_objective if variant=='A4' else variant;parent=phase1(tag,kind,seed);e.u.seed(seed+14000);net,ck=build(kind,parent)
    assert ck['fit_ids']==(fitids+1).tolist() and ck['validation_ids']==([] if valids is None else (valids+1).tolist())
    net.to(DEVICE);epochs=epochs or PLAN['epochs_max'][kind];decoder=None;feature_box={};hook=None
    if variant=='A4':
        rs=torch.get_rng_state();torch.manual_seed(seed+900);decoder=Aux({'resnet8':64,'hires19':32,'densenet121':1024}[kind]).to(DEVICE);torch.set_rng_state(rs)
        module=net.blocks if kind=='resnet8' else net.features
        def capture(m,inp,out):feature_box['h']=out
        hook=module.register_forward_hook(capture)
    teachers=[]
    if objective=='A3':
        for ts in SEEDS[teacher_kind]:
            teacher,tck=build(teacher_kind,phase1(tag,teacher_kind,ts));assert tck['fit_ids']==ck['fit_ids'] and tck['validation_ids']==ck['validation_ids'];teacher.to(DEVICE).eval()
            for z in teacher.parameters():z.requires_grad=False
            teachers.append(teacher)
    params=[z for z in net.parameters() if z.requires_grad]+([] if decoder is None else list(decoder.parameters()))
    opt=torch.optim.AdamW(params,lr=PLAN['learning_rate'][kind],weight_decay=.001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    ty=torch.from_numpy((y==1).astype('float32'));rng=np.random.default_rng(seed+226);bs=32 if kind=='densenet121' else 64
    vi=e.validation(x[valids]) if valids is not None else None;best=float('inf');beststate=None;auxstate=None;be=epochs;runlogs=[];start_run=time.monotonic()
    print('START',tag,variant,kind,seed,'objective',objective,'epochs',epochs,flush=True)
    for epoch in range(1,epochs+1):
        net.train();freeze_bn(net,kind);sums=np.zeros(5);seen=0;eligible=0;lr=opt.param_groups[0]['lr'];t0=time.monotonic()
        for bi,ids in enumerate(np.array_split(rng.permutation(fitids),int(np.ceil(len(fitids)/bs)))):
            random.seed(seed+200000+epoch*1000+bi);torch.manual_seed(seed+200000+epoch*1000+bi)
            clean=base_view(x[ids]);d1=corrupt(clean);d2=corrupt(clean);images=torch.cat([clean,d1,d2]);target=ty[ids].to(DEVICE);logit=net(e.n.prep(images.to(DEVICE),kind)).reshape(3,len(ids))
            ce=F.binary_cross_entropy_with_logits(logit,target[None,:].expand_as(logit));j=logit.new_zeros(());kd=j;auxloss=j;loss=ce
            if objective=='A2':j=js(logit.sigmoid());loss=loss+.1*j
            if objective=='A3':
                digest=hashlib.sha256(clean.numpy().tobytes()+ids.tobytes()).hexdigest();key=(tag,teacher_kind,seed,epoch,bi,len(ids))
                if key in TEACHER_CACHE:
                    h,pt=TEACHER_CACHE[key];assert h==digest
                else:
                    pt=np.mean([e.predict(t,clean,teacher_kind) for t in teachers],0);TEACHER_CACHE[key]=(digest,pt)
                pt=torch.from_numpy(pt).to(DEVICE).clamp(1e-5,1-1e-5);truth=torch.where(target>.5,pt,1-pt);mask=truth>=.9
                ps=logit[1:].sigmoid().clamp(1e-5,1-1e-5);kl=pt[None]*torch.log(pt[None]/ps)+(1-pt[None])*torch.log((1-pt[None])/(1-ps));kd=(kl*mask[None]).sum()/(2*mask.sum().clamp_min(1));loss=loss+.1*kd;eligible+=int(mask.sum())
            if decoder is not None:
                h=feature_box['h'];recon=decoder(h[len(ids):]);tar=e.u.blur(clean,1.).to(DEVICE).repeat(2,1,1,1);auxloss=F.l1_loss(recon,tar);loss=loss+.1*auxloss
            opt.zero_grad();loss.backward();opt.step();n=len(ids);sums+=np.array([float(v.detach().cpu()) for v in [loss,ce,j,kd,auxloss]])*n;seen+=n
        sched.step();val_bce=None;val_clean=None;train_clean=None;train_robust=None;val_acc=None
        if valids is not None and (epoch==1 or epoch%2==0 or epoch==epochs):
            if hook is not None:feature_box.clear()
            vp=e.predict(net,vi,kind).reshape(16,len(valids));val_bce=e.bce(vp,y[valids]);val_clean=e.bce(vp[:1],y[valids],False);val_acc=metrics(vp,y[valids])['accuracy']
            train_clean=e.bce(e.predict(net,x[fitids],kind),y[fitids],False)
            improved=val_bce<best
            if epoch%10==0 or improved or epoch==epochs:train_robust=e.bce(e.predict(net,e.validation(x[fitids]),kind),y[fitids])
            if improved:
                best=val_bce;be=epoch;beststate={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};auxstate=None if decoder is None else {k:v.detach().cpu().clone() for k,v in decoder.state_dict().items()}
        row={'tag':tag,'variant':variant,'objective':objective,'kind':kind,'seed':seed,'epoch':epoch,'lr':lr,'train_objective':sums[0]/seen,'train_ce':sums[1]/seen,'train_js':sums[2]/seen,'train_kd':sums[3]/seen,'train_aux':sums[4]/seen,'teacher_eligible':eligible,'fit_presentations':seen,'train_clean_bce':train_clean,'train_robust_bce':train_robust,'validation_clean_bce':val_clean,'validation_robust_bce':val_bce,'validation_accuracy':val_acc,'seconds':time.monotonic()-t0}
        row={k:(v.item() if isinstance(v,np.generic) else v) for k,v in row.items()}
        runlogs.append(row);ALL_LOGS.append(row)
        if epoch==1 or epoch%10==0 or epoch==epochs:print('EPOCH',tag,variant,kind,seed,epoch,'ce',round(row['train_ce'],4),'val',None if val_bce is None else round(val_bce,5),'sec',round(row['seconds'],1),flush=True)
        dump('运行进度.json',{'status':'training','current':row,'finished_log_rows':len(ALL_LOGS)})
    if beststate is not None:net.load_state_dict(beststate)
    if hook is not None:hook.remove()
    rec={'tag':tag,'variant':variant,'base_objective':objective,'kind':kind,'seed':seed,'phase':2,'epoch':be,'executed_epochs':epochs,'fit_ids':(fitids+1).tolist(),'validation_ids':[] if valids is None else (valids+1).tolist(),'parent_file':str(parent),'parent_sha256':sha(parent),'teacher_kind':teacher_kind if objective=='A3' else None,'best_validation_bce':best if valids is not None else None,'history':runlogs,'seconds':time.monotonic()-start_run}
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()},'aux_state_dict':auxstate if valids is not None else (None if decoder is None else {k:v.detach().cpu() for k,v in decoder.state_dict().items()}),'record':rec,'fit_ids':rec['fit_ids'],'validation_ids':rec['validation_ids'],'parent_sha256':rec['parent_sha256']},path)
    save_csv(ALL_LOGS);print('DONE',tag,variant,kind,seed,'best',be,'seconds',round(rec['seconds'],1),flush=True)
    del net,opt,decoder,teachers
    if DEVICE=='mps':torch.mps.empty_cache()
    return rec

def baseline_scores(tag,images,phase=2):
    family=[]
    for kind in KINDS:
        pp=[]
        for seed in SEEDS[kind]:
            p=BASE/f'baseline_models/{tag}_{kind}_{seed}_base_stage{phase}.pt';net,_=build(kind,p);net.to(DEVICE);pp.append(e.predict(net,images,kind));del net
        family.append(np.mean(pp,0))
    return np.mean(family,0),family
def variant_scores(tag,variant,images):
    family=[]
    for kind in KINDS:
        pp=[]
        for seed in SEEDS[kind]:
            ck=torch.load(path_for(tag,variant,kind,seed),map_location='cpu',weights_only=True);net=e.build(kind);net.load_state_dict(ck['state_dict']);net.to(DEVICE);pp.append(e.predict(net,images,kind));del net
        family.append(np.mean(pp,0))
    return np.mean(family,0),family

def train():
    attempts=allowed_guard('train')
    if (OUT/'运行前固定方案.json').exists():assert json.loads((OUT/'运行前固定方案.json').read_text())==PLAN
    else:dump('运行前固定方案.json',PLAN)
    folds=json.loads((BASE/'源图分折.json').read_text());x=e.u.image(loadmat(ROOT/'work/data/train_data.mat')['train_data']);y=loadmat(ROOT/'work/data/train_label.mat')['train_label'].ravel()
    manifest=[]
    for tag in ['fold1','fold2','fold3','full']:
        for kind in KINDS:
            for seed in SEEDS[kind]:
                for phase in [1,2]:
                    p=BASE/f'baseline_models/{tag}_{kind}_{seed}_base_stage{phase}.pt';manifest.append({'file':str(p),'sha256':sha(p)})
    dump('基线权重来源.json',manifest);sourcecv=np.load(BASE/'仅源交叉验证分数.npz');records=[];teachers={};teacher_stats=[];inner={};outer={};saved={}
    # Reuse completed runs if an external interruption occurs.
    if (OUT/'逐轮训练验证.csv').exists():
        with (OUT/'逐轮训练验证.csv').open(encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):ALL_LOGS.append(row)
    for f in folds:
        tag=f'fold{f["fold"]}';fitids=np.array(f['fit_ids'])-1;valids=np.array(f['inner_ids'])-1;outerids=np.array(f['outer_ids'])-1
        assert not set(fitids)&set(valids) and not set(fitids)&set(outerids) and not set(valids)&set(outerids)
        _,ff=baseline_scores(tag,x[valids],phase=1);tm=[e.bce(p,y[valids],False) for p in ff];teachers[tag]=KINDS[int(np.argmin(tm))];teacher_stats.append({'tag':tag,'clean_bce_by_family':dict(zip(KINDS,tm)),'selected':teachers[tag]})
        vi=e.validation(x[valids]);vo=torch.cat([e.validation(x[outerids]),held_views(x[outerids])]);a0i=sourcecv[f'{tag}_inner_A'].reshape(16,len(valids));a0o,_=baseline_scores(tag,vo)
        inner[tag]={'A0':metrics(a0i,y[valids])};outer[tag]={'A0':metrics20(a0o.reshape(20,len(outerids)),y[outerids])};saved[f'{tag}_A0_outer']=a0o;saved[f'{tag}_A0_inner']=a0i.flatten()
        dump('教师选择.json',teacher_stats)
    # Complete A1 then A2 then A3 before the source-selected structural A4.
    for variant in ['A1','A2','A3']:
        for f in folds:
            tag=f'fold{f["fold"]}';fitids=np.array(f['fit_ids'])-1;va=np.array(f['inner_ids'])-1;ot=np.array(f['outer_ids'])-1
            for kind in KINDS:
                for seed in SEEDS[kind]:records.append(fit(variant,kind,seed,tag,x,y,fitids,va,teachers[tag]))
            ip,_=variant_scores(tag,variant,e.validation(x[va]));op,_=variant_scores(tag,variant,torch.cat([e.validation(x[ot]),held_views(x[ot])]))
            inner[tag][variant]=metrics(ip.reshape(16,len(va)),y[va]);outer[tag][variant]=metrics20(op.reshape(20,len(ot)),y[ot]);saved[f'{tag}_{variant}_inner']=ip;saved[f'{tag}_{variant}_outer']=op
            dump('内部验证与外层进度.json',{'inner':inner,'outer':outer});np.savez_compressed(OUT/'源图评分_进行中.npz',**saved);TEACHER_CACHE.clear()
    a4base={}
    for f in folds:
        tag=f'fold{f["fold"]}';rows=[{'variant':v,**inner[tag][v]} for v in ['A1','A2','A3']];sel,passed=choose(rows,inner[tag]['A0']);a4base[tag]=sel['variant'];dump('A4配方选择.json',a4base)
        fitids=np.array(f['fit_ids'])-1;va=np.array(f['inner_ids'])-1;ot=np.array(f['outer_ids'])-1
        for kind in KINDS:
            for seed in SEEDS[kind]:records.append(fit('A4',kind,seed,tag,x,y,fitids,va,teachers[tag],a4base[tag]))
        ip,_=variant_scores(tag,'A4',e.validation(x[va]));op,_=variant_scores(tag,'A4',torch.cat([e.validation(x[ot]),held_views(x[ot])]))
        inner[tag]['A4']=metrics(ip.reshape(16,len(va)),y[va]);outer[tag]['A4']=metrics20(op.reshape(20,len(ot)),y[ot]);saved[f'{tag}_A4_inner']=ip;saved[f'{tag}_A4_outer']=op
        dump('内部验证与外层进度.json',{'inner':inner,'outer':outer});TEACHER_CACHE.clear()
    means={v:{k:float(np.mean([inner[f'fold{i}'][v][k] for i in [1,2,3]])) for k in ['accuracy','recall','specificity','clean_accuracy','worst_recall','bce']} for v in ['A0','A1','A2','A3','A4']}
    chosen,passed=choose([{'variant':v,**means[v]} for v in ['A1','A2','A3']],means['A0']);a4base['full']=chosen['variant']
    means_teacher={k:np.mean([r['clean_bce_by_family'][k] for r in teacher_stats]) for k in KINDS};teachers['full']=min(means_teacher,key=means_teacher.get)
    full_epochs={v:{k:int(np.floor(np.median([r['epoch'] for r in records if r['variant']==v and r['kind']==k])+.5)) for k in KINDS} for v in VARIANTS}
    dump('源数据锁定配方.json',{'A4_base':a4base,'full_epochs':full_epochs,'teachers':teachers,'inner_means':means})
    for variant in VARIANTS:
        for kind in KINDS:
            for seed in SEEDS[kind]:records.append(fit(variant,kind,seed,'full',x,y,np.arange(800),None,teachers['full'],a4base['full'] if variant=='A4' else None,full_epochs[variant][kind]))
        TEACHER_CACHE.clear()
    final=[{'file':str(path_for('full',v,k,s).relative_to(OUT)),'sha256':sha(path_for('full',v,k,s))} for v in VARIANTS for k in KINDS for s in SEEDS[k]]
    dump('全部训练记录.json',records);np.savez_compressed(OUT/'源图内部与外层评分.npz',**saved)
    assert all(sha(Path(r['file']))==r['sha256'] for r in manifest)
    lock={'variants':VARIANTS,'threshold':.37,'A4_base':a4base,'full_epochs':full_epochs,'teachers':teachers,'inner_means':means,'code_sha256':sha(__file__),'plan_sha256':sha(OUT/'运行前固定方案.json'),'final_models':final,'source_weight_manifest_sha256':sha(OUT/'基线权重来源.json'),'source_data_sha256':{n:sha(ROOT/'work/data'/n) for n in ['train_data.mat','train_label.mat']},'forbidden_attempts':attempts,'source_only_training':True}
    dump('模型锁定.json',lock);dump('运行进度.json',{'status':'complete','runs':len(records),'rows':len(ALL_LOGS)});print('LOCKED',len(records),full_epochs,flush=True)

def predict():
    attempts=allowed_guard('predict');lock=json.loads((OUT/'模型锁定.json').read_text());assert lock['code_sha256']==sha(__file__) and lock['plan_sha256']==sha(OUT/'运行前固定方案.json');before=sha(OUT/'模型锁定.json')
    assert all(sha(OUT/r['file'])==r['sha256'] for r in lock['final_models'])
    x=e.u.image(loadmat(ROOT/'work/data/test_data.mat')['test_data']);scores={};files=[]
    a0,_=baseline_scores('full',x);scores['A0']=a0
    for v in VARIANTS:scores[v],_=variant_scores('full',v,x)
    for v,p in scores.items():
        fn=OUT/f'result_{v}.txt';fn.write_bytes(''.join(f'{i} {int(q)}\n' for i,q in enumerate(np.where(p>=.37,1,-1),1)).encode());files.append({'variant':v,'file':fn.name,'sha256':sha(fn)})
    assert (OUT/'result_A0.txt').read_bytes()==(BASE/'A.txt').read_bytes();np.savez_compressed(OUT/'锁定后预测分数.npz',**scores)
    assert sha(OUT/'模型锁定.json')==before;dump('预测核验.json',{'files':files,'A0_matches_original_A_byte_for_byte':True,'lock_unchanged':True,'forbidden_attempts':attempts,'manual_overrides':False});print('PREDICTED',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['train','predict'],required=True);arg=ap.parse_args();{'train':train,'predict':predict}[arg.stage]()
