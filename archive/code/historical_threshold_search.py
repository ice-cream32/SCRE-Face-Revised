"""Two new small CNN families, each clean stage1 -> degraded stage2.
Selection/complementarity only on source-validation originals and views.
Predict is a separate post-lock command. No target labels or historic scores.
"""
from pathlib import Path
import sys,argparse,json,hashlib,itertools
import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.nn import functional as F
import model_core as m
import augmentation_core as aug

OUT=Path(__file__).resolve().parent;ROOT=OUT.parent.parent;SRC=ROOT/'outputs/两阶段退化训练'
NEW=['hires19','aa_res8'];SEEDS=[42,7,2026]
PLAN={'new_architectures':NEW,'seeds':SEEDS,'stage1_epochs':80,'stage2_epochs':40,
      'lr_stage1':.001,'lr_stage2':.0001,'weight_decay':.001,
      'stage1':'original fit images + basic augmentation; no degradation',
      'stage2':'continue stage1; same 50% degradation policy as preceding two-stage run',
      'validation_views':['clean','blur1','blur1.5','blur2','down12','down7','motion5','contrast0.3'],
      'validation_weights':[.5]+[.5/7]*7,'threshold_grid':[i/100 for i in range(1,100)],
      'threshold_rule':'weighted accuracy, tie nearest0.5 then smaller',
      'fusion_weights':[.25,.5,.75],
      'fusion_controls':'all nonempty family subsets; equal weights for >=3; all pairs use .25/.5/.75; add baseline+new controls',
      'numeric_tie_rounding_decimals':12,
      'candidate_sort':'weighted accuracy, worst-view accuracy, weighted BCE, network count, candidate name',
      'test_reference_access':False,'training_test_images_used':False,
      'paper_scope':'design motivation only; custom hires19 is not original HRNet; BlurPool is not deblurring'}
def dump(n,o):(OUT/n).write_text(json.dumps(o,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def guard(stage):
    trusted={SRC/'两阶段模型锁定.json',SRC/'继承的训练集内部划分.json',SRC/'第二阶段预先固定方案.json'}
    trusted|={SRC/'models'/f'{k}_seed{s}.pt' for k in m.KINDS for s in ([42] if k=='densenet121' else SEEDS)}
    attempts=[]
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        p=Path(args[0]).resolve();bad=stage=='train' and p==ROOT/'work/data/test_data.mat'
        bad|=p.is_relative_to(ROOT/'outputs') and not p.is_relative_to(OUT) and p not in trusted
        bad|=any(p.is_relative_to(ROOT/q) for q in ['work/annotation','work/top5','work/review_v2','work/review_v3','work/ablation1764','work/simplified_ablation'])
        if bad:attempts.append(str(p));raise RuntimeError('BLOCKED historical labels/test access '+str(p))
    sys.addaudithook(hook);return attempts
class DilatedBlock(nn.Module):
    def __init__(self,ch,d):
        super().__init__();self.main=nn.Sequential(nn.Conv2d(ch,ch,3,padding=d,dilation=d,bias=False),nn.GroupNorm(4,ch),nn.ReLU(),nn.Conv2d(ch,ch,3,padding=d,dilation=d,bias=False),nn.GroupNorm(4,ch))
    def forward(self,x):return F.relu(x+self.main(x))
class HiRes19(nn.Module):
    def __init__(self):
        super().__init__();self.features=nn.Sequential(m.conv(1,32),DilatedBlock(32,1),DilatedBlock(32,2),DilatedBlock(32,3));self.head=nn.Sequential(nn.Linear(32*10,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,1))
    def forward(self,x):
        h=self.features(x);height,width=h.shape[-2:]
        # Exact adaptive-average bins; avoids MPS's non-divisible 19->3 limitation.
        spatial=torch.stack([h[:,:,i*height//3:((i+1)*height+2)//3,j*width//3:((j+1)*width+2)//3].mean((2,3)) for i in range(3) for j in range(3)],dim=-1).flatten(1)
        z=torch.cat([h.mean((2,3)),spatial],dim=1);return self.head(z).reshape(-1)
class BlurPool(nn.Module):
    def __init__(self,channels):
        super().__init__();v=torch.tensor([1.,2.,1.]);self.register_buffer('kernel',((v[:,None]*v[None,:])/16)[None,None].repeat(channels,1,1,1));self.channels=channels
    def forward(self,x):return F.conv2d(F.pad(x,(1,1,1,1),mode='reflect'),self.kernel,stride=2,groups=self.channels)
class AARes8(nn.Module):
    def __init__(self):
        super().__init__();self.features=nn.Sequential(m.conv(1,16),m.Residual(16,16,1),m.Residual(16,32,1),BlurPool(32),m.Residual(32,64,1),BlurPool(64));self.head=nn.Sequential(nn.Flatten(),nn.Dropout(.2),nn.Linear(64*5*5,1))
    def forward(self,x):return self.head(self.features(x)).reshape(-1)
def build(kind):return HiRes19() if kind=='hires19' else AARes8()
def motion(x):
    k=x.new_zeros(1,1,5,5);k[0,0,2,:]=.2;return F.conv2d(F.pad(x,(2,2,2,2),mode='reflect'),k)
def views(x):return [x,aug.blur(x,1),aug.blur(x,1.5),aug.blur(x,2),aug.down(x,12),aug.down(x,7),motion(x),x*.3+.35]
def bce_views(p,y):
    p=p.reshape(len(PLAN['validation_views']),len(y));q=p.clip(1e-7,1-1e-7);target=(y==1)[None,:]
    loss=-(target*np.log(q)+(1-target)*np.log(1-q)).mean(axis=1);return float(np.dot(PLAN['validation_weights'],loss))
def stats(p,y,t):
    p=p.reshape(8,len(y));pred=np.where(p>=t,1,-1);rs=[]
    for name,v in zip(PLAN['validation_views'],pred):rs.append({'view':name,'accuracy':float(np.mean(v==y)),'face_recall':float(np.mean(v[y==1]==1)),'nonface_specificity':float(np.mean(v[y==-1]==-1))})
    return {'weighted_accuracy':round(float(np.dot(PLAN['validation_weights'],[r['accuracy'] for r in rs])),12),'worst_view_accuracy':min(r['accuracy'] for r in rs),'weighted_logloss':bce_views(p,y),'views':rs}
def choose_threshold(p,y):
    acc=((p.reshape(8,len(y))[None,:,:]>=np.array(PLAN['threshold_grid'])[:,None,None])==((y==1)[None,None,:])).mean(axis=2)@np.array(PLAN['validation_weights'])
    i=min(range(len(acc)),key=lambda j:(-round(float(acc[j]),12),abs(PLAN['threshold_grid'][j]-.5),PLAN['threshold_grid'][j]));return PLAN['threshold_grid'][i]
def fit(kind,seed,phase,x,y,tr,va,valx):
    m.seed(seed+7000*phase);net=build(kind)
    if phase==2:
        prior=OUT/'stage1_models'/f'{kind}_seed{seed}.pt';saved=torch.load(prior,map_location='cpu',weights_only=True);net.load_state_dict(saved['state_dict'])
    net.to(m.DEVICE);epochs=PLAN[f'stage{phase}_epochs'];lr=PLAN[f'lr_stage{phase}'];opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=PLAN['weight_decay']);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    ty=torch.from_numpy((y==1).astype('float32'));best=float('inf');state=None;be=0;history=[]
    for ep in range(epochs):
        net.train()
        for ids in np.array_split(np.random.permutation(tr),10):
            batch=m.augment(x[ids].to(m.DEVICE));batch=aug.degrade(batch) if phase==2 else batch
            z=net(m.prep(batch,kind));loss=F.binary_cross_entropy_with_logits(z,ty[ids].to(m.DEVICE));opt.zero_grad();loss.backward();opt.step()
        sched.step()
        if phase==1:
            pp=m.predict(net,x[va],kind);v=pp.clip(1e-7,1-1e-7);target=y[va]==1;vl=float(-(target*np.log(v)+(1-target)*np.log(1-v)).mean())
        else:pp=m.predict(net,valx,kind);vl=bce_views(pp,y[va])
        history.append(vl)
        if vl<best:best=vl;be=ep+1;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    net.load_state_dict(state);p=m.predict(net,valx,kind);path=OUT/f'stage{phase}_models'/f'{kind}_seed{seed}.pt'
    row={'kind':kind,'seed':seed,'phase':phase,'best_epoch':be,'validation_loss':best,'lr':lr,'parameters':sum(v.numel() for v in net.parameters()),'fit_ids_1based':(tr+1).tolist(),'validation_ids_1based':(va+1).tolist(),'history':history,'parent_sha256':sha(prior) if phase==2 else None}
    torch.save({**row,'state_dict':state},path);del net
    print('TRAINED',kind,seed,'phase',phase,'epoch',be,'loss',best,flush=True);return p,row
def old_experts(valx):
    original=json.loads((SRC/'两阶段模型锁定.json').read_text())
    for ck in original['stage2_checkpoints']:assert sha(SRC/ck['file'])==ck['sha256']
    groups={};all_scores={}
    for kind in m.KINDS:
        ps=[]
        for seed in ([42] if kind=='densenet121' else SEEDS):
            n=f'{kind}_seed{seed}';saved=torch.load(SRC/'models'/f'{n}.pt',map_location='cpu',weights_only=True);net=m.build(kind,False);net.load_state_dict(saved['state_dict']);net.to(m.DEVICE);p=m.predict(net,valx,kind);ps.append(p);all_scores[n]=p;del net
        groups[kind]=np.mean(ps,axis=0)
    groups['baseline_ten']=m.combine(all_scores)[0];return groups
def make_candidates(base):
    family=m.KINDS+NEW;out={k:({'members':[k],'weights':[1.]},v) for k,v in base.items()}
    for a,b in itertools.combinations(family,2):
        for w in PLAN['fusion_weights']:out[f'pair_{a}_{b}_{w}']=({'members':[a,b],'weights':[w,1-w]},w*base[a]+(1-w)*base[b])
    for a in NEW:
        for w in PLAN['fusion_weights']:out[f'add_baseline_{a}_{w}']=({'members':['baseline_ten',a],'weights':[w,1-w]},w*base['baseline_ten']+(1-w)*base[a])
    for size in range(3,7):
        for ns in itertools.combinations(family,size):
            out['equal__'+'__'.join(ns)]=({'members':list(ns),'weights':[1/size]*size},np.mean([base[n] for n in ns],axis=0))
    return out
def network_count(members):
    names=set()
    for n in members:
        if n=='baseline_ten':names|={f'{k}_{s}' for k in m.KINDS for s in ([42] if k=='densenet121' else SEEDS)}
        else:names|={f'{n}_{s}' for s in ([42] if n=='densenet121' else SEEDS)}
    return len(names)
def train():
    attempts=guard('train');(OUT/'stage1_models').mkdir(exist_ok=True);(OUT/'stage2_models').mkdir(exist_ok=True);dump('训练前架构与选型方案.json',PLAN)
    split=json.loads((SRC/'继承的训练集内部划分.json').read_text());dump('原图ID划分.json',split);tr=np.array(split['fit_ids_1based'])-1;va=np.array(split['validation_ids_1based'])-1
    assert len(tr)==640 and len(va)==160 and not set(tr)&set(va)
    X=loadmat(ROOT/'work/data/train_data.mat')['train_data'];y=loadmat(ROOT/'work/data/train_label.mat')['train_label'].ravel();x=m.image(X);valx=torch.cat(views(x[va]));base=old_experts(valx);records=[]
    first={}
    for kind in NEW:
        p2=[];p1=[]
        for seed in SEEDS:
            p,r=fit(kind,seed,1,x,y,tr,va,valx);records.append(r);p1.append(p)
            p,r=fit(kind,seed,2,x,y,tr,va,valx);records.append(r);p2.append(p)
        base[kind]=np.mean(p2,axis=0);first[kind]=np.mean(p1,axis=0)
    pool=make_candidates(base);ranking=[]
    for name,(recipe,p) in pool.items():
        t=choose_threshold(p,y[va]);ranking.append({'candidate':name,'recipe':recipe,'threshold':t,'networks':network_count(recipe['members']),**stats(p,y[va],t)})
    ranking.sort(key=lambda r:(-r['weighted_accuracy'],-r['worst_view_accuracy'],r['weighted_logloss'],r['networks'],r['candidate']))
    # Error overlap uses only source validation labels. Thresholds fitted per family.
    families=[];errors={}
    for n in base:
        t=choose_threshold(base[n],y[va]);families.append({'family':n,'threshold':t,**stats(base[n],y[va],t)});errors[n]=(np.where(base[n]>=t,1,-1)!=np.tile(y[va],8))
    overlap=[]
    for a,b in itertools.combinations(base,2):
        union=errors[a]|errors[b];inter=errors[a]&errors[b]
        overlap.append({'a':a,'b':b,'error_a':int(sum(errors[a])),'error_b':int(sum(errors[b])),'double_errors':int(sum(inter)),'error_jaccard':float(sum(inter)/sum(union)) if union.any() else 0,'oracle_accuracy_any_correct':float(1-inter.mean())})
    dump('来源验证架构表现.json',families);dump('来源验证错误互补性.json',overlap);dump('来源验证候选排名.json',ranking);dump('新架构两阶段训练记录.json',records)
    winner=ranking[0]
    lock={'selected':winner,'plan_sha256':sha(OUT/'训练前架构与选型方案.json'),'code_sha256':sha(__file__),'model_core_sha256':sha(OUT/'model_core.py'),'augmentation_core_sha256':sha(OUT/'augmentation_core.py'),
          'old_source_lock_sha256':sha(SRC/'两阶段模型锁定.json'),'test_data_read':False,'test_labels_used':False,'blocked_attempts':attempts,
          'new_checkpoints':[{'file':f'stage2_models/{k}_seed{s}.pt','sha256':sha(OUT/'stage2_models'/f'{k}_seed{s}.pt')} for k in NEW for s in SEEDS],
          'family_thresholds':{r['family']:r['threshold'] for r in families},'baseline_fixed_threshold':.5}
    dump('模型锁定.json',lock);np.savez(OUT/'仅训练集内部验证分数.npz',ids=va+1,labels=y[va],**base,**{'stage1_'+k:v for k,v in first.items()})
    print('LOCKED',json.dumps(winner,ensure_ascii=False),flush=True)
def infer():
    guard('predict');lock=json.loads((OUT/'模型锁定.json').read_text());before=sha(OUT/'模型锁定.json')
    assert lock['code_sha256']==sha(__file__) and lock['model_core_sha256']==sha(OUT/'model_core.py') and lock['augmentation_core_sha256']==sha(OUT/'augmentation_core.py')
    assert lock['old_source_lock_sha256']==sha(SRC/'两阶段模型锁定.json')
    for ck in lock['new_checkpoints']:assert ck['sha256']==sha(OUT/ck['file'])
    T=loadmat(ROOT/'work/data/test_data.mat')['test_data'];assert T.shape==(420,361);x=m.image(T);base=old_experts(x)
    for kind in NEW:
        ps=[]
        for seed in SEEDS:
            b=torch.load(OUT/'stage2_models'/f'{kind}_seed{seed}.pt',map_location='cpu',weights_only=True);net=build(kind);net.load_state_dict(b['state_dict']);net.to(m.DEVICE);ps.append(m.predict(net,x,kind));del net
        base[kind]=np.mean(ps,axis=0)
    selected=lock['selected'];recipe=selected['recipe'];p=sum(w*base[n] for n,w in zip(recipe['members'],recipe['weights']))
    cases=[('result_architecture_selected.txt',p,selected['threshold']),('baseline_two_stage_locked.txt',base['baseline_ten'],.5)]
    cases += [(f'{n}_two_stage.txt',base[n],lock['family_thresholds'][n]) for n in NEW]
    manifest=[]
    for file,prob,t in cases:
        pred=np.where(prob>=t,1,-1);blob=''.join(f'{i} {int(v)}\n' for i,v in enumerate(pred,1)).encode('ascii');(OUT/file).write_bytes(blob)
        assert np.array_equal(np.loadtxt(OUT/file,dtype=int),np.c_[np.arange(1,421),pred]);manifest.append({'file':file,'threshold':t,'sha256':sha(OUT/file),'positive_count':int(sum(pred==1))})
    assert before==sha(OUT/'模型锁定.json');dump('锁定后预测核验.json',{'lock_unchanged':True,'test_labels_used':False,'manual_overrides':False,'files':manifest})
    print(json.dumps(manifest,ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['train','predict'],required=True)
    train() if p.parse_args().stage=='train' else infer()
