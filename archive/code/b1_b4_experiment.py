"""Source-only B0--B4 experiments with nested threshold calibration.

Training and target inference run in separate processes. No historical target
labels or target score files are allowed in the training process.
"""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, math, random, sys, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.io import loadmat

ROOT = Path('${ORIGINAL_WORKSPACE}')
PREV = ROOT/'outputs/A1-A4模糊鲁棒性对照'
BASE = ROOT/'outputs/七网络原图通道与条件蒸馏'
OUT = ROOT/'outputs/B1-B4判别鲁棒性与独立阈值'
OUT.mkdir(exist_ok=True); (OUT/'models').mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location('a_experiment', ROOT/'work/a1_a4_experiment.py')
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
# Reuse the exact frozen-encoder cache, but write verification only here.
a.OUT = OUT
e = a.e; DEVICE = a.DEVICE; KINDS = a.KINDS; SEEDS = a.SEEDS
NEW = ['B2', 'B3', 'B4']
NAMES = ['B0', 'B0_T', 'A1_T', 'B1', 'B2', 'B3', 'B4']
PLAN = {
 'baseline': 'B0 exact historical A0 weights and threshold0.37; B0_T same scores, source-recalibrated threshold',
 'A1_T': 'prior A1 exact weights, source-recalibrated threshold; no new expert training',
 'B1': 'convex A0/A1 score fusion; alpha grid0,0.1,...,1 and independent exact threshold search',
 'B2': 'same A1 clean/two-corruption CE; top25% degraded negatives per batch have capped weight2; normalize by total weights',
 'B3': 'same A1 CE plus0.1 feature triplet hinge, margin0.2; paired same-source positives and hardest opposite-class negatives; both classes',
 'B4': 'inner-only choose B1/B2/B3 base; restart stage1 using its objective, progressive corruption mixing and bounded four-group DRO; source-calibrated A0 fusion and constrained threshold',
 'B4_curriculum': 'd=(1-strength)c+strength*T(c); strength0.35 to0.85 during first half, then0.85; both corrupted views and both classes',
 'B4_group': 'four groups: class x normal/degraded; q <- normalize(q*exp(0.01*detached_group_loss)), then0.95q+0.05/4; objective0.5CE+0.5q-weighted groupCE',
 'B4_constraints': 'inner clean balanced accuracy>=B0_T-0.01 and weighted specificity>=B0_T-0.005; A0 alpha0 guarantees a feasible option',
 'B4_base_selection': 'inner calibrated balanced accuracy desc, worst-view balanced accuracy desc, clean balanced accuracy desc, simpler variant',
 'threshold_search': 'all label partitions attainable with threshold in[0,1], midpoint of each interval, plus endpoints; weighted balanced accuracy; tie worst-viewBA, cleanBA, interval width, proximity0.5',
 'alpha_grid': [i/10 for i in range(11)],
 'source_views': e.VIEWS, 'source_view_weights': e.VW.tolist(),
 'held_views': a.VIEW_NAMES[16:], 'held_selection': False,
 'partition': 'same3 original-source outer folds; fit426/427, inner107; all augmentation after split',
 'stage1': 'reuse exactly matched source-only stage1; each new run restarts there, no third expert stage',
 'epochs_max': {'resnet8':40, 'hires19':40, 'densenet121':20},
 'learning_rate': {'resnet8':1e-4,'hires19':1e-4,'densenet121':3e-5},
 'batch': {'small':64,'dense':32}, 'optimizer': 'AdamW wd0.001; cosine schedule',
 'schedule_horizon': 'cosine Tmax=epochs_max per family in both folds and full refit; stop full at selected epochs, do not compress LR schedule',
 'curriculum_horizon': 'B4 strength ramps over first half of epochs_max, also in full refit; correction recorded before first full run',
 'checkpoint': 'inner16-view weighted BCE at epoch1/even/last, independent of threshold search',
 'fusion_experts': 'three seed means per small family, DenseNet single seed; family probability mean1/3',
 'contrast_head': 'GAP plus3x3 exact spatial bins -> Linear64 LayerNorm ReLU Linear64 -> unit norm; discarded at inference',
 'final_thresholds': 'concatenate3 inner107 score sets, source labels only; outer scores never used for fitting thresholds/alpha/base/epochs',
 'full_epochs': 'per-family median source-selected stage2 epochs; fixed-length full800 fit',
 'paired_rng': 'same seed/epoch/batch images for B2/B3; B4 adds curriculum after matched corruption stream',
 'target_labels_read':False, 'target_scores_used_for_selection':False,
}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name, value): (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
def guard(stage):
    allowed={ROOT/'work/data/train_data.mat',ROOT/'work/data/train_label.mat',BASE/'源图分折.json',PREV/'源图内部与外层评分.npz',BASE/'A.txt'}
    if stage=='predict': allowed.add(ROOT/'work/data/test_data.mat')
    attempts=[]
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)): return
        p=Path(args[0]).resolve(); mode=args[1] if len(args)>1 else None
        if isinstance(mode,str) and any(t in mode for t in 'wax+'): return
        if not p.is_relative_to(ROOT) or p.suffix.lower() not in ['.mat','.npz','.pt','.pth','.json','.csv','.txt','.pkl']: return
        good=p in allowed or p.is_relative_to(OUT) or p.is_relative_to(BASE/'baseline_models') or p.is_relative_to(ROOT/'work/cnn_env')
        good |= p.parent==PREV/'models' and '_A1_' in p.name and p.suffix=='.pt'
        if not good: attempts.append(str(p)); raise RuntimeError('Forbidden task-data read '+str(p))
    sys.addaudithook(hook); return attempts

def metrics(p,y,t):
    p=np.asarray(p,dtype=np.float64).reshape(-1,len(y)); d=p>=t; truth=y==1
    rec=d[:,truth].mean(1); sp=(~d[:,~truth]).mean(1); acc=(d==truth).mean(1); ba=(rec+sp)/2
    w=e.VW if len(p)==16 else np.ones(len(p))/len(p)
    pp=p.clip(1e-6,1-1e-6); ce=-(truth*np.log(pp)+(~truth)*np.log(1-pp)).mean(1)
    return {'threshold':float(t),'balanced_accuracy':float(w@ba),'accuracy':float(w@acc),'recall':float(w@rec),'specificity':float(w@sp),'clean_balanced_accuracy':float(ba[0]),'clean_accuracy':float(acc[0]),'worst_balanced_accuracy':float(ba.min()),'worst_recall':float(rec.min()),'bce':float(w@ce),
            'views':[{'name':n,'accuracy':float(aa),'balanced_accuracy':float(bb),'recall':float(rr),'specificity':float(ss)} for n,aa,bb,rr,ss in zip(e.VIEWS if len(p)==16 else a.VIEW_NAMES[16:],acc,ba,rec,sp)]}

def threshold_search(p,y,constraint=None):
    """Exact cumulative scan; work in float64 to retain midpoint precision."""
    p=np.asarray(p,dtype=np.float64).reshape(16,len(y)); n=len(y); positive=y==1
    flat=p.ravel(); order=np.argsort(flat,kind='stable'); vals=flat[order]
    unique=np.unique(vals); lo=np.r_[0.,unique[:-1],unique[-1]]; hi=np.r_[unique[0],unique[1:],1.]
    tt=(lo+hi)/2; tt=np.r_[0.,tt,1.]; widths=np.r_[0.,hi-lo,0.]
    ix=np.searchsorted(vals,tt,side='left'); vi=np.repeat(np.arange(16),n)[order]; yy=np.tile(positive,16)[order]
    pos=np.zeros((16,len(flat))); neg=np.zeros_like(pos)
    pos[vi,np.arange(len(flat))]=yy/positive.sum(); neg[vi,np.arange(len(flat))]=(~yy)/(~positive).sum()
    cp=np.c_[np.zeros(16),np.cumsum(pos,axis=1)]; cn=np.c_[np.zeros(16),np.cumsum(neg,axis=1)]
    rec=1-cp[:,ix]; sp=cn[:,ix]; ba=(rec+sp)/2
    score=e.VW@ba; clean=ba[0]; spec=e.VW@sp; worst=ba.min(0)
    ok=np.ones(len(tt),dtype=bool)
    if constraint:
        ok &= clean >= constraint['clean_balanced_accuracy']-.01-1e-12
        ok &= spec >= constraint['specificity']-.005-1e-12
    idx=np.flatnonzero(ok)
    if not len(idx): return None
    # np.lexsort's final key is primary. Original index makes all ties stable.
    rank=np.lexsort((idx,tt[idx],np.abs(tt[idx]-.5),-widths[idx],-np.round(clean[idx],12),-np.round(worst[idx],12),-np.round(score[idx],12)))
    j=int(idx[rank[0]]); t=float(tt[j]); result=metrics(p,y,t)
    result.update({'interval_width':float(widths[j]),'candidate_count':len(tt),'feasible_count':int(ok.sum()),'search_metric':'16-view weighted balanced accuracy'})
    return result

def rankkey(r):
    return (-round(r['balanced_accuracy'],12),-round(r['worst_balanced_accuracy'],12),-round(r['clean_balanced_accuracy'],12),-r.get('interval_width',0),abs(r['threshold']-.5),r.get('alpha',0))
def calibrate(p,y,old=None,constraint=None):
    p=np.asarray(p,dtype=np.float64)
    if old is not None: old=np.asarray(old,dtype=np.float64)
    candidates=[]
    for alpha in (PLAN['alpha_grid'] if old is not None else [1.]):
        q=p if old is None else (1-alpha)*old+alpha*p
        r=threshold_search(q,y,constraint)
        if r is not None: r.update({'alpha':float(alpha)}); candidates.append(r)
    assert candidates, 'No feasible threshold/fusion (alpha0 should allow baseline).'
    return sorted(candidates,key=rankkey)[0]
def calibrated_scores(base,cal,old=None):
    base=np.asarray(base,dtype=np.float64)
    return base if old is None else (1-cal['alpha'])*np.asarray(old,dtype=np.float64)+cal['alpha']*base
def short(r): return {k:v for k,v in r.items() if k!='views'}

def spatial_pool(h):
    H,W=h.shape[-2:]; grid=[]
    for i in range(3):
        for j in range(3): grid.append(h[:,:,i*H//3:((i+1)*H+2)//3,j*W//3:((j+1)*W+2)//3].mean((2,3)))
    return torch.cat([h.mean((2,3)),*grid],dim=1)
class Projection(nn.Module):
    def __init__(self,ch):
        super().__init__(); self.net=nn.Sequential(nn.Linear(ch*10,64),nn.LayerNorm(64),nn.ReLU(),nn.Linear(64,64))
    def forward(self,h): return F.normalize(self.net(spatial_pool(h)),dim=1)
def structure_loss(z,target):
    n=len(target); q=z.reshape(3,n,-1); opposite=target[:,None]!=target[None,:]
    if not opposite.any(): return z.sum()*0
    has=opposite.any(1); terms=[]
    for j in [1,2]:
        sim=q[0]@q[j].T; hardest=sim.masked_fill(~opposite,-2).max(1).values
        terms.append(F.relu(.2+hardest-sim.diag())[has].mean())
    return torch.stack(terms).mean()
def hard_negative_loss(per,target):
    w=torch.ones_like(per); indices=torch.nonzero((target<.5)[None,:].expand(2,-1).reshape(-1),as_tuple=False).flatten()
    k=max(1,int(math.ceil(len(indices)*.25))) if len(indices) else 0
    if k:
        selected=indices[torch.topk(per[1:].detach().reshape(-1)[indices],k).indices]
        # Map flattened degraded-view locations back to the three-view tensor.
        w.view(-1)[selected+len(target)]=2.
    return (w*per).sum()/w.sum(),k,float(w.max().detach().cpu())
def group_loss(per,target,q):
    g=torch.stack([per[0,target==c].mean() for c in [0,1]]+[per[1:,target==c].mean() for c in [0,1]])
    if not torch.isfinite(g).all(): return per.mean(),q
    with torch.no_grad(): q=q*torch.exp(.01*g.detach()); q=q/q.sum(); q=.95*q+.05/4
    return .5*per.mean()+.5*(q*g).sum(),q

LOGS=[]
def save_logs():
    if LOGS:
        with (OUT/'逐轮训练验证.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(LOGS[0])); w.writeheader(); w.writerows(LOGS)
def path_for(tag,v,k,s): return OUT/'models'/f'{tag}_{v}_{k}_{s}.pt'
def fit(v,kind,seed,tag,x,y,ids,valids,base='A1',epochs=None):
    path=path_for(tag,v,kind,seed)
    if path.exists():
        ck=torch.load(path,map_location='cpu',weights_only=True); r=ck['record']
        assert r['fit_ids']==(ids+1).tolist() and r['validation_ids']==([] if valids is None else (valids+1).tolist()) and r['base']==base
        print('REUSE',path.name,flush=True); return r
    parent=BASE/f'baseline_models/{tag}_{kind}_{seed}_base_stage1.pt'; e.u.seed(seed+14000); net,ck=a.build(kind,parent)
    assert ck['fit_ids']==(ids+1).tolist() and ck['validation_ids']==([] if valids is None else (valids+1).tolist())
    net.to(DEVICE); total=epochs or PLAN['epochs_max'][kind]; horizon=PLAN['epochs_max'][kind]; objective=base if v=='B4' else v
    projection=None; hook=None; box={}
    if objective=='B3':
        rng=torch.get_rng_state(); torch.manual_seed(seed+986); projection=Projection({'resnet8':64,'hires19':32,'densenet121':1024}[kind]).to(DEVICE); torch.set_rng_state(rng)
        def capture(module,inp,out): box['h']=out
        hook=(net.blocks if kind=='resnet8' else net.features).register_forward_hook(capture)
    params=[p for p in net.parameters() if p.requires_grad]+([] if projection is None else list(projection.parameters()))
    opt=torch.optim.AdamW(params,lr=PLAN['learning_rate'][kind],weight_decay=.001); schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,horizon)
    rng=np.random.default_rng(seed+226); targets=torch.from_numpy((y==1).astype('float32')); bs=32 if kind=='densenet121' else 64
    vx=e.validation(x[valids]) if valids is not None else None; best=float('inf'); best_state=None; proj_state=None; best_epoch=total; history=[]; started=time.monotonic(); q=torch.full((4,),.25,device=DEVICE)
    print('START',tag,v,kind,seed,'base',base,'epochs',total,flush=True)
    for epoch in range(1,total+1):
        net.train(); a.freeze_bn(net,kind); sums=np.zeros(4); seen=0; hard_count=0; max_weight=1.; started_epoch=time.monotonic(); lr=opt.param_groups[0]['lr']
        strength=.35+.5*min(1,(epoch-1)/max(1,horizon*.5-1)) if v=='B4' else 1.
        for bi,bids in enumerate(np.array_split(rng.permutation(ids),math.ceil(len(ids)/bs))):
            random.seed(seed+200000+epoch*1000+bi); torch.manual_seed(seed+200000+epoch*1000+bi)
            clean=a.base_view(x[bids]); d1=a.corrupt(clean); d2=a.corrupt(clean)
            if v=='B4': d1=(1-strength)*clean+strength*d1; d2=(1-strength)*clean+strength*d2
            target=targets[bids].to(DEVICE); inp=torch.cat([clean,d1,d2]); logits=net(e.n.prep(inp.to(DEVICE),kind)).reshape(3,len(bids))
            per=F.binary_cross_entropy_with_logits(logits,target[None,:].expand_as(logits),reduction='none'); ce=per.mean(); loss=ce; hard=ce*0; ctr=ce*0
            if v=='B4': loss,q=group_loss(per,target,q)
            if objective=='B2':
                hard,k,mx=hard_negative_loss(per,target); loss=loss+(hard-ce); hard_count+=k; max_weight=max(max_weight,mx)
            if objective=='B3':
                ctr=structure_loss(projection(box['h']),target); loss=loss+.1*ctr
            opt.zero_grad(); loss.backward(); opt.step(); n=len(bids); seen+=n; sums+=np.array([float(z.detach().cpu()) for z in [loss,ce,hard,ctr]])*n; box.clear()
        schedule.step(); tc=tr=vc=vr=None
        if valids is not None and (epoch==1 or epoch%2==0 or epoch==total):
            vp=e.predict(net,vx,kind).reshape(16,len(valids)); vr=e.bce(vp,y[valids]); vc=e.bce(vp[:1],y[valids],False); tc=e.bce(e.predict(net,x[ids],kind),y[ids],False)
            improved=vr<best
            if epoch%10==0 or improved or epoch==total: tr=e.bce(e.predict(net,e.validation(x[ids]),kind),y[ids])
            if improved:
                best=vr; best_epoch=epoch; best_state={k:z.detach().cpu().clone() for k,z in net.state_dict().items()}; proj_state=None if projection is None else {k:z.detach().cpu().clone() for k,z in projection.state_dict().items()}
            box.clear()
        row={'tag':tag,'variant':v,'base':base,'kind':kind,'seed':seed,'epoch':epoch,'lr':lr,'train_objective':float(sums[0]/seen),'train_ce':float(sums[1]/seen),'train_hard_weighted_ce':float(sums[2]/seen),'train_structure_hinge':float(sums[3]/seen),'hard_negative_presentations':hard_count,'maximum_sample_weight':max_weight,'curriculum_strength':strength,'group_weights':json.dumps(q.detach().cpu().tolist()),'train_clean_bce':tc,'train_robust_bce':tr,'validation_clean_bce':vc,'validation_robust_bce':vr,'fit_presentations':seen,'seconds':time.monotonic()-started_epoch}
        history.append(row); LOGS.append(row); dump('运行进度.json',{'status':'training','current':row,'rows':len(LOGS)})
        if epoch==1 or epoch%10==0 or epoch==total: print('EPOCH',tag,v,kind,seed,epoch,'CE',round(row['train_ce'],4),'val',None if vr is None else round(vr,5),'sec',round(row['seconds'],1),flush=True)
    if best_state is not None: net.load_state_dict(best_state)
    if hook: hook.remove()
    record={'tag':tag,'variant':v,'base':base,'kind':kind,'seed':seed,'phase':2,'epoch':best_epoch,'executed_epochs':total,'schedule_horizon':horizon,'fit_ids':(ids+1).tolist(),'validation_ids':[] if valids is None else (valids+1).tolist(),'parent_file':str(parent),'parent_sha256':sha(parent),'best_validation_bce':None if valids is None else best,'history':history,'seconds':time.monotonic()-started}
    torch.save({'state_dict':{k:z.detach().cpu() for k,z in net.state_dict().items()},'projection_state_dict':proj_state if valids is not None else (None if projection is None else {k:z.detach().cpu() for k,z in projection.state_dict().items()}),'record':record},path)
    save_logs(); print('DONE',tag,v,kind,seed,'best',best_epoch,'seconds',round(record['seconds'],1),flush=True)
    del net,opt,projection
    if DEVICE=='mps': torch.mps.empty_cache()
    return record

def score_models(tag,v,images):
    fam=[]
    for k in KINDS:
        probs=[]
        for s in SEEDS[k]:
            p=BASE/f'baseline_models/{tag}_{k}_{s}_base_stage2.pt' if v=='B0' else PREV/'models'/f'{tag}_A1_{k}_{s}.pt' if v=='A1' else path_for(tag,v,k,s)
            ck=torch.load(p,map_location='cpu',weights_only=True); net=e.build(k); net.load_state_dict(ck['state_dict']); net.to(DEVICE); probs.append(e.predict(net,images,k)); del net
        fam.append(np.mean(probs,axis=0))
    return np.mean(fam,axis=0)

def pool_inner(scores,folds,name):
    return np.concatenate([scores[f'fold{f["fold"]}_{name}_inner'].reshape(16,len(f['inner_ids'])) for f in folds],axis=1)
def choose_base(calibrations):
    return sorted(['B1','B2','B3'],key=lambda v:(-round(calibrations[v]['balanced_accuracy'],12),-round(calibrations[v]['worst_balanced_accuracy'],12),-round(calibrations[v]['clean_balanced_accuracy'],12),v))[0]
def evaluate(p,y,cal,old=None):
    p=calibrated_scores(p,cal,old); return {'standard':metrics(p[:16],y,cal['threshold']),'held':metrics(p[16:],y,cal['threshold'])}
def train():
    attempts=guard('train')
    if (OUT/'运行前固定方案.json').exists(): assert json.loads((OUT/'运行前固定方案.json').read_text())==PLAN
    else: dump('运行前固定方案.json',PLAN)
    folds=json.loads((BASE/'源图分折.json').read_text()); x=e.u.image(loadmat(ROOT/'work/data/train_data.mat')['train_data']); y=loadmat(ROOT/'work/data/train_label.mat')['train_label'].ravel()
    src=np.load(PREV/'源图内部与外层评分.npz'); scores={}; calibrations={}; outer={}; records=[]; bases={}
    manifest=[]
    for tag in ['fold1','fold2','fold3','full']:
        for k in KINDS:
            for s in SEEDS[k]:
                for phase in [1,2]:
                    p=BASE/f'baseline_models/{tag}_{k}_{s}_base_stage{phase}.pt'; manifest.append({'file':str(p),'sha256':sha(p)})
                p=PREV/'models'/f'{tag}_A1_{k}_{s}.pt'; manifest.append({'file':str(p),'sha256':sha(p)})
    dump('既有权重来源.json',manifest)
    if (OUT/'逐轮训练验证.csv').exists():
        with (OUT/'逐轮训练验证.csv').open(encoding='utf-8-sig') as f: LOGS.extend(csv.DictReader(f))
    for f in folds:
        tag=f'fold{f["fold"]}'; va=np.array(f['inner_ids'])-1; ot=np.array(f['outer_ids'])-1
        assert not set(f['fit_ids'])&set(f['inner_ids']) and not set(f['fit_ids'])&set(f['outer_ids']) and not set(f['inner_ids'])&set(f['outer_ids'])
        for v,old in [('B0','A0'),('A1','A1')]:
            for part,ids in [('inner',va),('outer',ot)]: scores[f'{tag}_{v}_{part}']=src[f'{tag}_{old}_{part}'].reshape(16 if part=='inner' else 20,len(ids))
        p=scores[f'{tag}_B0_inner']; r=scores[f'{tag}_A1_inner']; calibrations[tag]={'B0':{'threshold':.37,'alpha':1.},'B0_T':calibrate(p,y[va]),'A1_T':calibrate(r,y[va]),'B1':calibrate(r,y[va],p)}
        outer[tag]={v:evaluate(scores[f'{tag}_{"A1" if v in ["A1_T","B1"] else "B0"}_outer'],y[ot],calibrations[tag][v],scores[f'{tag}_B0_outer'] if v=='B1' else None) for v in ['B0','B0_T','A1_T','B1']}
    dump('源验证进度.json',{'calibrations':calibrations,'outer':outer})
    print('CALIBRATED_CONTROLS',json.dumps({t:{v:short(r) for v,r in c.items()} for t,c in calibrations.items()},ensure_ascii=False),flush=True)
    for v in ['B2','B3']:
        for f in folds:
            tag=f'fold{f["fold"]}'; ids=np.array(f['fit_ids'])-1; va=np.array(f['inner_ids'])-1; ot=np.array(f['outer_ids'])-1
            for k in KINDS:
                for s in SEEDS[k]: records.append(fit(v,k,s,tag,x,y,ids,va))
            ip=score_models(tag,v,e.validation(x[va])).reshape(16,len(va)); op=score_models(tag,v,torch.cat([e.validation(x[ot]),a.held_views(x[ot])])).reshape(20,len(ot))
            scores[f'{tag}_{v}_inner']=ip; scores[f'{tag}_{v}_outer']=op; calibrations[tag][v]=calibrate(ip,y[va]); outer[tag][v]=evaluate(op,y[ot],calibrations[tag][v])
            dump('源验证进度.json',{'calibrations':calibrations,'outer':outer}); np.savez_compressed(OUT/'源图分数_进行中.npz',**scores)
            print('CALIBRATED',tag,v,json.dumps(short(calibrations[tag][v])),flush=True)
    for f in folds:
        tag=f'fold{f["fold"]}'; ids=np.array(f['fit_ids'])-1; va=np.array(f['inner_ids'])-1; ot=np.array(f['outer_ids'])-1
        bases[tag]=choose_base(calibrations[tag]); objective='A1' if bases[tag]=='B1' else bases[tag]; dump('B4基底选择.json',bases)
        for k in KINDS:
            for s in SEEDS[k]: records.append(fit('B4',k,s,tag,x,y,ids,va,objective))
        ip=score_models(tag,'B4',e.validation(x[va])).reshape(16,len(va)); op=score_models(tag,'B4',torch.cat([e.validation(x[ot]),a.held_views(x[ot])])).reshape(20,len(ot))
        scores[f'{tag}_B4_inner']=ip; scores[f'{tag}_B4_outer']=op; calibrations[tag]['B4']=calibrate(ip,y[va],scores[f'{tag}_B0_inner'],calibrations[tag]['B0_T']); outer[tag]['B4']=evaluate(op,y[ot],calibrations[tag]['B4'],scores[f'{tag}_B0_outer'])
        dump('源验证进度.json',{'calibrations':calibrations,'outer':outer}); np.savez_compressed(OUT/'源图分数_进行中.npz',**scores)
        print('CALIBRATED',tag,'B4',json.dumps(short(calibrations[tag]['B4'])),flush=True)
    yy=np.concatenate([y[np.array(f['inner_ids'])-1] for f in folds]); old=pool_inner(scores,folds,'B0'); aa=pool_inner(scores,folds,'A1')
    full={'B0':{'threshold':.37,'alpha':1.},'B0_T':calibrate(old,yy),'A1_T':calibrate(aa,yy),'B1':calibrate(aa,yy,old)}
    for v in ['B2','B3']: full[v]=calibrate(pool_inner(scores,folds,v),yy)
    bases['full']=choose_base(full); full['B4']=calibrate(pool_inner(scores,folds,'B4'),yy,old,full['B0_T'])
    # Fold B4 objectives may differ: disclose that the pooled threshold uses the
    # selected procedure, whereas full refit uses the pooled-inner chosen base.
    epochs={v:{k:int(np.floor(np.median([r['epoch'] for r in records if r['variant']==v and r['kind']==k])+.5)) for k in KINDS} for v in NEW}
    dump('源数据锁定配方.json',{'calibrations':full,'B4_bases':bases,'full_epochs':epochs,'calibration_source':'three inner107 sets only','outer_not_used_for_selection':True})
    for v in NEW:
        base=('A1' if bases['full']=='B1' else bases['full']) if v=='B4' else 'A1'
        for k in KINDS:
            for s in SEEDS[k]: records.append(fit(v,k,s,'full',x,y,np.arange(800),None,base,epochs[v][k]))
    dump('全部训练记录.json',records); np.savez_compressed(OUT/'源图内部与外层评分.npz',**scores)
    assert all(sha(r['file'])==r['sha256'] for r in manifest)
    final=[{'file':str(path_for('full',v,k,s).relative_to(OUT)),'sha256':sha(path_for('full',v,k,s))} for v in NEW for k in KINDS for s in SEEDS[k]]
    lock={'calibrations':full,'B4_bases':bases,'full_epochs':epochs,'new_models':final,'code_sha256':sha(__file__),'plan_sha256':sha(OUT/'运行前固定方案.json'),'existing_manifest_sha256':sha(OUT/'既有权重来源.json'),'source_sha256':{n:sha(ROOT/'work/data'/n) for n in ['train_data.mat','train_label.mat']},'source_score_sha256':sha(PREV/'源图内部与外层评分.npz'),'forbidden_attempts':attempts,'target_data_read':False}
    dump('模型与阈值锁定.json',lock); dump('运行进度.json',{'status':'complete','training_runs':len(records),'rows':len(LOGS)}); print('LOCKED',len(records),{v:{k:r[k] for k in ['threshold','alpha']} for v,r in full.items()},flush=True)

def predict():
    attempts=guard('predict'); lock=json.loads((OUT/'模型与阈值锁定.json').read_text()); before=sha(OUT/'模型与阈值锁定.json')
    assert lock['code_sha256']==sha(__file__) and lock['plan_sha256']==sha(OUT/'运行前固定方案.json')
    assert all(sha(OUT/r['file'])==r['sha256'] for r in lock['new_models'])
    assert all(sha(r['file'])==r['sha256'] for r in json.loads((OUT/'既有权重来源.json').read_text()))
    x=e.u.image(loadmat(ROOT/'work/data/test_data.mat')['test_data']); raw={'B0':score_models('full','B0',x),'A1':score_models('full','A1',x)}
    for v in NEW: raw[v]=score_models('full',v,x)
    scores={'B0':raw['B0'],'B0_T':raw['B0'],'A1_T':raw['A1'],'B1':calibrated_scores(raw['A1'],lock['calibrations']['B1'],raw['B0']),'B2':raw['B2'],'B3':raw['B3'],'B4':calibrated_scores(raw['B4'],lock['calibrations']['B4'],raw['B0'])}
    files=[]
    scores={v:np.asarray(p,dtype=np.float64) for v,p in scores.items()}
    for v,p in scores.items():
        t=lock['calibrations'][v]['threshold']; path=OUT/f'result_{v}.txt'; pred=np.where(p>=t,1,-1)
        path.write_bytes(''.join(f'{i} {int(z)}\n' for i,z in enumerate(pred,1)).encode()); files.append({'variant':v,'file':path.name,'threshold':t,'sha256':sha(path)})
    assert (OUT/'result_B0.txt').read_bytes()==(BASE/'A.txt').read_bytes()
    assert sha(OUT/'模型与阈值锁定.json')==before
    np.savez_compressed(OUT/'锁定后预测分数.npz',**scores,**{'raw_'+k:v for k,v in raw.items()})
    dump('预测核验.json',{'files':files,'B0_original_byte_identical':True,'lock_unchanged':True,'manual_overrides':False,'forbidden_attempts':attempts}); print('PREDICTED',flush=True)

def check():
    # Numerical checks matter here: exact threshold scan and bounded mining.
    rng=np.random.default_rng(35); y=np.array([-1,-1,-1,1,1,1]); p=rng.uniform(.01,.99,(16,6)); r=threshold_search(p,y)
    vals=np.unique(p); ts=np.r_[0.,(vals[:-1]+vals[1:])/2,1.]; brute=max(metrics(p,y,float(t))['balanced_accuracy'] for t in ts)
    assert abs(r['balanced_accuracy']-brute)<1e-12
    constant=threshold_search(np.full((16,6),.5),y); assert abs(constant['balanced_accuracy']-.5)<1e-12
    per=torch.tensor([[.1,.2,.3,.4],[.5,.6,.7,.8],[.9,1.,1.1,1.2]],requires_grad=True); target=torch.tensor([0.,0.,1.,1.]); loss,k,w=hard_negative_loss(per,target); assert k==1 and w==2.; loss.backward(); assert per.grad.max()/per.grad.min()==2.
    z=F.normalize(torch.randn(12,8),dim=1).requires_grad_(); s=structure_loss(z,target); s.backward(); assert torch.isfinite(z.grad).all()
    print('CHECKS_PASSED',DEVICE,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['check','train','predict'],required=True); arg=p.parse_args(); {'check':check,'train':train,'predict':predict}[arg.stage]()
