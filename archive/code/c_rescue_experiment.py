"""Nested, calibrated rescue cascades. All task fitting uses source data only."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,math,random,sys,time
import numpy as np
import torch
from torch.nn import functional as F
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold,train_test_split

ROOT=Path('${ORIGINAL_WORKSPACE}')
BASE=ROOT/'outputs/七网络原图通道与条件蒸馏';AOUT=ROOT/'outputs/A1-A4模糊鲁棒性对照'
OUT=ROOT/'outputs/C组保留人脸与校准救援';OUT.mkdir(exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
sp=importlib.util.spec_from_file_location('butil',ROOT/'work/b1_b4_experiment.py');b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b)
b.OUT=OUT;b.a.OUT=OUT;a=b.a;e=b.e;DEVICE=b.DEVICE
KINDS=b.KINDS;SEEDS=b.SEEDS
GROUPS=['C1_OR','C1_SCORE','C1_GATE','C2_PHYS','C3_PC','C4_AT'];ALL=['C0']+GROUPS+['C5_SAFE']
VARIANTS=['PHYS','PC','AT'];CAP1={'resnet8':80,'hires19':80,'densenet121':30};CAP2={'resnet8':40,'hires19':40,'densenet121':20}
LR1={'resnet8':.001,'hires19':.001,'densenet121':.0001};LR2={'resnet8':.0001,'hires19':.0001,'densenet121':.00003}
PLAN={'reference':'C0 exact B0/A0 full seven networks with original threshold0.37; protected positive decisions for every group',
 'architecture':'frozen reference seven nets; rescue classifier consists of ResNet8 three seeds, DenseNet121 seed42, HiRes19 three seeds; new physical/PC/AT training modifies only HiRes19 family',
 'groups':{'C1_OR':'raw family probability mean with per-pair independently selected threshold','C1_SCORE':'nonnegative logit ridge fusion, no quality features','C1_GATE':'same score fusion plus three image quality features','C2_PHYS':'replace HiRes family by physical-curriculum HiRes; C1 R/D frozen','C3_PC':'C2 plus focal logit preservation','C4_AT':'C2 plus spatial attention transfer','C5_SAFE':'per pair source-select one group with a new-false-positive budget0.005; may disable rescue'},
 'nested_evaluation':'3 original outer folds; within each outer training set, 3 calibration pairs; fit/early-stop/calibration disjoint; calibration split2/3 gate-fit and1/3 threshold-selection; outer never used for choices',
 'deployment':'3 original source folds become deployed classifier-calibrator pairs: original fit/inner/outer roles become fit/early-stop/calibration; calibration split meta/selection; deploy these exact weights, no full800 refit',
 'teacher':'same-pair same-seed frozen A1 HiRes19; trained only on fit IDs, early stopping only ES; no calibration or outer images in teacher fitting',
 'stage1':'original source images with existing mild augmentation; first phase new for nested pairs, reuse matched stage1 for deployed pairs',
 'stage2':'paired normal and two degradation views; restart same stage1 for PHYS/PC/AT; no third expert training stage',
 'stage1_caps':CAP1,'stage2_caps':CAP2,'stage1_lr':LR1,'stage2_lr':LR2,'optimizer':'AdamW wd0.001 cosine over caps; batch64 small/32 Dense; all Dense BN frozen; only Dense block4/head trainable',
 'checkpoint':'ES clean BCE stage1; ES fixed16 weighted BCE stage2; first/even/final epoch; fit/ES clean paired logs, fit fixed16 every10/new best/final',
 'physical':'direct PSF-width, motion length, defocus radius, downsample size and noise progression0.35->1 in first half; no addition of a clean-image residual to degraded images; both classes same policy',
 'PC':'CE +0.01 weighted logit L2/2, teacher logits clamped[-10,10], weights1+5*I(teacher p>=0.37 agrees with source label); teacher sees identical normal/degraded input; normalized by total weights',
 'AT':'CE +0.1 squared-L2 spatial attention; 3x3 pooled sum of squared channel activations normalized; teacher sees same aligned normal view, students two degraded views; source-label-correct clean teacher mask; both classes',
 'gate':'meta-only weighted standardization; logistic ridge0.01, first3 logit coefficients in[0,4], quality coefficients in[-2,2] with4x ridge penalty, intercept[-4,4]; quality logstd/mean/log low-to-high radial Fourier energy ratio',
 'gate_features':'no source ID or reference score in the learned gate; quality alone is not face evidence; all views receive source/view weights',
 'thresholds':'each group/pair independently exact-search all attainable score partitions on selection IDs; primary standalone16-view weighted BA; ties worst-viewBA, cleanBA, wide plateau, near0.5',
 'safe_selection':'selection-only maximize complete cascade weightedBA under newly introduced weightedFPR<=0.005; disabled option; ties fewer new false positives and simpler group; no guarantee this finite-source constraint holds on unseen data',
 'vote':'at least2 of3 calibrated pairs vote rescue; original reference-positive always face; vote2 is predeclared, not target-tuned',
 'source_views':e.VIEWS,'weights':e.VW.tolist(),'held_stress':a.VIEW_NAMES[16:],'target_data_in_training':False,'target_labels_used':False,
 'new_fit_count':{'nested_stage1':63,'nested_reference_stage2':63,'nested_A1_stage2':63,'nested_new_HiRes_stage2':81,'deployed_new_HiRes_stage2':27,'total':297},
}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def guard(stage):
    allowed={ROOT/'work/data/train_data.mat',ROOT/'work/data/train_label.mat',BASE/'源图分折.json',AOUT/'源图内部与外层评分.npz',e.CACHE}
    if stage=='predict':allowed|={ROOT/'work/data/test_data.mat',BASE/'A.txt'}
    attempts=[]
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        p=Path(args[0]).resolve();mode=args[1] if len(args)>1 else None
        if isinstance(mode,str) and any(t in mode for t in 'wax+'):return
        if not p.is_relative_to(ROOT) or p.suffix.lower() not in ['.mat','.json','.npz','.pt','.pth','.csv','.txt','.pkl']:return
        ok=p in allowed or p.is_relative_to(OUT) or p.is_relative_to(ROOT/'work/cnn_env')
        ok|=p.parent==BASE/'baseline_models' and p.suffix=='.pt'
        ok|=p.parent==AOUT/'models' and p.name.startswith('fold') and '_A1_' in p.name and p.suffix=='.pt'
        if not ok:attempts.append(str(p));raise RuntimeError('Forbidden task data read '+str(p))
    sys.addaudithook(hook);return attempts

def partitions(y):
    folds=json.loads((BASE/'源图分折.json').read_text());allpairs=[];deploy=[]
    for outer in folds:
        pool=np.sort(np.array(outer['fit_ids']+outer['inner_ids'])-1)
        for j,(tr,ca) in enumerate(StratifiedKFold(3,shuffle=True,random_state=9120+outer['fold']).split(pool,y[pool]),1):
            fit,es=train_test_split(pool[tr],test_size=.2,stratify=y[pool[tr]],random_state=5340+outer['fold']*10+j)
            cal=pool[ca];meta,sel=train_test_split(cal,test_size=1/3,stratify=y[cal],random_state=7180+outer['fold']*10+j)
            allpairs.append({'tag':f'o{outer["fold"]}p{j}','outer_fold':outer['fold'],'pair':j,'fit_ids':(fit+1).tolist(),'es_ids':(es+1).tolist(),'meta_ids':(meta+1).tolist(),'selection_ids':(sel+1).tolist(),'outer_ids':outer['outer_ids'],'external_fold':None})
        cal=np.array(outer['outer_ids'])-1;meta,sel=train_test_split(cal,test_size=1/3,stratify=y[cal],random_state=8810+outer['fold'])
        deploy.append({'tag':f'deploy{outer["fold"]}','outer_fold':None,'pair':outer['fold'],'fit_ids':outer['fit_ids'],'es_ids':outer['inner_ids'],'meta_ids':(meta+1).tolist(),'selection_ids':(sel+1).tolist(),'outer_ids':[],'external_fold':outer['fold']})
    for p in allpairs+deploy:
        roles=[set(p[k]) for k in ['fit_ids','es_ids','meta_ids','selection_ids','outer_ids']]
        assert all(not roles[i]&roles[j] for i in range(5) for j in range(i+1,5))
        assert set.union(*roles)==set(range(1,801))
    return folds,allpairs,deploy
def ckpath(pair,k,s,stage):
    if pair['external_fold'] and stage in ['stage1','B0','A1']:
        tag=f'fold{pair["external_fold"]}'
        return AOUT/'models'/f'{tag}_A1_{k}_{s}.pt' if stage=='A1' else BASE/f'baseline_models/{tag}_{k}_{s}_base_stage{1 if stage=="stage1" else 2}.pt'
    return OUT/'models'/f'{pair["tag"]}_{k}_{s}_{stage}.pt'
def loadnet(path,k):
    ck=torch.load(path,map_location='cpu',weights_only=True);net=e.build(k);net.load_state_dict(ck['state_dict']);return net,ck
def energy_attention(h):
    energy=h.square().sum(1,keepdim=True);v=b.spatial_pool(energy)[:,1:];return F.normalize(v,dim=1,eps=1e-8)
def physical(clean,progress):
    out=clean.clone();types=torch.randint(6,(len(out),));strength=.35+.65*min(1,progress*2)
    for typ in range(6):
        ids=torch.where(types==typ)[0]
        if not len(ids):continue
        z=out[ids];size=None
        if typ in [1,5]:z=e.u.blur(z,random.choice([.4,.7,1.,1.4,1.8])*strength)
        if typ==2:z=a.motion(z,random.uniform(0,math.pi),3 if strength<.6 else random.choice([3,5,7]))
        if typ==3:
            radius=random.choice([.8,1.2,1.6])*strength;t=torch.arange(-2,3).float();kernel=((t[:,None]**2+t[None,:]**2)<=radius**2).float();kernel/=kernel.sum();z=F.conv2d(F.pad(z,(2,)*4,mode='reflect'),kernel[None,None])
        if typ in [4,5]:size=int(round(19-strength*(19-random.choice([6,8,10,12,15]))))
        if random.random()<.3:
            peak=random.choice([100.,250.,500.])/max(strength,.1);z=torch.poisson(z.clamp_min(0)*peak)/peak
        if size is not None:
            mode=random.choice(['area','bilinear']);z=F.interpolate(z,(size,size),mode=mode,**({'align_corners':False} if mode=='bilinear' else {}));z=F.interpolate(z,(19,19),mode='bilinear',align_corners=False)
        gain=1-strength*(1-torch.tensor([1.,.8,.6,.4])[torch.randint(4,(len(ids),))][:,None,None,None]);mean=z.mean((2,3),keepdim=True)
        noise=torch.tensor([0.,.005,.015,.025])[torch.randint(4,(len(ids),))][:,None,None,None]*strength
        z=((z-mean)*gain+mean+torch.randn_like(z)*noise).clamp(0,1)
        if random.random()<.2:z=e.jpeg(z,random.choice([40,60,80]))
        out[ids]=z
    return out.clamp(0,1)

RECORDS=[];LOGS=[]
def logsave():
    if LOGS:
        with (OUT/'逐轮训练验证.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(LOGS[0]));w.writeheader();w.writerows(LOGS)
def fit(pair,k,s,stage,x,y):
    path=ckpath(pair,k,s,stage)
    if path.exists():
        ck=torch.load(path,map_location='cpu',weights_only=True)
        if pair['external_fold'] and stage in ['stage1','B0','A1']:
            r=ck.get('record',ck);assert r['fit_ids']==pair['fit_ids'] and r.get('validation_ids',[])==pair['es_ids'];return
        assert ck['record']['fit_ids']==pair['fit_ids'] and ck['record']['es_ids']==pair['es_ids'];RECORDS.append(ck['record']);print('REUSE',path.name,flush=True);return
    phase=1 if stage=='stage1' else 2;cap=(CAP1 if phase==1 else CAP2)[k];lr=(LR1 if phase==1 else LR2)[k]
    e.u.seed(s+7000*phase);net=e.build(k,initial=phase==1);parent=None
    if phase==2:parent=ckpath(pair,k,s,'stage1');net.load_state_dict(torch.load(parent,map_location='cpu',weights_only=True)['state_dict'])
    net.to(DEVICE);teacher=None;thook=shook=None;box={};teacher_path=None
    if stage in ['PC','AT']:
        teacher_path=ckpath(pair,k,s,'A1');teacher,_=loadnet(teacher_path,k);teacher.to(DEVICE).eval()
        for t in teacher.parameters():t.requires_grad=False
        if stage=='AT':
            def capture_t(module,inp,out):box['teacher']=out
            def capture_s(module,inp,out):box['student']=out
            thook=teacher.features.register_forward_hook(capture_t);shook=net.features.register_forward_hook(capture_s)
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=lr,weight_decay=.001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cap)
    ids=np.array(pair['fit_ids'])-1;es=np.array(pair['es_ids'])-1;target=torch.tensor((y==1).astype('float32'));rng=np.random.default_rng(s+phase*113);bs=32 if k=='densenet121' else 64
    vx=x[es] if phase==1 else e.validation(x[es]);best=float('inf');state=None;bestepoch=cap;history=[];started=time.monotonic()
    print('START',pair['tag'],k,s,stage,'n',len(ids),'epochs',cap,flush=True)
    for ep in range(1,cap+1):
        net.train();a.freeze_bn(net,k);sums=np.zeros(4);seen=0;tick=time.monotonic();eligible=0;currentlr=opt.param_groups[0]['lr']
        for bi,ii in enumerate(np.array_split(rng.permutation(ids),math.ceil(len(ids)/bs))):
            random.seed(s+phase*100000+ep*1000+bi);torch.manual_seed(s+phase*100000+ep*1000+bi);tt=target[ii].to(DEVICE)
            if phase==1:images=e.n.augment(x[ii]);ttall=tt
            elif stage=='B0':images=e.legacy(e.n.augment(x[ii]));ttall=tt
            else:
                clean=a.base_view(x[ii]);corrupt=a.corrupt if stage=='A1' else lambda z:physical(z,(ep-1)/max(1,cap-1));d1=corrupt(clean);d2=corrupt(clean);images=torch.cat([clean,d1,d2]);ttall=tt.repeat(3)
            z=net(e.n.prep(images.to(DEVICE),k)).flatten();ce=F.binary_cross_entropy_with_logits(z,ttall);pc=ce*0;at=ce*0;loss=ce
            if stage=='PC':
                with torch.no_grad():zt=teacher(e.n.prep(images.to(DEVICE),k)).flatten();mask=((zt>=math.log(.37/.63))==(ttall>.5));weight=1+5*mask.float()
                pc=(weight*.5*(z-zt.clamp(-10,10)).square()).sum()/weight.sum();loss=loss+.01*pc;eligible+=int(mask.sum())
            if stage=='AT':
                hs=box['student']
                with torch.no_grad():zt=teacher(e.n.prep(clean.to(DEVICE),k)).flatten();mask=((zt>=math.log(.37/.63))==(tt>.5));ta=energy_attention(box['teacher'])
                sa=energy_attention(hs[len(ii):]);dist=(sa-ta.repeat(2,1)).square().sum(1);mm=mask.repeat(2);at=(dist*mm).sum()/mm.sum().clamp_min(1);loss=loss+.1*at;eligible+=int(mask.sum())
            opt.zero_grad();loss.backward();opt.step();n=len(ii);seen+=n;sums+=np.array([float(v.detach().cpu()) for v in [loss,ce,pc,at]])*n;box.clear()
        scheduler.step();tc=tr=vc=vr=None
        if ep==1 or ep%2==0 or ep==cap:
            vp=e.predict(net,vx,k);vr=e.bce(vp,y[es],phase==2);vc=e.bce(vp.reshape(-1,len(es))[:1],y[es],False);tc=e.bce(e.predict(net,x[ids],k),y[ids],False)
            improved=vr<best
            if phase==2 and (ep%10==0 or improved or ep==cap):tr=e.bce(e.predict(net,e.validation(x[ids]),k),y[ids])
            if improved:best=vr;bestepoch=ep;state={key:v.detach().cpu().clone() for key,v in net.state_dict().items()}
            box.clear()
        row={'tag':pair['tag'],'kind':k,'seed':s,'stage':stage,'epoch':ep,'lr':currentlr,'train_objective':float(sums[0]/seen),'train_ce':float(sums[1]/seen),'train_pc':float(sums[2]/seen),'train_at':float(sums[3]/seen),'teacher_eligible':eligible,'fit_presentations':seen,'train_clean_bce':tc,'train_robust_bce':tr,'es_clean_bce':vc,'es_checkpoint_bce':vr,'seconds':time.monotonic()-tick}
        history.append(row);LOGS.append(row);dump('运行进度.json',{'status':'training','current':row,'completed_models':len(RECORDS)})
        if ep==1 or ep%20==0 or ep==cap:print('EPOCH',pair['tag'],k,s,stage,ep,'CE',round(row['train_ce'],4),'ES',None if vr is None else round(vr,5),flush=True)
    net.load_state_dict(state)
    if thook:thook.remove()
    if shook:shook.remove()
    rec={'tag':pair['tag'],'kind':k,'seed':s,'stage':stage,'phase':phase,'epoch':bestepoch,'executed_epochs':cap,'fit_ids':pair['fit_ids'],'es_ids':pair['es_ids'],'parent_file':None if parent is None else str(parent),'parent_sha256':None if parent is None else sha(parent),'teacher_file':None if teacher_path is None else str(teacher_path),'teacher_sha256':None if teacher_path is None else sha(teacher_path),'pretrained_sha256':e.CACHE_SHA if k=='densenet121' and phase==1 else None,'best_es_bce':best,'history':history,'seconds':time.monotonic()-started}
    # Atomic replacement leaves no partial checkpoint when a run is interrupted.
    temp=path.with_suffix('.tmp');torch.save({'state_dict':{key:v.detach().cpu() for key,v in net.state_dict().items()},'record':rec},temp);temp.replace(path);RECORDS.append(rec);logsave();dump('训练记录_进行中.json',RECORDS)
    print('DONE',pair['tag'],k,s,stage,'best',bestepoch,'sec',round(rec['seconds'],1),flush=True);del net,teacher,opt
    if DEVICE=='mps':torch.mps.empty_cache()

def family(pair,k,stage,images):
    pp=[]
    for s in SEEDS[k]:
        net,_=loadnet(ckpath(pair,k,s,stage),k);net.to(DEVICE);pp.append(e.predict(net,images,k));del net
    return np.mean(pp,axis=0)
def quality(images):
    z=images[:,0].numpy().astype('float64');mean=z.mean((1,2));std=z.std((1,2));zz=z-mean[:,None,None]
    power=np.abs(np.fft.rfft2(zz,axes=(1,2)))**2;rad=np.hypot(np.fft.fftfreq(19)[:,None],np.fft.rfftfreq(19)[None,:]);lo=(rad>0)&(rad<=.15);hi=rad>.15
    ratio=np.log((power[:,lo].mean(1)+1e-8)/(power[:,hi].mean(1)+1e-8)).clip(-8,8)
    return np.stack([np.log(std+.001),mean,ratio],axis=1)
def features(families,q):
    p=np.asarray(families).T.clip(1e-5,1-1e-5);return np.c_[np.log(p/(1-p)).clip(-8,8),q]
def fit_gate(X,y,dimensions):
    n=len(y);yy=np.tile(y==1,16);w=np.concatenate([np.full(n,v/n) for v in e.VW]);xx=X[:,:dimensions].astype('float64');mean=w@xx;scale=np.sqrt(w@((xx-mean)**2)).clip(.05,None);xx=(xx-mean)/scale
    bw=np.tile(np.where(y==1,.5/(y==1).sum(),.5/(y==-1).sum()),16)*np.repeat(e.VW,n)
    penalty=np.r_[np.ones(3),np.full(max(0,dimensions-3),4.)]
    def objective(theta):
        z=xx@theta[:-1]+theta[-1];p=expit(z);loss=float(bw@(np.logaddexp(0,z)-yy*z)+.01*np.sum(penalty*theta[:-1]**2));error=bw*(p-yy)
        return loss,np.r_[xx.T@error+.02*penalty*theta[:-1],error.sum()]
    initial=np.r_[np.full(3,1/3),np.zeros(dimensions-3),0.];bounds=[(0,4)]*3+[(-2,2)]*(dimensions-3)+[(-4,4)]
    r=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=bounds,options={'maxiter':300,'ftol':1e-12});assert r.success,r.message
    return {'dimensions':dimensions,'mean':mean.tolist(),'scale':scale.tolist(),'coefficients':r.x[:-1].tolist(),'intercept':float(r.x[-1]),'meta_objective':float(r.fun)}
def gate_predict(X,gate):
    xx=(X[:,:gate['dimensions']]-gate['mean'])/gate['scale'];return expit(xx@np.array(gate['coefficients'])+gate['intercept'])
def decisions_metrics(d,y):
    d=np.asarray(d,dtype=bool).reshape(-1,len(y));truth=y==1;rec=d[:,truth].mean(1);spec=(~d[:,~truth]).mean(1);acc=(d==truth).mean(1);w=e.VW if len(d)==16 else np.ones(len(d))/len(d)
    return {'accuracy':float(w@acc),'balanced_accuracy':float(w@((rec+spec)/2)),'recall':float(w@rec),'specificity':float(w@spec),'clean_accuracy':float(acc[0]),'worst_accuracy':float(acc.min()),'views':[{'name':name,'accuracy':float(v),'recall':float(r),'specificity':float(s)} for name,v,r,s in zip(e.VIEWS if len(d)==16 else a.VIEW_NAMES[16:],acc,rec,spec)]}
def safe_threshold(p,base,y):
    p=np.asarray(p,dtype='float64').reshape(16,len(y));base=np.asarray(base).reshape(16,len(y))>=.37;u=np.unique(p);t=np.r_[0.,u[0]/2,(u[:-1]+u[1:])/2,(u[-1]+1)/2,1.];truth=y==1
    candidates=[{'enabled':False,'threshold':None,'new_fpr':0.,**decisions_metrics(base,y)}]
    for threshold in t:
        new=(p>=threshold)&~base;fpr=float(e.VW@new[:,~truth].mean(1))
        if fpr>.005+1e-12:continue
        r=decisions_metrics(base|new,y);r.update({'enabled':True,'threshold':float(threshold),'new_fpr':fpr});candidates.append(r)
    return sorted(candidates,key=lambda r:(-round(r['balanced_accuracy'],12),r['new_fpr'],r['enabled'],-r['worst_accuracy'],abs((r['threshold'] if r['enabled'] else .5)-.5)))[0]

def process_pair(pair,x,y):
    # Deployed legacy models are reused only when gradient and ES IDs exactly match.
    for k in KINDS:
        for s in SEEDS[k]:
            fit(pair,k,s,'stage1',x,y);fit(pair,k,s,'B0',x,y);fit(pair,k,s,'A1',x,y)
    for variant in VARIANTS:
        for s in SEEDS['hires19']:fit(pair,'hires19',s,variant,x,y)
    calids=np.array(pair['meta_ids']+pair['selection_ids'])-1;nm=len(pair['meta_ids']);ncal=len(calids);cal=e.validation(x[calids]);qc=quality(cal)
    outerids=np.array(pair['outer_ids'])-1;ev=None if not len(outerids) else torch.cat([e.validation(x[outerids]),a.held_views(x[outerids])]);qe=None if ev is None else quality(ev)
    shared_c={k:family(pair,k,'A1',cal) for k in KINDS};shared_e={} if ev is None else {k:family(pair,k,'A1',ev) for k in KINDS}
    base_cal=np.mean([family(pair,k,'B0',cal) for k in KINDS],axis=0).reshape(16,ncal)
    families_c={'A1':np.stack([shared_c[k] for k in KINDS])};families_e={} if ev is None else {'A1':np.stack([shared_e[k] for k in KINDS])}
    for variant in VARIANTS:
        families_c[variant]=np.stack([shared_c['resnet8'],family(pair,'hires19',variant,cal),shared_c['densenet121']])
        # Preserve the common feature order R, H, D, matching KINDS.
        if ev is not None:families_e[variant]=np.stack([shared_e['resnet8'],family(pair,'hires19',variant,ev),shared_e['densenet121']])
    # Source family order is KINDS = R,H,D; all code consistently uses this order.
    pair_configs={};eval_scores={};cal_scores={};sel_y=y[calids[nm:]];meta_y=y[calids[:nm]]
    for group in GROUPS:
        policy={'C2_PHYS':'PHYS','C3_PC':'PC','C4_AT':'AT'}.get(group,'A1');fc=families_c[policy];Xc=features(fc,qc).reshape(16,ncal,6);gate=None
        if group=='C1_OR':pc=fc.mean(0).reshape(16,ncal)
        else:
            gate=fit_gate(Xc[:,:nm].reshape(-1,6),meta_y,3 if group=='C1_SCORE' else 6);pc=gate_predict(Xc.reshape(-1,6),gate).reshape(16,ncal)
        threshold=b.threshold_search(pc[:,nm:],sel_y);cfg={'policy':policy,'gate':gate,'threshold':threshold['threshold'],'threshold_selection':threshold,'meta_ids':pair['meta_ids'],'selection_ids':pair['selection_ids']};pair_configs[group]=cfg;cal_scores[group]=pc[:,nm:]
        if ev is not None:
            fe=families_e[policy];X=features(fe,qe);pe=fe.mean(0) if group=='C1_OR' else gate_predict(X,gate);eval_scores[group]=np.asarray(pe,dtype='float64').reshape(20,len(outerids))
    safety=[]
    for group in GROUPS:
        r=safe_threshold(cal_scores[group],base_cal[:,nm:],sel_y);r['group']=group;safety.append(r)
    safe=sorted(safety,key=lambda r:(-round(r['balanced_accuracy'],12),r['new_fpr'],r['enabled'],GROUPS.index(r['group'])))[0];pair_configs['C5_SAFE']=safe
    allcal={group:cal_scores[group] for group in GROUPS};allcal['baseline']=base_cal[:,nm:]
    np.savez_compressed(OUT/f'{pair["tag"]}_选择集评分.npz',**allcal)
    np.savez_compressed(OUT/f'{pair["tag"]}_折外评分.npz',**eval_scores)
    dump(f'{pair["tag"]}_校准配方.json',pair_configs)
    print('CALIBRATED',pair['tag'],{g:round(pair_configs[g]['threshold'],5) for g in GROUPS},'safe',safe['group'],safe['enabled'],flush=True)
    return pair_configs,eval_scores

def rescue_votes(configs,scores):
    votes={g:np.zeros_like(next(iter(scores[0].values())),dtype='int64') for g in GROUPS+['C5_SAFE']}
    for cfg,pp in zip(configs,scores):
        for g in GROUPS:votes[g]+=(pp[g]>=cfg[g]['threshold'])
        safe=cfg['C5_SAFE']
        if safe['enabled']:votes['C5_SAFE']+=(pp[safe['group']]>=safe['threshold'])
    return votes
def train():
    attempts=guard('train')
    if (OUT/'运行前固定方案.json').exists():assert json.loads((OUT/'运行前固定方案.json').read_text())==PLAN
    else:dump('运行前固定方案.json',PLAN)
    x=e.u.image(loadmat(ROOT/'work/data/train_data.mat')['train_data']);y=loadmat(ROOT/'work/data/train_label.mat')['train_label'].ravel();folds,nested,deployed=partitions(y);dump('全部源图角色划分.json',{'outer':folds,'nested':nested,'deployed':deployed})
    manifest=[]
    for pair in deployed:
        for k in KINDS:
            for s in SEEDS[k]:
                for st in ['stage1','B0','A1']:
                    p=ckpath(pair,k,s,st);manifest.append({'file':str(p),'sha256':sha(p)})
    for k in KINDS:
        for s in SEEDS[k]:
            p=BASE/f'baseline_models/full_{k}_{s}_base_stage2.pt';manifest.append({'file':str(p),'sha256':sha(p)})
    dump('复用权重来源.json',manifest)
    if (OUT/'逐轮训练验证.csv').exists():
        with (OUT/'逐轮训练验证.csv').open(encoding='utf-8-sig') as f:LOGS.extend(csv.DictReader(f))
    oldsrc=np.load(AOUT/'源图内部与外层评分.npz');source_results=[];deployed_configs=[]
    for outer in folds:
        configs=[];scores=[]
        for pair in [p for p in nested if p['outer_fold']==outer['fold']]:
            cfg,pp=process_pair(pair,x,y);configs.append(cfg);scores.append(pp)
        votes=rescue_votes(configs,scores);ids=np.array(outer['outer_ids'])-1;base=oldsrc[f'fold{outer["fold"]}_A0_outer'].reshape(20,len(ids))>=.37
        decisions={'C0':base,**{g:base|(votes[g]>=2) for g in votes}}
        for g,d in decisions.items():
            assert not np.any(base&~d)
            source_results.append({'outer_fold':outer['fold'],'group':g,'standard':decisions_metrics(d[:16],y[ids]),'held':decisions_metrics(d[16:],y[ids]),'new_face_count_clean':int((d[0]&~base[0]).sum())})
        np.savez_compressed(OUT/f'outer{outer["fold"]}_集成决策.npz',**{g:d.astype('int8') for g,d in decisions.items()},**{'vote_'+g:v for g,v in votes.items()})
        dump('源折外评估进度.json',source_results);print('OUTER',outer['fold'],{r['group']:round(r['standard']['accuracy'],5) for r in source_results if r['outer_fold']==outer['fold']},flush=True)
    for pair in deployed:
        cfg,_=process_pair(pair,x,y);deployed_configs.append(cfg)
    assert len(RECORDS)==297,(len(RECORDS),PLAN['new_fit_count'])
    dump('全部训练记录.json',RECORDS);dump('部署校准器与阈值.json',deployed_configs)
    newmodels=[{'file':str(p.relative_to(OUT)),'sha256':sha(p)} for p in sorted((OUT/'models').glob('*.pt'))]
    assert len(newmodels)==297 and all(sha(r['file'])==r['sha256'] for r in manifest)
    lock={'new_models':newmodels,'new_training_runs':len(RECORDS),'deployed_pairs':deployed,'configs':deployed_configs,'baseline_threshold':.37,'votes_required':2,'code_sha256':sha(__file__),'plan_sha256':sha(OUT/'运行前固定方案.json'),'manifest_sha256':sha(OUT/'复用权重来源.json'),'split_sha256':sha(OUT/'全部源图角色划分.json'),'source_data_sha256':{n:sha(ROOT/'work/data'/n) for n in ['train_data.mat','train_label.mat']},'forbidden_attempts':attempts,'target_read':False}
    dump('模型与阈值锁定.json',lock);dump('运行进度.json',{'status':'complete','new_models':len(RECORDS)});print('LOCKED',len(RECORDS),flush=True)

def predict():
    attempts=guard('predict');lock=json.loads((OUT/'模型与阈值锁定.json').read_text());before=sha(OUT/'模型与阈值锁定.json')
    assert lock['code_sha256']==sha(__file__) and lock['plan_sha256']==sha(OUT/'运行前固定方案.json')
    assert all(sha(OUT/r['file'])==r['sha256'] for r in lock['new_models'])
    assert all(sha(r['file'])==r['sha256'] for r in json.loads((OUT/'复用权重来源.json').read_text()))
    x=e.u.image(loadmat(ROOT/'work/data/test_data.mat')['test_data']);q=quality(x);base=b.score_models('full','B0',x).astype('float64');allprob=[]
    for pair,cfg in zip(lock['deployed_pairs'],lock['configs']):
        common={k:family(pair,k,'A1',x) for k in KINDS};families={'A1':np.stack([common[k] for k in KINDS])}
        for policy in VARIANTS:families[policy]=np.stack([common['resnet8'],family(pair,'hires19',policy,x),common['densenet121']])
        pp={}
        for group in GROUPS:
            c=cfg[group];p=families[c['policy']];pp[group]=p.mean(0) if c['gate'] is None else gate_predict(features(p,q),c['gate'])
        allprob.append(pp)
    votes=rescue_votes(lock['configs'],allprob);labels={'C0':np.where(base>=.37,1,-1),**{g:np.where((base>=.37)|(v>=2),1,-1) for g,v in votes.items()}};files=[]
    for g,p in labels.items():
        assert not np.any((base>=.37)&(p==-1));path=OUT/f'result_{g}.txt';path.write_bytes(''.join(f'{i} {int(v)}\n' for i,v in enumerate(p,1)).encode());files.append({'group':g,'file':path.name,'sha256':sha(path)})
    assert (OUT/'result_C0.txt').read_bytes()==(BASE/'A.txt').read_bytes();assert sha(OUT/'模型与阈值锁定.json')==before
    np.savez_compressed(OUT/'锁定后评分与投票.npz',baseline=base,**{'vote_'+g:v for g,v in votes.items()},**{f'pair{j+1}_{g}':v for j,pp in enumerate(allprob) for g,v in pp.items()})
    dump('预测核验.json',{'files':files,'baseline_byte_identical':True,'all_old_faces_preserved':True,'all420_labels_from_models':True,'manual_overrides':False,'lock_unchanged':True,'forbidden_attempts':attempts});print('PREDICTED',flush=True)
def check():
    torch.manual_seed(11);random.seed(11);xx=torch.rand(8,1,19,19);zz=physical(xx,1);assert zz.shape==xx.shape and torch.isfinite(zz).all() and zz.min()>=0 and zz.max()<=1
    hh=torch.randn(8,32,19,19,requires_grad=True);aa=energy_attention(hh);assert aa.shape==(8,9);aa.square().sum().backward();assert torch.isfinite(hh.grad).all()
    yy=np.array([-1,-1,1,1]);X=np.tile(np.array([[-2,-1,-3,.1,.2,.3],[-1,-2,-2,.3,.2,.1],[1,2,2,.3,.2,.1],[2,1,3,.1,.2,.3]]),(16,1));g=fit_gate(X,yy,6);p=gate_predict(X,g);assert p.reshape(16,4)[:,:2].mean()<p.reshape(16,4)[:,2:].mean()
    base=np.tile(np.array([.1,.1,.8,.2]),(16,1));r=safe_threshold(p.reshape(16,4),base,yy);assert r['new_fpr']<=.005
    print('CHECKS_PASSED',DEVICE,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['check','train','predict'],required=True);args=p.parse_args();{'check':check,'train':train,'predict':predict}[args.stage]()
