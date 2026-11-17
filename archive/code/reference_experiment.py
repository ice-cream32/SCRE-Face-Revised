"""Same seven backbones: add raw input at zero initialization; conditional KD.
Reuse only audited checkpoints with EXACTLY matching source partitions.
"""
from pathlib import Path
import argparse,sys,json,hashlib,random,platform
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold,train_test_split
import networks as n
import utils as u
import protocol as pr
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent.parent;SRC=ROOT/'outputs/七网络定向鲁棒优化'
DEVICE='mps' if torch.backends.mps.is_available() else 'cpu';torch.set_num_threads(4)
SEEDS=[42,7,2026];KINDS=['resnet8','densenet121','hires19'];DENSE_INIT='a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395'
RECIPES={'A':{'resnet8':'base','densenet121':'base','hires19':'base'},'B':{'resnet8':'base','densenet121':'base','hires19':'raw'},'C':{'resnet8':'raw','densenet121':'base','hires19':'raw'},'D':{'resnet8':'kd','densenet121':'base','hires19':'kd'}}
PLAN={'recipes':RECIPES,'seeds':SEEDS,'networks_per_candidate':7,'family_fusion':'probability means within seed, equal1/3 across3 families','threshold':.37,
 'source_cv':'3 outer folds seed20261112; inner20% seed871+fold; checkpoints require exact fit/validation ID equality',
 'new_training':'stage2 only, initialized from audited same-partition stage1, not previous stage2; first-stage training already completed',
 'raw_input':'channel0 clip(zscore,-3,3); channel1=2*x-1; expanded first convolution=[oldweight,zeros], bias unchanged',
 'stage2_epochs_max':40,'stage2_lr':.0001,'weight_decay':.001,'batch':64,'validation':'all16 original source views, clean0.5 remaining0.5; best weighted BCE',
 'augmentation':'exact legacy n.augment -> 50% Gaussian/downsample/combined; unchanged to isolate raw-input and KD',
 'kd_teacher':'frozen DenseNet BASE stage2 with same fit/inner IDs; same augmented image, both classes',
 'kd_lambda':.1,'kd_teacher_true_class_min':.9,'kd_student_true_class_max':.7,
 'kd_mask':'only fit labels, teacher confidence>=.9 and student true-class score<.7; no outer/test labels',
 'kd_loss':'BCE logits +0.1*mean(masked Bernoulli KL(stopgrad teacher||student)); no feature-matching loss',
 'teacher_cache':'same source IDs and augmented image hash required for reuse; training inputs only',
 'selection':'mean inner weighted accuracy, worst-view accuracy, clean accuracy, A/B/C/D order; no outer or target selection',
 'full_refit':'matching full800 first-stage weights; median inner best epochs per kind/variant; baseline full800 already trained',
 'residual_feature_head':'not included; isolate input changes and conditional supervision before adding a joint head',
 'test_labels_used':False,'test_overrides':False}
ACTIVE_TAG=None;TEACHER_CACHE={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,v):(OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def guard(stage):
    attempts=[];allowed={ROOT/'work/data/train_data.mat',ROOT/'work/data/train_label.mat'}
    if stage=='predict':allowed.add(ROOT/'work/data/test_data.mat')
    permitted=set()
    for tag in ['fold1','fold2','fold3','full']:
        for kind in KINDS:
            for seed in [42] if kind=='densenet121' else SEEDS:
                for phase in [1,2]:permitted.add(SRC/'models'/f'{tag}_{kind}_{seed}_base_stage{phase}.pt')
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        path=Path(args[0]).resolve();bad=path.is_relative_to(ROOT/'outputs') and not path.is_relative_to(OUT)
        if stage=='train' and path in permitted and ACTIVE_TAG is not None and path.name.startswith(ACTIVE_TAG+'_'):bad=False
        bad|=path.is_relative_to(ROOT/'work') and path.suffix in ['.mat','.json','.npz','.txt','.csv','.pt','.pth','.pkl'] and path not in allowed and not path.is_relative_to(ROOT/'work/cnn_env')
        if bad:attempts.append(str(path));raise RuntimeError('Forbidden history/target or mismatched fold checkpoint '+str(path))
    sys.addaudithook(hook);return attempts
def build(kind):
    if kind=='resnet8':return n.Res8()
    if kind=='hires19':return n.HiRes19()
    net=models.densenet121(weights=None);net.classifier=nn.Linear(1024,1);return net
def expand(net,kind):
    if kind=='densenet121':raise ValueError('DenseNet input unchanged')
    layer=net.stem if kind=='resnet8' else net.features[0];old=layer[0];rng=torch.random.get_rng_state()
    new=nn.Conv2d(2,old.out_channels,old.kernel_size,old.stride,old.padding,old.dilation,old.groups,old.bias is not None,padding_mode=old.padding_mode);torch.random.set_rng_state(rng)
    with torch.no_grad():
        new.weight.zero_();new.weight[:,:1].copy_(old.weight)
        if old.bias is not None:new.bias.copy_(old.bias)
    layer[0]=new;net._raw_channel=True;return net
def prep(net,x,kind):
    z=n.prep(x,kind);return torch.cat([z,2*x-1],1) if getattr(net,'_raw_channel',False) else z
@torch.no_grad()
def predict(net,x,kind):
    net.eval();return np.concatenate([net(prep(net,b.to(DEVICE),kind)).flatten().sigmoid().cpu().numpy() for b in x.split(64)])
def import_baselines(tag,fitids,valids):
    global ACTIVE_TAG
    ACTIVE_TAG=tag;records=[]
    for kind in KINDS:
        for seed in [42] if kind=='densenet121' else SEEDS:
            parent=None
            for phase in [1,2]:
                path=SRC/'models'/f'{tag}_{kind}_{seed}_base_stage{phase}.pt';b=torch.load(path,map_location='cpu',weights_only=True)
                assert set(b['fit_ids'])==set((fitids+1).tolist()) and b['validation_ids']==([] if valids is None else (valids+1).tolist()) and b['phase']==phase and b['policy']=='base'
                if phase==2:assert b['parent_sha256']==sha(parent)
                if kind=='densenet121':assert b['init_sha256']==DENSE_INIT
                dest=OUT/'baseline_models'/path.name;dest.write_bytes(path.read_bytes());assert sha(dest)==sha(path)
                records.append({'tag':tag,'kind':kind,'seed':seed,'phase':phase,'file':str(dest.relative_to(OUT)),'sha256':sha(dest),'fit_ids':b['fit_ids'],'validation_ids':b['validation_ids'],'parent_sha256':b['parent_sha256']});parent=path
    return records
def load_baseline(tag,kind,seed,phase=2):
    path=OUT/'baseline_models'/f'{tag}_{kind}_{seed}_base_stage{phase}.pt';b=torch.load(path,map_location='cpu',weights_only=True);net=build(kind);net.load_state_dict(b['state_dict']);return net,path,b
def cached_teacher(teacher,batch,ids,tag,seed,epoch,bi):
    key=(tag,seed,epoch,bi);digest=hashlib.sha256(batch.numpy().tobytes()+ids.tobytes()).hexdigest()
    if key in TEACHER_CACHE:
        h,p=TEACHER_CACHE[key];assert h==digest;return p,True
    p=predict(teacher,batch,'densenet121');TEACHER_CACHE[key]=(digest,p);return p,False
def fit(kind,seed,variant,x,y,fitids,valids=None,epochs=None,tag=''):
    u.seed(seed+14000);net,parent,b=load_baseline(tag,kind,seed,phase=1);expand(net,kind);net.to(DEVICE);epochs=epochs or 40
    opt=torch.optim.AdamW(net.parameters(),lr=.0001,weight_decay=.001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    teacher=None;teacher_path=None
    if variant=='kd':
        teacher,teacher_path,tb=load_baseline(tag,'densenet121',42,2);assert tb['fit_ids']==b['fit_ids'] and tb['validation_ids']==b['validation_ids'];teacher.to(DEVICE).eval()
        for p in teacher.parameters():p.requires_grad=False
    ty=torch.from_numpy((y==1).astype('float32'));rng=np.random.default_rng(seed+226);best=float('inf');bs=None;be=epochs;history=[];kd_stats=[];vx=pr.validation(x[valids]) if valids is not None else None
    for epoch in range(1,epochs+1):
        net.train();counts={'epoch':epoch,'seen':0,'eligible_positive':0,'eligible_negative':0,'cache_hits':0}
        for bi,ids in enumerate(np.array_split(rng.permutation(fitids),int(np.ceil(len(fitids)/64)))):
            random.seed(seed+200000+epoch*1000+bi);torch.manual_seed(seed+200000+epoch*1000+bi);batch=pr.legacy(n.augment(x[ids]));t=ty[ids].to(DEVICE);zs=net(prep(net,batch.to(DEVICE),kind)).flatten();ll=F.binary_cross_entropy_with_logits(zs,t)
            if teacher is not None:
                pt,hit=cached_teacher(teacher,batch,ids,tag,seed,epoch,bi);pt=torch.from_numpy(pt).to(DEVICE).clamp(1e-5,1-1e-5);ps=zs.sigmoid().clamp(1e-5,1-1e-5);teacher_true=torch.where(t>.5,pt,1-pt);student_true=torch.where(t>.5,ps.detach(),1-ps.detach());mask=(teacher_true>=.9)&(student_true<.7)
                kl=pt*torch.log(pt/ps)+(1-pt)*torch.log((1-pt)/(1-ps));ll=ll+.1*(mask*kl).sum()/mask.sum().clamp_min(1)
                counts['seen']+=len(ids);counts['eligible_positive']+=int((mask&(t>.5)).sum());counts['eligible_negative']+=int((mask&(t<.5)).sum());counts['cache_hits']+=int(hit)
            opt.zero_grad();ll.backward();opt.step()
        sched.step()
        if teacher is not None:kd_stats.append(counts)
        if valids is not None and (epoch%2==0 or epoch==1 or epoch==epochs):
            vl=pr.bce(predict(net,vx,kind),y[valids]);history.append({'epoch':epoch,'loss':vl})
            if vl<best:best=vl;be=epoch;bs={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    if valids is not None:net.load_state_dict(bs)
    record={'kind':kind,'seed':seed,'variant':variant,'phase':2,'two_channels':True,'epoch':be,'history':history,'fit_ids':b['fit_ids'],'validation_ids':b['validation_ids'],'parent_file':str(parent.relative_to(OUT)),'parent_sha256':sha(parent),'teacher_file':str(teacher_path.relative_to(OUT)) if teacher_path else None,'teacher_sha256':sha(teacher_path) if teacher_path else None,'kd_statistics':kd_stats,'parameters':sum(p.numel() for p in net.parameters())}
    path=OUT/'models'/f'{tag}_{kind}_{seed}_{variant}_stage2.pt';torch.save({**record,'state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()}},path);print('FIT',tag,kind,seed,variant,be,round(best,5) if valids is not None else 'full800',flush=True);return net,path,record
def candidates(groups):return {name:np.mean([groups[f'{k}_{v}'] for k,v in recipe.items()],0) for name,recipe in RECIPES.items()}
def rank(rows):return sorted(rows,key=lambda r:(-round(r['weighted_accuracy'],12),-round(r['worst_view_accuracy'],12),-round(r['clean_accuracy'],12),list(RECIPES).index(r['candidate'])))
def train():
    attempts=guard('train');(OUT/'models').mkdir(exist_ok=True);(OUT/'baseline_models').mkdir(exist_ok=True);dump('运行前固定方案.json',PLAN)
    X=loadmat(ROOT/'work/data/train_data.mat')['train_data'];y=loadmat(ROOT/'work/data/train_label.mat')['train_label'].ravel();assert X.shape==(800,361);x=u.image(X);folds=[];imports=[];records=[];innerrows=[];outerrows=[];procedure=[];scores_saved={};epochs={}
    for fold,(outertr,outerva) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261112).split(X,y),1):
        ids,va=train_test_split(outertr,test_size=.2,stratify=y[outertr],random_state=871+fold);tag=f'fold{fold}';imports+=import_baselines(tag,ids,va);folds.append({'fold':fold,'fit_ids':(ids+1).tolist(),'inner_ids':(va+1).tolist(),'outer_ids':(outerva+1).tolist()});vi=pr.validation(x[va]);vo=pr.validation(x[outerva]);ig={};og={}
        for kind in KINDS:
            ips=[];ops=[]
            for seed in [42] if kind=='densenet121' else SEEDS:
                net,_,_=load_baseline(tag,kind,seed);net.to(DEVICE);ips.append(predict(net,vi,kind));ops.append(predict(net,vo,kind));del net
            ig[kind+'_base']=np.mean(ips,0);og[kind+'_base']=np.mean(ops,0)
            if kind=='densenet121':continue
            for variant in ['raw','kd']:
                ips=[];ops=[]
                for seed in SEEDS:
                    net,path,r=fit(kind,seed,variant,x,y,ids,va,tag=tag);records.append(r);epochs.setdefault(kind+'_'+variant,[]).append(r['epoch']);ips.append(predict(net,vi,kind));ops.append(predict(net,vo,kind));del net
                ig[kind+'_'+variant]=np.mean(ips,0);og[kind+'_'+variant]=np.mean(ops,0);dump('新增训练记录_进行中.json',records)
        ins=candidates(ig);outs=candidates(og);fr=[{'fold':fold,'candidate':k,**pr.metrics(p,y[va])} for k,p in ins.items()];innerrows+=fr;outerrows +=[{'fold':fold,'candidate':k,**pr.metrics(p,y[outerva])} for k,p in outs.items()];winner=rank(fr)[0]['candidate'];procedure.append({'fold':fold,'inner_winner':winner,**pr.metrics(outs[winner],y[outerva])})
        for k,p in ins.items():scores_saved[f'fold{fold}_inner_{k}']=p
        for k,p in outs.items():scores_saved[f'fold{fold}_outer_{k}']=p
        dump('交叉验证进度.json',{'folds':folds,'inner':innerrows,'outer':outerrows,'procedure':procedure});print('OUTER',fold,winner,procedure[-1]['weighted_accuracy'],flush=True);TEACHER_CACHE.clear()
    summaries=[]
    for name in RECIPES:
        ii=[r for r in innerrows if r['candidate']==name];oo=[r for r in outerrows if r['candidate']==name];summaries.append({'candidate':name,**{k:float(np.mean([r[k] for r in ii])) for k in ['weighted_accuracy','worst_view_accuracy','clean_accuracy']},'outer_mean':float(np.mean([r['weighted_accuracy'] for r in oo])),'outer_std':float(np.std([r['weighted_accuracy'] for r in oo],ddof=1))})
    ranking=rank(summaries);winner=ranking[0]['candidate'];dump('仅源数据选型.json',{'primary':winner,'ranking':ranking,'outer_procedure':procedure,'target_read':False});dump('源图分折.json',folds);full_epochs={k:int(np.floor(np.median(v)+.5)) for k,v in epochs.items()};imports+=import_baselines('full',np.arange(800),None);final=[]
    for kind in ['resnet8','hires19']:
        for variant in ['raw','kd']:
            for seed in SEEDS:
                net,path,r=fit(kind,seed,variant,x,y,np.arange(800),epochs=full_epochs[kind+'_'+variant],tag='full');records.append(r);del net;final.append({'file':str(path.relative_to(OUT)),'sha256':sha(path)})
    for kind in KINDS:
        for seed in [42] if kind=='densenet121' else SEEDS:
            path=OUT/'baseline_models'/f'full_{kind}_{seed}_base_stage2.pt';final.append({'file':str(path.relative_to(OUT)),'sha256':sha(path)})
    dump('复用权重来源核验.json',imports);dump('全部新增训练记录.json',records);np.savez(OUT/'仅源交叉验证分数.npz',**scores_saved)
    import torchvision,sklearn,scipy,PIL
    dump('运行环境.json',{'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,'Pillow':PIL.__version__,'device':DEVICE})
    lock={'primary':winner,'ranking':ranking,'threshold':.37,'recipes':RECIPES,'epochs':full_epochs,'models':final,'imported_checkpoints':[{'file':r['file'],'sha256':r['sha256']} for r in imports],'code_sha256':sha(__file__),'helper_sha256':{n:sha(OUT/n) for n in ['networks.py','utils.py','protocol.py']},'plan_sha256':sha(OUT/'运行前固定方案.json'),'split_sha256':sha(OUT/'源图分折.json'),'source_sha256':{n:sha(ROOT/'work/data'/n) for n in ['train_data.mat','train_label.mat']},'test_data_read':False,'test_labels_used':False,'forbidden_attempts':attempts};dump('模型锁定.json',lock);print('LOCKED',winner,json.dumps(ranking,ensure_ascii=False),flush=True)
def infer():
    guard('predict');lock=json.loads((OUT/'模型锁定.json').read_text());before=sha(OUT/'模型锁定.json');assert lock['code_sha256']==sha(__file__) and lock['threshold']==.37
    for n,h in lock['helper_sha256'].items():assert sha(OUT/n)==h
    for r in lock['models']:assert sha(OUT/r['file'])==r['sha256']
    x=u.image(loadmat(ROOT/'work/data/test_data.mat')['test_data']);assert len(x)==420;groups={}
    for kind in KINDS:
        for variant in ['base'] if kind=='densenet121' else ['base','raw','kd']:
            runs=[]
            for seed in [42] if kind=='densenet121' else SEEDS:
                path=OUT/('baseline_models' if variant=='base' else 'models')/f'full_{kind}_{seed}_{variant}_stage2.pt';net=build(kind)
                if variant!='base':expand(net,kind)
                net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict']);net.to(DEVICE);runs.append(predict(net,x,kind));del net
            groups[kind+'_'+variant]=np.mean(runs,0)
    scores=candidates(groups);files=[]
    for name,p in scores.items():
        path=OUT/f'{name}.txt';path.write_bytes(''.join(f'{i} {int(v)}\n' for i,v in enumerate(np.where(p>=.37,1,-1),1)).encode());files.append({'candidate':name,'file':path.name,'sha256':sha(path)})
    (OUT/'result_seven_raw_kd.txt').write_bytes((OUT/f"{lock['primary']}.txt").read_bytes());np.savez(OUT/'锁定后测试分数.npz',**scores,**groups);assert before==sha(OUT/'模型锁定.json');dump('预测核验.json',{'files':files,'primary':lock['primary'],'threshold':.37,'lock_unchanged':True,'manual_changes':False,'test_truth_used':False});print('PREDICTED',lock['primary'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['train','predict'],required=True);args=p.parse_args();train() if args.stage=='train' else infer()
