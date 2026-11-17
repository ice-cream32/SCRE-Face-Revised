"""Source-only figures and numerical evidence for the final S01 model report."""
from pathlib import Path
import csv,hashlib,importlib.util,json,math,shutil,sys
import numpy as np
import torch
from scipy.io import loadmat
from scipy.special import expit,logit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/S01最终模型完整报告'
OLD=ROOT/'outputs/C1_SCORE详细方法报告';AB=ROOT/'outputs/C1_SCORE消融与简化'
AS=OUT/'assets';EV=OUT/'evidence';CODE=OUT/'code'
for p in [AS,EV,CODE]:p.mkdir(exist_ok=True,parents=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,obj):(EV/n).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def readcsv(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvout(n,rows):
 with (EV/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

# Import pure source view generators; no training or inference entry point runs.
sp=importlib.util.spec_from_file_location('s01_views',ROOT/'work/c_rescue_experiment.py');c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c);c.a.OUT=OUT
def audit(event,args):
 if event!='open' or not isinstance(args[0],(str,bytes)):return
 p=Path(args[0]).resolve();mode=args[1] if len(args)>1 else None
 if isinstance(mode,str) and any(k in mode for k in 'wax+'):return
 if p.suffix=='.mat' and p not in {ROOT/'work/data/train_data.mat',ROOT/'work/data/train_label.mat'}:raise RuntimeError('Report reads only original source images/labels')
 if p.is_relative_to(AB) and ('target' in p.name or p.name in ['锁定后预测与分支评分.npz','全部预测差异.json']):raise RuntimeError('No deployment-image analysis in source report')
sys.addaudithook(audit)
FONT=Path('${ORIGINAL_HOME}/Library/Fonts/MedicalMVP-MicrosoftYaHei.ttc');BOLD=FONT.with_name('MedicalMVP-MicrosoftYaHei-Bold.ttc')
for p in [FONT,BOLD]:fm.fontManager.addfont(str(p))
plt.rcParams.update({'font.family':fm.FontProperties(fname=str(FONT)).get_name(),'axes.unicode_minus':False,'font.size':11,'axes.titlesize':14,'figure.dpi':130,'savefig.dpi':170,'axes.spines.top':False,'axes.spines.right':False})
COL=['#1766A2','#C47D28','#168575','#AF4F70'];FONTS=[]
def save(fig,name):
 fig.canvas.draw()
 from matplotlib.text import Text
 n=0
 for t in fig.findobj(match=Text):
  if t.get_text():t.set_fontproperties(fm.FontProperties(fname=str(BOLD if t.get_fontweight() in ['bold',700] else FONT),size=t.get_fontsize()));n+=1
 fig.savefig(AS/name,bbox_inches='tight',facecolor='white');plt.close(fig)
 FONTS.append({'file':name,'font':str(FONT),'text_objects':n})

# These source-only assets already contain no obsolete model comparison.
copied=['02_训练样本.png','03_数据统计.png','04_预处理.png','05_固定退化面板.png','06_保留压力面板.png','07_训练验证配对曲线.png','08_独立阈值搜索.png']
oldfonts={r['figure']:r for r in json.loads((OLD/'evidence/图像字体核验.json').read_text())}
for name in copied:
 shutil.copyfile(OLD/'assets'/name,AS/name);FONTS.append({'file':name,'font':str(FONT),'reused_source_only_figure':True,'source_sha256':sha(OLD/'assets'/name),'prior_font_audit':oldfonts[name]})
for name in ['训练数据核验.json','训练图像描述统计.csv','原图划分人数.csv','原图划分完整记录.json','28网络部署权重与轮次.json','配对曲线真实记录.csv','三组阈值搜索复算.csv','三组部署校准器.json']:
 shutil.copyfile(OLD/'evidence'/name,EV/name)
def report_rows(rows):
 result=[]
 for r in rows:
  if r['name'] not in ['S01_EXACT'] and not r['name'].startswith(('F','R')):continue
  r=dict(r)
  if r['name']=='S01_EXACT':r['name']='S01'
  r={k:v for k,v in r.items() if 'differences' not in k}
  if 'title' in r:r['title']=r['title'].replace('删除原判别器','删除初判分支').replace('旧B0退化策略','基础退化策略').replace('旧阈值','既有阈值').replace('旧系数','既有系数').replace('旧融合','既有融合')
  result.append(r)
 return result
csvout('源图全部消融结果.csv',report_rows(readcsv(AB/'源图全部消融结果.csv')))
csvout('成对差值与条件bootstrap区间.csv',report_rows(readcsv(AB/'evidence/成对差值与条件bootstrap区间.csv')))
original_metrics=json.loads((AB/'源图全部指标.json').read_text())
dump('源图全部指标.json',{('S01' if k=='S01_EXACT' else k):v for k,v in original_metrics.items() if k=='S01_EXACT' or k.startswith(('F','R'))})
shutil.copyfile(AB/'evidence/条件区间方法.json',EV/'条件区间方法.json')
constraint_rows=json.loads((AB/'evidence/非负约束触边核验.json').read_text())
for r in constraint_rows:
 if r['case']=='BASE':r['case']='S01'
dump('非负约束触边核验.json',constraint_rows)
for unused in ['逐组模型规模与结果.csv','运行前方案.json']:
 (EV/unused).unlink(missing_ok=True)
dump('来源摘录说明.json',{'scope':'S01 final architecture plus its source-only fixed/refitted ablations. Unused predecessor, pruning-candidate and deployment-image comparison columns omitted.',
 'source_files':[{'file':str(p),'sha256':sha(p)} for p in [AB/'源图全部消融结果.csv',AB/'源图全部指标.json',AB/'evidence/成对差值与条件bootstrap区间.csv']],
 'relabel':'S01_EXACT is named S01; numbers unchanged; no model selection or fitting.'})
for name,path in {'c1_shared_runtime.py':AB/'compact/c1_shared_runtime.py','networks.py':AB/'compact/networks.py','c_rescue_experiment.py':ROOT/'work/c_rescue_experiment.py','a1_a4_experiment.py':ROOT/'work/a1_a4_experiment.py','b1_b4_experiment.py':ROOT/'work/b1_b4_experiment.py','source_experiment.py':ROOT/'outputs/七网络定向鲁棒优化/experiment.py','utils.py':ROOT/'outputs/七网络定向鲁棒优化/utils.py','reference_protocol.py':ROOT/'outputs/七网络原图通道与条件蒸馏/protocol.py','reference_experiment.py':ROOT/'outputs/七网络原图通道与条件蒸馏/experiment.py','historical_threshold_search.py':ROOT/'outputs/模糊架构互补实验/architecture_study.py'}.items():shutil.copyfile(path,CODE/name)
meta=json.loads((AB/'compact/metadata.json').read_text());model=AB/'compact/compact_model.pt'
assert sha(model)==meta['portable_files_sha256']['compact_model.pt']
dump('S01部署模型身份.json',{'name':'S01','parameters':15294620,'logical_members':28,'small_members':24,'dense_tails':4,'shared_prefix_output':[512,4,4],'weights_file':str(model),'weights_sha256':sha(model),'runtime_sha256':sha(AB/'compact/c1_shared_runtime.py'),'compiled_gates':[{'weights':g['weights'],'decision_bias':g['decision_bias']} for g in meta['compiled_gates']]})

mat=loadmat(ROOT/'work/data/train_data.mat')['train_data'];y=loadmat(ROOT/'work/data/train_label.mat')['train_label'].ravel().astype(int);x=c.e.u.image(mat);xx=x[:,0].numpy();truth=y==1
split=json.loads((OLD/'evidence/原图划分完整记录.json').read_text());nested=json.loads((AB/'嵌套重新校准配方.json').read_text());source=np.load(AB/'源图全部消融决策.npz');decision=source['S01_EXACT'];W=c.e.VW
members=[(k,s) for k in c.KINDS for s in c.SEEDS[k]]
def family(raw):return np.stack([raw[[i for i,(kk,ss) in enumerate(members) if kk==k]].mean(0) for k in c.KINDS],-1)
def compiled_vote(p,cfg):
 p=p.clip(1e-5,1-1e-5);l=np.log(p/(1-p)).clip(-8,8).astype('float64');g=cfg['gate'];a=np.array(g['coefficients']);mu=np.array(g['mean']);s=np.array(g['scale']);w=a/s;gamma=g['intercept']-mu@w-logit(cfg['threshold']);h=l@w+gamma;d=h>=0
 near=np.abs(h)<=1e-10*(1+np.abs(l)@np.abs(w)+abs(gamma))
 if near.any():d[near]=expit(((l[near]-mu)/s)@a+g['intercept'])>=cfg['threshold']
 return d
vfull=np.empty((20,800,19,19),np.float32);initial=np.empty((20,800),bool);votes=np.empty((3,20,800),bool);fold_id=np.zeros(800,int)
for outer in split['outer']:
 f=outer['fold'];ids=np.array(outer['outer_ids'])-1;n=len(ids);fold_id[ids]=f
 images=torch.cat([c.e.validation(x[ids]),c.a.held_views(x[ids])]);vfull[:,ids]=images.reshape(20,n,19,19).numpy()
 rb=np.load(AB/f'banks/reference_fold{f}.npz');assert str(rb['view_sha'])==hashlib.sha256(images.numpy().tobytes()).hexdigest()
 initial[:,ids]=family(rb['B0']).mean(-1).astype('float64')>=.37
 for j in range(1,4):
  b=np.load(AB/f'banks/o{f}p{j}_outer.npz');assert str(b['view_sha'])==str(rb['view_sha'])
  votes[j-1][:,ids]=compiled_vote(family(b['A1']),nested[f'o{f}p{j}']['S01_EXACT'])
assert np.array_equal(initial|(votes.sum(0)>=2),decision)
np.savez_compressed(EV/'S01源图折外决策.npz',labels=y,decisions=decision,initial_decisions=initial,rescue_votes=votes,outer_fold=fold_id)
csvout('S01训练原图折外决策.csv',[{'source_id':i+1,'label':int(y[i]),'outer_fold':int(fold_id[i]),'initial_face':bool(initial[0,i]),'rescue_votes':int(votes[:,0,i].sum()),'S01_label':1 if decision[0,i] else -1} for i in range(800)])
acc=(decision==truth).mean(1);rec=decision[:,truth].mean(1);spec=(~decision[:,~truth]).mean(1)
metrics={'clean_accuracy':float(acc[0]),'weighted_accuracy':float(W@acc[:16]),'weighted_recall':float(W@rec[:16]),'weighted_specificity':float(W@spec[:16]),'worst_standard_accuracy':float(acc[:16].min()),'held_accuracy':float(acc[16:].mean()),'held_recall':float(rec[16:].mean()),'held_specificity':float(spec[16:].mean()),'TP':int((decision[0]&truth).sum()),'TN':int((~decision[0]&~truth).sum()),'FN':int((~decision[0]&truth).sum()),'FP':int((decision[0]&~truth).sum())}
names=['原图','高斯σ=1','高斯σ=1.5','高斯σ=2','降至12×12','降至7×7','水平运动5','对比度0.3','噪声σ=0.025','高斯1＋噪声','高斯1.5＋噪声','降至9＋噪声','模糊＋低对比＋噪声','对角运动5','JPEG45','右移1像素','圆盘散焦','JPEG25','椒盐噪声2%','模糊＋散斑']
vr=[{'view':i,'name':name,'accuracy':float(acc[i]),'face_recall':float(rec[i]),'nonface_specificity':float(spec[i]),'kind':'standard' if i<16 else 'held'} for i,name in enumerate(names)]
dump('S01内部评价指标.json',metrics);csvout('S01逐退化视图指标.csv',vr)

# Absolute conditional intervals resample original clusters within fold x class.
rng=np.random.default_rng(20261116);boot=5000;bw=np.zeros((boot,800))
for f in [1,2,3]:
 for cl in [-1,1]:
  ii=np.flatnonzero((fold_id==f)&(y==cl));bw[:,ii]=rng.multinomial(len(ii),np.repeat(1/len(ii),len(ii)),size=boot)/800
correct=(decision==truth).astype(float);per=np.stack([correct[0],W@correct[:16],correct[16:].mean(0)])
lo,hi=np.quantile(bw@per.T,[.025,.975],axis=0);ci={key:{'point':float(per[i].mean()),'low':float(lo[i]),'high':float(hi[i])} for i,key in enumerate(['clean_accuracy','weighted_accuracy','held_accuracy'])}
dump('S01绝对指标条件区间.json',{'method':'5000 original-cluster bootstrap, stratify outer-fold x label, seed20261116; fixed weights/thresholds/procedure, no training/selection variance','intervals':ci})

# 09 spectrum is descriptive only; never fed into the model.
z=xx-xx.mean((1,2),keepdims=True);power=np.abs(np.fft.fftshift(np.fft.fft2(z),axes=(1,2)))**2
freq=np.fft.fftshift(np.fft.fftfreq(19));rad=np.hypot(freq[:,None],freq[None,:]);low=(rad>0)&(rad<=.15);allf=rad>0
ratio=power[:,low].sum(1)/(power[:,allf].sum(1)+1e-12)
fig,axs=plt.subplots(1,3,figsize=(13,4))
for ax,t,title in zip(axs[:2],[True,False],['人脸平均对数频谱','非人脸平均对数频谱']):
 im=ax.imshow(np.log1p(power[truth==t].mean(0)),cmap='magma',extent=[freq[0],freq[-1],freq[-1],freq[0]]);ax.set_title(title);ax.set_xlabel('水平频率：周期/像素');ax.set_ylabel('垂直频率');fig.colorbar(im,ax=ax,fraction=.045)
for t,color,name in [(True,COL[0],'人脸'),(False,COL[1],'非人脸')]:axs[2].hist(ratio[truth==t],bins=20,alpha=.55,density=True,label=name,color=color)
axs[2].set_title('半径≤0.15的低频能量比例');axs[2].set_xlabel('非直流能量占比');axs[2].set_ylabel('密度');axs[2].legend();fig.suptitle('训练数据的频谱描述：分析指标不参与S01推理或路由',fontsize=16);fig.tight_layout(rect=[0,0,1,.92]);save(fig,'09_训练数据频谱描述.png')
csvout('频谱描述统计.csv',[{'source_id':i+1,'label':int(y[i]),'low_frequency_fraction':float(ratio[i])} for i in range(800)])

# 10 one final model confusion matrix, no predecessor comparison.
cm=np.array([[metrics['TN'],metrics['FP']],[metrics['FN'],metrics['TP']]])
fig,ax=plt.subplots(figsize=(6,5));ax.imshow(cm,cmap='Blues',vmin=0,vmax=400);ax.set_xticks([0,1],['非人脸','人脸']);ax.set_yticks([0,1],['非人脸','人脸']);ax.set_xlabel('S01预测');ax.set_ylabel('原训练标签');ax.set_title('S01原图折外混淆矩阵：790/800')
for i in range(2):
 for j in range(2):ax.text(j,i,str(cm[i,j]),fontsize=23,ha='center',va='center',color='white' if cm[i,j]>200 else '#172b4d')
save(fig,'10_S01折外混淆矩阵.png')

fig,axs=plt.subplots(2,6,figsize=(12,5))
for row,t in enumerate([True,False]):
 ids=np.flatnonzero((decision[0]==truth)&(truth==t))
 for ax,i in zip(axs[row],ids[np.linspace(0,len(ids)-1,6).astype(int)]):
  ax.imshow(xx[i],cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_title(f'训练 #{i+1} · 外折{fold_id[i]}\n标签/预测：{"人脸" if t else "非人脸"}',fontsize=10);ax.axis('off')
fig.suptitle('S01分类正确的训练原图折外示例',fontsize=17);fig.tight_layout(rect=[0,0,1,.92]);save(fig,'11_S01正确分类示例.png')
wrong=np.flatnonzero(decision[0]!=truth);fig,axs=plt.subplots(2,5,figsize=(11,5.6))
for ax,i in zip(axs.ravel(),wrong):
 ax.imshow(xx[i],cmap='gray',vmin=0,vmax=1,interpolation='nearest');path='初判分支通过' if initial[0,i] else '救援多数通过';ax.set_title(f'训练 #{i+1} · 外折{fold_id[i]}\n标签：非人脸；预测：人脸\n{path}，救援{votes[:,0,i].sum()}/3票',fontsize=9);ax.axis('off')
fig.suptitle('S01原图折外全部10个错误：按原训练标签评价',fontsize=16);fig.tight_layout(rect=[0,0,1,.92]);save(fig,'12_S01全部原图错误.png')

fig,axs=plt.subplots(3,1,figsize=(14,9),sharex=True)
for ax,values,label,color in zip(axs,[acc,rec,spec],['准确率','人脸召回率','非人脸特异度'],COL[:3]):
 ax.axvspan(15.5,19.5,color='#fbefd9',alpha=.85);ax.plot(range(20),values*100,marker='o',ms=4,lw=1.8,color=color);ax.set_ylabel(label+'（%）');ax.set_ylim(max(0,math.floor((min(values)*100-3)/5)*5),100.8);ax.grid(alpha=.15)
axs[-1].set_xticks(range(20),names,rotation=42,ha='right');fig.suptitle('S01训练集内部逐退化鲁棒性：浅色区为保留压力视图',fontsize=17);fig.tight_layout(rect=[0,0,1,.95]);save(fig,'13_S01逐视图鲁棒性.png')
fig,ax=plt.subplots(figsize=(8,4.5));sig=[0,1,1.5,2]
for val,label,color in [(acc,'准确率',COL[0]),(rec,'人脸召回率',COL[1]),(spec,'非人脸特异度',COL[2])]:ax.plot(sig,val[[0,1,2,3]]*100,marker='o',label=label,color=color)
ax.set_xlabel('高斯模糊σ（原图像素单位）');ax.set_ylabel('比例（%）');ax.set_xticks(sig);ax.set_ylim(90,100.7);ax.set_title('S01的高斯模糊强度响应：固定4个已评估条件');ax.legend();ax.grid(alpha=.18);save(fig,'14_S01模糊强度响应.png')

# 15 paired ablations: S01 is the only reference.
abnames=['F01_NO_C0','F02_NO_RESCUE','R11_NO_RESNET8','R12_NO_HIRES19','R13_NO_DENSENET121','R14_SEED42','R20_MEAN_PROB','R26_EQUAL_LOGIT','R21_RAW_LOGIT','R22_STAGE1','R23_OLD_DEGRADE','R24_NO_RIDGE','R25_SIGNED']
ablabels=['去初判分支','去救援委员会','救援去ResNet8','救援去HiRes19','救援去DenseNet121','小网络仅种子42','概率均值融合','等权logit融合','去融合标准化','救援仅第一阶段','救援使用基础退化','去L2正则','允许负融合系数']
cis={r['name']:r for r in readcsv(EV/'成对差值与条件bootstrap区间.csv')};fig,axs=plt.subplots(1,2,figsize=(12,8),sharey=True)
for ax,key,title in zip(axs,['accuracy','held_accuracy'],['标准16视图加权准确率变化','4压力视图平均准确率变化']):
 for j,name in enumerate(abnames):
  r=cis[name];d=float(r[key+'_delta'])*100;lo=float(r[key+'_ci_low'])*100;hi=float(r[key+'_ci_high'])*100;ax.errorbar(d,j,xerr=[[d-lo],[hi-d]],fmt='o',capsize=3,color=COL[0] if d>=0 else COL[3])
 ax.axvline(0,color='#475569',lw=1);ax.set_xlabel('相对于S01的变化（百分点）');ax.set_title(title);ax.grid(axis='x',alpha=.15)
axs[0].set_yticks(range(len(abnames)),ablabels);axs[0].invert_yaxis();fig.suptitle('消融与替换：点估计及按原图聚类的条件95%区间',fontsize=17);fig.tight_layout(rect=[0,0,1,.94]);save(fig,'15_S01模块消融与条件区间.png')

# 16 show successes and failures at exactly the scored source view tensors.
items=[]
for ok in [True,False]:
 count=0;seen=set()
 for vi in [3,10,12,19]:
  for i in np.flatnonzero(truth & ((decision[vi]==truth)==ok)):
   if i not in seen:items.append((vi,int(i),ok));seen.add(int(i));count+=1
   if count==6:break
  if count==6:break
fig,axs=plt.subplots(4,6,figsize=(12,9))
for n,(vi,i,ok) in enumerate(items):
 row=0 if ok else 2;col=n%6
 for ax,z,tt in [(axs[row,col],xx[i],f'训练 #{i+1} · 人脸原图'),(axs[row+1,col],vfull[vi,i],f'{names[vi]}\nS01：{"人脸" if decision[vi,i] else "非人脸"}')]:ax.imshow(z,cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_title(tt,fontsize=9);ax.axis('off')
fig.suptitle('模糊人脸的折外识别：上两行成功，下两行漏检；同时展示两类结果',fontsize=16);fig.tight_layout(rect=[0,0,1,.94]);save(fig,'16_S01退化成功与漏检.png')
csvout('退化展示索引.csv',[{'source_id':i+1,'view':vi,'view_name':names[vi],'true_label':1,'predicted_label':1 if decision[vi,i] else -1,'correct':bool(ok)} for vi,i,ok in items])

# 17 source-flow frequencies, purely logical counts rather than timing.
calls=7+14*(~initial)+7*((~initial)&(votes[0]!=votes[1]));dist=np.array([(calls==n).mean(1) for n in [7,21,28]])
fig,ax=plt.subplots(figsize=(8,4.6));pos=np.arange(3);vals=dist[:,:16]@W
ax.bar(pos,vals*100,color=COL[:3]);ax.set_xticks(pos,['初判即结束\n7个成员评分','前两组救援一致\n21个成员评分','需要第三组救援\n28个成员评分']);ax.set_ylabel('标准16视图加权比例（%）');ax.set_ylim(0,max(vals)*100+12)
for i,v in enumerate(vals):ax.text(i,v*100+1,f'{v:.2%}',ha='center')
ax.set_title('S01条件执行路径：基于源图折外决策的逻辑回放');save(fig,'17_S01条件执行路径.png')
dump('条件执行源图逻辑统计.json',{'standard_weighted_path_fraction':dict(zip(['7','21','28'],vals.tolist())),'standard_weighted_mean_members':float(np.array([7,21,28])@vals),'note':'Replay of source-fold decisions, not actual GPU timing; Dense prefix/norm5 shared; different members have different costs.'})

dump('报告图像字体核验.json',FONTS)
summary={'metrics':metrics,'views':vr,'absolute_ci':ci,'compiled_gates':[{'weights':g['weights'],'decision_bias':g['decision_bias']} for g in meta['compiled_gates']],'parameter_count':15294620,'wrong_source_ids':(wrong+1).tolist(),'source_images':800,'source_views':20,'logical_source_calls':{'fractions':vals.tolist(),'mean_members':float(np.array([7,21,28])@vals)}}
(OUT/'report_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
dump('图表数值核验.json',{'source_only':True,'S01_compiled_source_decisions_rebuilt':16000,'matches_locked_source_decisions':True,'source_view_tensor_hashes_match_score_banks':True,'no_training_or_retuning':True,'font':str(FONT),'source_figures_generated_or_copied':len(FONTS),'model_weight_hash_verified':True})
print(json.dumps({'metrics':metrics,'figures':len(FONTS),'source_calls':summary['logical_source_calls']},ensure_ascii=False))
