"""Write the S01 final-model manuscript using audited source-only evidence."""
from pathlib import Path
import csv,json,re,shutil

ROOT=Path(__file__).resolve().parents[1];P=ROOT/'outputs/S01最终模型完整报告';EV=P/'evidence'
s=json.loads((P/'report_summary.json').read_text());m=s['metrics']
def readcsv(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def tab(head,rows):return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+'\n'.join('| '+' | '.join(str(x).replace('|','／') for x in row)+' |' for row in rows)
def pct(x,d=4):return f'{100*float(x):.{d}f}%'
def pp(x):return f'{100*float(x):+.3f}'
roles=readcsv(EV/'原图划分人数.csv')
role_table=tab(['用途','配对','FIT','ES','META','SEL','外层评价'],[['流程评价' if r['type']=='nested' else '最终部署',r['tag'],r['fit'],r['es'],r['meta'],r['selection'],r['outer']] for r in roles])
weights=json.loads((EV/'28网络部署权重与轮次.json').read_text())
def branch(name):return '初判B' if name=='C0_reference' else '救援'+name[6]
epoch_table=tab(['分支','架构','种子','阶段1选中/执行','阶段2选中/执行'],[[branch(r['branch']),r['kind'],r['seed'],f"{r['stage1_epoch']}/{r['stage1_epoch'] if r['branch']=='C0_reference' else (30 if r['kind']=='densenet121' else 80)}",f"{r['stage2_epoch']}/{r['stage2_epoch'] if r['branch']=='C0_reference' else (20 if r['kind']=='densenet121' else 40)}"] for r in weights])
th=readcsv(EV/'三组阈值搜索复算.csv')
threshold_table=tab(['救援组','META原图','SEL原图','训练选择的τ','候选数','等价区间宽度','SEL加权BA'],[[r['pair'],r['META_originals'],r['SEL_originals'],r['threshold'],r['candidates'],f"{float(r['interval_width']):.9g}",pct(r['SEL_weighted_BA'])] for r in th])
compiled_table=tab(['救援组','w_R','w_H','w_D','γ'],[[i+1,*[f'{v:.12f}' for v in g['weights']],f"{g['decision_bias']:.12f}"] for i,g in enumerate(s['compiled_gates'])])
view_table=tab(['条件','准确率','人脸召回率','非人脸特异度'],[[r['name'],pct(r['accuracy']),pct(r['face_recall']),pct(r['nonface_specificity'])] for r in s['views']])
summary_table=tab(['S01内部评价指标','结果'],[[name,pct(m[key])] for name,key in [('原图准确率','clean_accuracy'),('标准16视图加权准确率','weighted_accuracy'),('标准16视图加权人脸召回率','weighted_recall'),('标准16视图加权非人脸特异度','weighted_specificity'),('标准面板最差视图准确率','worst_standard_accuracy'),('4压力视图平均准确率','held_accuracy'),('4压力视图平均人脸召回率','held_recall'),('4压力视图平均非人脸特异度','held_specificity')]])
ci_table=tab(['评价指标','点估计','条件95%区间'],[[{'clean_accuracy':'原图准确率','weighted_accuracy':'标准加权准确率','held_accuracy':'压力平均准确率'}[k],pct(v['point']),f"[{pct(v['low'])}, {pct(v['high'])}]"] for k,v in s['absolute_ci'].items()])
ab={r['name']:r for r in readcsv(EV/'源图全部消融结果.csv')};ci={r['name']:r for r in readcsv(EV/'成对差值与条件bootstrap区间.csv')}
sel=[('F01_NO_C0','去初判分支，保留三组救援'),('F02_NO_RESCUE','去救援委员会，仅初判分支'),('R11_NO_RESNET8','救援去ResNet8并重新校准'),('R12_NO_HIRES19','救援去HiRes19并重新校准'),('R13_NO_DENSENET121','救援去DenseNet并重新校准'),('R14_SEED42','小网络仅种子42并重新校准'),('R15_SEED7','小网络仅种子7并重新校准'),('R16_SEED2026','小网络仅种子2026并重新校准'),('R20_MEAN_PROB','救援概率均值＋独立阈值'),('R26_EQUAL_LOGIT','救援等权logit＋独立阈值'),('R21_RAW_LOGIT','无融合标准化，重新校准'),('R22_STAGE1','救援仅第一阶段权重并重校准'),('R23_OLD_DEGRADE','救援使用基础退化训练并重校准'),('R24_NO_RIDGE','去L2正则，重新拟合'),('R25_SIGNED','允许负融合系数，重新拟合')]
ab_table=tab(['组','处理','标准加权准确率','人脸召回率','非人脸特异度','压力平均准确率'],[['S01','最终完整模型',pct(m['weighted_accuracy'],3),pct(m['weighted_recall'],3),pct(m['weighted_specificity'],3),pct(m['held_accuracy'],3)]]+[[name,title,*[pct(ab[name][k],3) for k in ['accuracy','recall','specificity','held_accuracy']]] for name,title in sel])
ab_ci_table=tab(['组','标准Δ百分点','标准条件95%区间','压力Δ百分点','压力条件95%区间'],[[name,pp(ci[name]['accuracy_delta']),f"[{pp(ci[name]['accuracy_ci_low'])}, {pp(ci[name]['accuracy_ci_high'])}]",pp(ci[name]['held_accuracy_delta']),f"[{pp(ci[name]['held_accuracy_ci_low'])}, {pp(ci[name]['held_accuracy_ci_high'])}]"] for name,title in sel])
full_ab_table=tab(['组','删改内容','标准准确率','人脸召回','非人脸特异度','压力准确率'],[[r['name'],r['title'].replace('删除原判别器','删除初判分支').replace('旧B0退化策略','基础退化策略').replace('旧阈值','既有阈值').replace('旧系数','既有系数').replace('旧融合','既有融合'),*[pct(r[k],3) for k in ['accuracy','recall','specificity','held_accuracy']]] for r in ab.values() if r['name'].startswith(('F','R'))])
hyper_path=Path('/tmp/s01_hyperparameters.md')
assert hyper_path.exists(),'Independent hyperparameter audit must be ready.'
hyper=hyper_path.read_text()
# Use one notation and protect table cells from unescaped absolute-value bars.
hyper=hyper.replace(r"\sum_k|L_k||w_k|+|b'|",r"\sum_k\lvert l_k\rvert\lvert w_k\rvert+\lvert\gamma_j\rvert")
hyper=hyper.replace(r'\|\theta\|',r'\lVert\theta\rVert')
for old,new in [('p_{0R}','p_{BR}'),('p_{0H}','p_{BH}'),('p_{0D}','p_{BD}'),('p_0','p_B'),('L_{jk}','l_{jk}'),('m_j','h_j'),("b'_j",r'\gamma_j')]:hyper=hyper.replace(old,new)
hyper=hyper.replace(r'\[','$$').replace(r'\]','$$').replace(r'\(','$').replace(r'\)','$')
(EV/'超参数完整账本.md').write_text(hyper,encoding='utf-8')
# Place the independent ledger under one manuscript section.
hyper=re.sub(r'^# .+\n','',hyper,count=1)
hyper=re.sub(r'^(#{2,4}) ',lambda mt:'#'+mt.group(1)+' ',hyper,flags=re.M)
for file in ['/tmp/s01_final_report_audit.md','/tmp/s01_model_audit.md']:
 if Path(file).exists():shutil.copyfile(file,EV/Path(file).name)

report=r'''# S01最终模型：面向低分辨率与模糊人脸的共享特征、校准救援与条件推理

> **报告范围**：以S01为最终交付模型，依据实际权重、运行代码、训练记录、分折、校准配方和消融记录整理。本文只分析原训练集及其内部评价，图中编号均为训练数据矩阵的原始行号；所有图片中的中文使用**微软雅黑**。本次报告整理没有重新训练或重新选择参数。

## 1. 模型定义与基本流程

S01完成“人脸／非人脸”二分类，不涉及人员身份识别。它由一个**初判分支B**与三个**救援分支**组成，每个分支包含三个ResNet8、三个HiRes19和一个DenseNet121分类成员。模型共保留28个逻辑评分成员，通过共享相同的冻结DenseNet特征模块，实际存储 **15,294,620个神经参数**。

最终决策遵循：初判分支通过就输出人脸；初判未通过时，要求三组救援中至少两组通过。运行时先执行前两组救援，它们结论相同时即可结束，结论不同时再执行第三组。

![S01整体结构与条件执行](assets/01_S01整体结构.png)

**图1**　S01的完整结构和条件执行。共享的是DenseNet的冻结参数与公共前缀特征；四个可训练尾部具有各自权重。三个小网络种子的概率先在家族内平均，再进入分支决策。

设训练图像为 $x\in[0,1]^{19\times19}$，输出标签为 $\widehat y\in\{-1,+1\}$。$+1$为人脸，$-1$为非人脸。对分支 $j\in\{B,1,2,3\}$、家族 $k\in\{R,H,D\}$，定义

$$
p_{jk}(x)=\frac1{|S_k|}\sum_{s\in S_k}
\sigma\bigl(f_{jks}(x)\bigr),\qquad
S_R=S_H=\{42,7,2026\},\quad S_D=\{42\},
$$

其中 $\sigma(z)=1/(1+e^{-z})$。初判概率为

$$
p_B(x)=\frac{p_{BR}(x)+p_{BH}(x)+p_{BD}(x)}3,
\qquad B(x)=\mathbb1[p_B(x)\ge0.37].
$$

对救援分支，将三个家族概率转为经过裁剪的logit $l_{jk}(x)$，执行

$$
h_j(x)=\sum_{k\in\{R,H,D\}}w_{jk}l_{jk}(x)+\gamma_j,
\qquad v_j(x)=\mathbb1[h_j(x)\ge0].
$$

最终规则为

$$
\widehat y(x)=
\begin{cases}
+1,&B(x)=1\;\lor\;\displaystyle\sum_{j=1}^{3}v_j(x)\ge2,\\
-1,&\text{其他情况}.
\end{cases}
$$

**这里没有一个可以直接解释为“最终人脸概率”的统一输出。** $p_B$是初判分数，$h_j$是线性决策边际，最后由逻辑关系给出标签。救援的零阈值与初判的0.37处于不同数值空间，不能相互替代。

## 2. 原训练数据与可视化分析

### 2.1 数据规模与标签

$$
\mathcal D=\{(x_i,y_i)\}_{i=1}^{800},\qquad
x_i\in[0,1]^{19\times19},\quad y_i\in\{-1,+1\}.
$$

| 项目 | 核验结果 |
| --- | --- |
| 图像文件及矩阵 | `train_data.mat`中的`train_data`，800×361，原存储float64 |
| 标签文件 | `train_label.mat`，800个±1标签 |
| 人脸 / 非人脸 | 400 / 400 |
| 恢复后张量 | 800×1×19×19，float32 |
| 像素范围 | [0,1]；没有非有限像素 |
| 像素完全相同的原图 | 未发现；800张均不同 |
| 描述性像素均值 / 标准差 | 约0.4176 / 0.2129，不作为全局归一化参数 |

没有身份、拍摄来源等分组元数据，因此像素去重不能证明不同角色中的人物或场景完全独立。[数据核验](evidence/训练数据核验.json)

![训练原图样本](assets/02_训练样本.png)

**图2**　两类各按原始编号等间隔抽取20张。原图采用最近邻放大展示，显示面积增大不会增加真实细节。

### 2.2 强度、纹理与空间布局

![训练数据统计](assets/03_数据统计.png)

**图3**　类别均值、逐像素标准差和每图描述统计。人脸均值图呈现较一致的粗布局，非人脸纹理更分散；这只是类别总体特征，不表示每张图都具有相同姿态和对齐。

统计量定义为

$$
m_i=\frac1{361}\sum_{u,v}x_i(u,v),\qquad
s_i=\sqrt{\frac1{361}\sum_{u,v}[x_i(u,v)-m_i]^2},\qquad
q_i=\operatorname{Var}[\Delta x_i].
$$

拉普拉斯 $\Delta$ 在每张19×19图像的两个空间轴上独立计算，不跨原图编号轴。$q_i$同时受到纹理、边缘、噪声和模糊影响，不能当作真实模糊标签。S01没有把这些统计量输入融合器，也没有依据它们手工分流。

### 2.3 低频信息的描述性分析

![训练图像频谱描述](assets/09_训练数据频谱描述.png)

**图4**　减去每图均值后，计算二维离散傅里叶功率谱；右图统计半径不超过0.15周期/像素的非直流能量比例。该半径仅用于此处数据分析，不是S01的滤波截止频率或可训练参数。

$$
X_i(a,b)=\sum_{u,v}[x_i(u,v)-m_i]
e^{-2\pi\mathrm i(au/19+bv/19)},\qquad
\rho_i=\frac{\sum_{0<\|\boldsymbol f\|\le0.15}|X_i(\boldsymbol f)|^2}
{\sum_{\|\boldsymbol f\|>0}|X_i(\boldsymbol f)|^2+\varepsilon}.
$$

人脸结构可以在低频区域留下线索，但非人脸也可能具有平滑结构。不能凭低频占比高就判为人脸。**S01没有频谱分解支路**；该分析解释低分辨率结构与退化问题，不增加模型计算流程。

## 3. 原图划分、校准与评价协议

### 3.1 先划分原图，再生成增强视图

对同一分类器—校准器配对，使用下列角色：

| 角色 | 用途 |
| --- | --- |
| FIT | 更新神经网络可训练参数 |
| ES | 依据验证BCE选择第一、二阶段检查点 |
| META | 拟合分数标准化统计量、融合系数与截距 |
| SEL | 对冻结的融合评分选择分类阈值 |
| OUT | 对应流程的外层评价 |

同一配对的五种原图集合两两不交；一张原图及其所有退化版本始终属于同一角色。交叉验证允许同一原图在不同配对中承担不同角色，隔离要求应逐配对检查。

### 3.2 训练集内部三折流程评价

800张原图首先分层三折，每折外层266或267张。剩余533或534张内部再组成三个配对，每组神经拟合约284张、ES71或72张、META118张、SEL59或60张。该外折图像不参加对应新增专家、融合或阈值的拟合与选择。

三个外折拼接得到800张原图的折外（OOF）判断。每张原图还产生16标准视图和4保留压力视图。16,000个评价项共享800个来源，不能视为16,000个独立样本。

### 3.3 最终部署的参数角色

初判分支使用800张原图进行固定轮次拟合；轮次由训练集内部记录决定，不用全量拟合误差选择。三组救援分别采用426、426、427张FIT原图，ES均107张；其余原图分为META178／178／177和SEL89／89／89。部署保留各权重与其校准参数的对应关系。

@@ROLE_TABLE@@

[完整原图划分](evidence/原图划分完整记录.json)。这些角色规模也解释了为何需要区分“流程评价”与“最终部署权重评价”：外折专家的拟合样本较少，最终部署使用的初判和救援权重并非外折评价中的同一组权重。

此外，架构与初判0.37来自同一训练数据较早的内部开发。本项目也经过多轮探索。因此下文准确表述为**训练集内部流程评价**，不是全新数据上的完全独立泛化认证。

## 4. 数据预处理

### 4.1 还原图像方向

图像行向量采用MATLAB式展开方向，实际代码为：

```python
def image(a):
    return torch.from_numpy(
        a.reshape(-1,19,19).transpose(0,2,1).copy()
    ).float()[:,None]
```

训练和推理必须一致交换空间轴，不能只做`reshape`。输入统一为float32、单通道、19×19。

### 4.2 小网络逐图标准化

ResNet8和HiRes19各自使用

$$
\mu_x=\frac1{HW}\sum x(u,v),\qquad
\sigma_x=\sqrt{\frac1{HW}\sum[x(u,v)-\mu_x]^2},
$$

$$
\widetilde x(u,v)=\operatorname{clip}\left(
\frac{x(u,v)-\mu_x}{\max(\sigma_x,0.01)},-3,3\right).
$$

它减轻绝对亮度和对比度差异；标准差下限和裁剪控制极端数值。低对比图中的噪声仍可能被放大，因此需要退化训练配合。

### 4.3 DenseNet输入与共享前缀

Dense路径保留原强度，双线性插值到128×128，复制为三通道，并按ImageNet统计归一化：

$$
\widetilde x_c=\frac{\operatorname{Resize}_{128}(x)-\mu_c}{s_c},\quad
\boldsymbol\mu=(0.485,0.456,0.406),\quad
\boldsymbol s=(0.229,0.224,0.225).
$$

插值适配网络输入，不恢复已经丢失的细节。相同输入经过共享前缀 $E$ 得到

$$
z(x)=E(\widetilde x)\in\mathbb R^{512\times4\times4}.
$$

四个Dense尾部按各自权重处理该特征。特征只需生成一次；不同block4后的特征不能互相替代。

```python
# 节选本模型使用的预处理规则
if kind != 'densenet121':
    x = ((x-x.mean((2,3),keepdim=True)) /
         x.std((2,3),correction=0,keepdim=True).clamp_min(.01)).clamp(-3,3)
else:
    x = F.interpolate(x,(128,128),mode='bilinear',
                      align_corners=False).repeat(1,3,1,1)
    x = (x-x.new_tensor([.485,.456,.406])[None,:,None,None]) / \
        x.new_tensor([.229,.224,.225])[None,:,None,None]
```

![预处理可视化](assets/04_预处理.png)

**图5**　两个输入体系。伪彩色仅显示标准化后的数值，不表示模型接收额外颜色信息。[完整预处理实现](code/networks.py)

## 5. 网络结构及共享参数组织

### 5.1 最终模型的实际组成

| 组件 | 数量 | 单个或合计参数 | 功能 |
| --- | --- | --- | --- |
| ResNet8 | 12 | 每个78,529；合计942,348 | 低分辨率残差特征 |
| HiRes19 | 12 | 每个76,673；合计920,076 | 保持19×19空间网格与粗布局 |
| Dense冻结共享前缀 | 1 | 4,793,728 | 公共监督预训练特征 |
| 共享末端norm5 | 1 | 2,048 | 固定归一化参数与运行统计 |
| 独立denseblock4 | 4 | 每个2,158,080；合计8,632,320 | 各分支可训练特征尾部 |
| Dense二分类头 | 4 | 每个1,025；合计4,100 | 各分支分类输出 |
| **神经参数总计** | — | **15,294,620** | 不含BN缓冲区、激活和校准元数据 |

按float32神经参数计算约58.34 MiB；实际权重文件还包括缓冲区、门控配置和序列化信息，不能用该数字代替文件大小或峰值内存。

四个Dense分支的共享前缀与norm5由逐张量键、形状、类型和内容核验确定相同。共享norm5意味着使用相同的固定参数，但各分支仍需对自身block4输出执行norm5。

### 5.2 ResNet8：原生低分辨率残差建模

$$
1\times19\times19
\xrightarrow{3\times3,16}16\times19\times19
\xrightarrow{\mathrm{Res}(16,16,1)}16\times19\times19
\xrightarrow{\mathrm{Res}(16,32,2)}32\times10\times10
\xrightarrow{\mathrm{Res}(32,64,2)}64\times5\times5
\xrightarrow{\mathrm{Flatten,Dropout}(0.2),\mathrm{FC}}\mathbb R.
$$

残差块满足

$$
h_{l+1}=\operatorname{ReLU}\{F_l(h_l)+S_l(h_l)\}.
$$

$F_l$是两层3×3卷积、GroupNorm与激活，$S_l$是恒等映射或匹配步长/通道的1×1投影。残差学习有利于优化，但本任务的小型结构仍需要自己的数据证据。[ResNet原论文](https://arxiv.org/abs/1512.03385)

小容量和原生19×19输入适合当前样本规模；两次下采样又可能损失微弱结构，不能把残差连接本身等同于去模糊。

### 5.3 HiRes19：保持网格和多尺度粗结构

HiRes19先用3×3卷积映射到32通道，再通过膨胀率1、2、3的三个残差块，每块含两层3×3卷积，始终保持19×19网格。膨胀卷积可写为

$$
(W*_dh)(u,v)=\sum_{a,b}W(a,b)h(u+da,v+db).
$$

末端拼接全局均值与3×3空间区域均值：

$$
\phi(h)=\left[\frac1{HW}\sum_{u,v}h(u,v);\;
\left\{\frac1{|B_{ab}|}\sum_{(u,v)\in B_{ab}}h(u,v)\right\}_{a,b=0}^{2}\right]
\in\mathbb R^{320}.
$$

实际19像素轴的区域为`[0:7]`、`[6:13]`、`[12:19]`，遵循自适应平均池化分区，有一像素重叠。分类头是`320→64→1`，中间ReLU和Dropout0.2。

```python
self.features = nn.Sequential(
    conv(1,32), DilatedBlock(32,1),
    DilatedBlock(32,2), DilatedBlock(32,3))
self.head = nn.Sequential(
    nn.Linear(320,64), nn.ReLU(),
    nn.Dropout(.2), nn.Linear(64,1))
```

它希望表达眼、鼻、嘴之间的粗相对位置；这是自定义小网络，不是原版HRNet，也不包含显式低频分解。高分辨率网格仍可能保留噪声，膨胀采样也可能有栅格效应。

两个小网络均使用4组GroupNorm，统计不依赖batch维。GroupNorm的功能依据与本任务实测贡献应区分；本轮未单独重训比较所有归一化种类。[Group Normalization原论文](https://arxiv.org/abs/1803.08494)

### 5.4 DenseNet121：监督迁移与独立尾部

DenseNet通过通道拼接复用前层特征：

$$
h_l=H_l([h_0,h_1,\ldots,h_{l-1}]).
$$

其密集块层数为6、12、24、16，增长率32。128×128输入经过前缀得到512×4×4，block4再得到1024×4×4，随后norm5、ReLU、全局均值池化和二分类头。[DenseNet原论文](https://arxiv.org/abs/1608.06993)

采用常规ImageNet监督预训练初始化，每个Dense分支仅更新block4与新二分类头，共2,159,105参数。全部BN运行统计固定；block4中BN仿射参数仍更新，norm5参数冻结。小网络随机初始化。

因此，“本任务拟合和选型只依赖原训练集”不意味着Dense全部初始权重都从这800张随机学得。预训练来源与冻结范围必须披露。

三家族的互补动机分别来自小容量残差、空间布局和监督迁移/强度路径；它们在本数据上的作用由消融评估，而不是由架构名称决定。S01没有可学习的逐图家族权重或注意力模块。

## 6. 两阶段训练与分支专长

### 6.1 第一阶段：原图内容与常规增强

将标签变换为 $t_i=(y_i+1)/2$。使用稳定的二元logit交叉熵：

$$
\ell(z,t)=\log(1+e^z)-tz,\qquad
\mathcal L_1=\frac1B\sum_{i=1}^{B}\ell(f_\theta(A(x_i)),t_i).
$$

第一阶段以原训练图像为内容基础，使用旋转±0.12弧度、平移约±0.8像素、0.5概率翻转、增益[0.5,1.5]、偏移[-0.1,0.1]与σ=0.015高斯噪声，裁剪到[0,1]。它不是逐像素完全不变的训练，但不使用第二阶段的严重复合退化策略。

### 6.2 初判分支的第二阶段

初判分支从对应第一阶段检查点继续优化。常规增强后，以0.5概率加入基础退化，每次每原图一个视图，使用BCE。基础退化包括高斯σ∈{0.5,1,1.5}、双线性降至9/12/15再回19，或高斯σ0.5/1加降至12/15再回19。

### 6.3 救援分支的第二阶段

各救援成员从对应第一阶段最佳检查点开始，生成常规视图 $c_i=A_0(x_i)$ 和两个独立复合退化视图：

$$
d_i^{(1)}=T_i^{(1)}(c_i),\quad d_i^{(2)}=T_i^{(2)}(c_i),
$$

$$
\mathcal L_2=\frac1{3B}\sum_{i=1}^{B}
\left[\ell(f_\theta(c_i),t_i)
+\ell(f_\theta(d_i^{(1)}),t_i)
+\ell(f_\theta(d_i^{(2)}),t_i)\right].
$$

正常/退化训练权重为1/3与2/3。两个退化共享常规几何视图，但退化采样独立；两类图像使用相同策略，避免把“有模糊”变成人脸标签捷径。

```python
clean = base_view(x[ii])
d1, d2 = corrupt(clean), corrupt(clean)
images = torch.cat([clean,d1,d2])
targets = target[ii].repeat(3)
logits = net(prep(images.to(device),kind)).flatten()
loss = F.binary_cross_entropy_with_logits(logits,targets.to(device))
```

常规视图保留弱几何和翻转，增益[0.8,1.2]、偏移[-0.05,0.05]。每次退化均匀抽取五种基础类型之一、强度 $s\sim U(0.5,1)$：无基础空间滤波、高斯、任意方向运动、高斯后降采样、降采样。还叠加对比度、噪声、可选弱模糊和JPEG，改变噪声与滤波顺序。具体范围列入超参数账本。

### 6.4 为什么有助于模糊结构识别

理想高斯模糊的频率响应为

$$
x_{\mathrm{blur}}=h_\sigma*x,\qquad
\widehat x_{\mathrm{blur}}(\boldsymbol f)
=\exp[-2\pi^2\sigma^2\|\boldsymbol f\|^2]\widehat x(\boldsymbol f).
$$

高频衰减更明显，粗空间布局相对保留。对同一内容的常规和退化视图施加相同监督，可减少对易消失细纹理的依赖；HiRes19的空间表达和其他家族提供不同证据。

这是一种机制解释，不表示模型恢复了丢失的细节。噪声、运动和强模糊也会破坏粗结构；S01仍可能漏检，必须同时报告成功和失败样本。

## 7. 验证损失、检查点与实际训练轮次

### 7.1 固定16视图与保留压力视图

![标准退化面板](assets/05_固定退化面板.png)

**图6**　标准16视图。原图权重为0.5，其余15种各为1/30，用于阶段2ES、META和SEL的相应目标。

![保留压力视图](assets/06_保留压力面板.png)

**图7**　圆盘散焦、JPEG25、椒盐噪声、模糊＋散斑作为保留压力视图，不用于本轮检查点、融合器或阈值选择。它们与增强可能有物理相似性，不等于所有退化机制从未见过。

训练梯度使用`BCEWithLogits`；实际验证和固定诊断先把概率裁剪到 $[10^{-6},1-10^{-6}]$：

$$
\widehat p_{vi}=\operatorname{clip}(p_\theta(T_vx_i),10^{-6},1-10^{-6}),
$$

$$
\mathcal L_{\mathrm{ES,16}}=
\sum_{v=0}^{15}\omega_v\frac1{n_{\mathrm{ES}}}\sum_i
\left[-t_i\log\widehat p_{vi}-(1-t_i)\log(1-\widehat p_{vi})\right],
\quad \omega_0=\tfrac12,\quad\omega_{v>0}=\tfrac1{30}.
$$

第一阶段使用ES原图BCE；第二阶段使用上述16视图加权BCE。在第1轮、每个偶数轮和最后一轮检查，保存最低BCE权重；代码执行满阶段上限再恢复检查点，不是patience提前终止。

### 7.2 优化器与学习率

| 设置 | ResNet8 | HiRes19 | DenseNet121 |
| --- | --- | --- | --- |
| 阶段1初始LR | 0.001 | 0.001 | 0.0001 |
| 阶段2初始LR | 0.0001 | 0.0001 | 0.00003 |
| 折内阶段1执行上限 | 80 | 80 | 30 |
| 折内阶段2执行上限 | 40 | 40 | 20 |
| 原图batch上限 | 64 | 64 | 32 |

使用AdamW、weight decay0.001，默认betas=(0.9,0.999)、eps=1e-8；阶段切换时重新建立优化器。余弦学习率为

$$
\eta_e=\frac{\eta_0}{2}\left[1+\cos\frac{\pi(e-1)}{T_{\max}}\right].
$$

折内 $T_{\max}$ 为阶段上限；初判最终全800固定长度拟合的 $T_{\max}$ 为实际执行轮数。救援三视图使每批前向图像数至多约192、192、96。**这些初始LR和上限是预设，不是逐项搜索所得。**

### 7.3 真实配对曲线

![S01训练流程配对记录](assets/07_训练验证配对曲线.png)

**图8**　流程评价外折1、内部配对1、种子42，FIT284张、ES71张。第一行为阶段1、第二行为救援阶段2；每个点来自真实日志。固定退化拟合BCE仅在部分轮次记录，连线不表示中间轮次都有测量。

比较拟合与验证时应匹配视图口径：原图对原图，固定16视图对固定16视图。在线随机增强目标与固定原图损失输入分布不同，二者差距不能直接当作泛化差距。

验证BCE可能因少量高置信错误而上升。例如真实人脸的分数由0.9变成0.01时，其BCE从约0.105变为4.605。训练损失继续下降不意味着应该无限加轮次，应由ES决定检查点。

初判最终全800记录没有逐轮验证历史，本文没有为它补造曲线。图8是对应训练流程的真实嵌套记录，不冒充最终全量拟合过程。

### 7.4 全部28个分类成员的轮次

“选中/执行”分开列出：如6/30表示实际运行30轮、保存第6轮。初判全800按来源折的家族中位轮次固定执行，没有全量ES；救援保留各自ES所选权重。

@@EPOCH_TABLE@@

三个救援DenseNet的阶段2实际执行20轮，分别选中18、20、18轮。不能把某个分支的轮数直接套到全部Dense成员。[成员及父权重清单](evidence/28网络部署权重与轮次.json)

## 8. 校准器训练：由META学习证据组合

### 8.1 概率与logit的数值规则

家族种子先平均sigmoid概率，不平均原始logit。救援特征按R/H/D顺序计算：

$$
\bar p_{jk}=\operatorname{clip}(p_{jk},10^{-5},1-10^{-5}),\qquad
l_{jk}=\operatorname{clip}\left(\log\frac{\bar p_{jk}}{1-\bar p_{jk}},-8,8\right).
$$

裁剪防止极端置信产生无穷值。成员概率、种子均值及logit先采用float32，随后升float64进行融合；这一顺序属于实际数值实现。

### 8.2 META标准化与类别平衡损失

对每组META原图数 $n$，标准化使用每原图等权和视图权重：

$$
\mu_k=\sum_v\frac{\omega_v}{n}\sum_i l_{vik},\quad
s_k=\max\left(0.05,\sqrt{\sum_v\frac{\omega_v}{n}\sum_i(l_{vik}-\mu_k)^2}\right),\quad
z_{vik}=\frac{l_{vik}-\mu_k}{s_k}.
$$

这一步不按类别平衡；拟合损失另使用

$$
\beta_{vi}=\omega_v\begin{cases}1/(2n_+),&t_i=1,\\1/(2n_-),&t_i=0.\end{cases}
$$

学习目标为

$$
\min_{\boldsymbol a,b}
\sum_{v,i}\beta_{vi}\left[\log(1+e^{u_{vi}})-t_i u_{vi}\right]
+0.01\sum_k a_k^2,\qquad u_{vi}=b+\sum_k a_kz_{vik},
$$

$$
0\le a_k\le4,\qquad -4\le b\le4.
$$

这是特征固定后的凸约束问题，使用L-BFGS-B、解析梯度、maxiter300、ftol1e-12。截距不受L2惩罚，初始三个系数均1/3、截距0。0.01、上下界、容差均为预设，并未经过λ网格搜索。

```python
def objective(theta):
    z = Xstd @ theta[:-1] + theta[-1]
    error = beta * (expit(z)-target)
    loss = beta @ (np.logaddexp(0,z)-target*z)
    loss += .01*np.sum(theta[:-1]**2)
    grad = np.r_[Xstd.T@error+.02*theta[:-1],error.sum()]
    return float(loss),grad
```

分数变大影响救援判断，但学习到的系数在部署时固定；它们不是针对每张图像重新生成的专家权重。系数也不要求加和为1，不能直接作为架构贡献百分比。

## 9. 分类阈值、部署边际与零阈值的含义

### 9.1 初判阈值0.37的来源

该设置来自早期原训练集内部的640/160划分、8视图开发选择：原图权重0.5、其他7种各0.5/7，阈值候选为0.01至0.99共99点，最大化加权准确率；同分先距0.5近、再选较小者。

$$
\tau_B^*=\arg\max_{\tau\in\{0.01,0.02,\ldots,0.99\}}
\sum_{v=0}^{7}\omega_v^{\mathrm{dev}}\operatorname{Acc}_v(\tau)=0.37.
$$

S01继承并冻结该值，没有重新联合搜索整个最终级联。因此它有训练集内部选择来源，但不能称为当前全部模块的联合全局最优阈值。[搜索实现快照](code/historical_threshold_search.py)

### 9.2 救援阈值来自独立SEL

META拟合结束后，冻结融合器，以辅助分数 $g_j=\sigma(u_j)$ 在独立SEL原图的16视图上选择 $\tau_j$：

$$
\mathrm{BA}_v(\tau)=\frac12\left[
\frac{\sum_{i:t_i=1}\mathbb1[g_j(T_vx_i)\ge\tau]}{n_+}
+\frac{\sum_{i:t_i=0}\mathbb1[g_j(T_vx_i)<\tau]}{n_-}\right],
$$

$$
\tau_j^*=\arg\max_\tau\sum_{v=0}^{15}\omega_v\mathrm{BA}_v(\tau).
$$

实际代码枚举不同分数之间的区间中点、边界区间中点及0/1，覆盖可达分类划分。同分按最差视图BA、原图BA、区间宽度、距0.5距离、较小阈值和稳定索引依次选择，指标比较保留12位小数。

@@THRESHOLD_TABLE@@

![救援阈值选择](assets/08_独立阈值搜索.png)

**图9**　SEL上的阈值选择曲线；纵轴放大到0.75—1显示差异。它是选择过程，不是独立评价。每组89张原图的16视图仍只有89个独立来源。

长小数来自相邻分数中点，不代表统计精确到十几位。第2组等价区间仅约3.34×10⁻⁵，说明小样本阈值稳定性仍需关注。阈值目标是单个救援分支的BA，没有联合优化整个逻辑级联。

### 9.3 S01的实际边际参数

将训练所得统计、系数和SEL阈值合成为

$$
w_{jk}=a_{jk}/s_{jk},\qquad
\gamma_j=b_j-\sum_k a_{jk}\mu_{jk}/s_{jk}-\operatorname{logit}(\tau_j).
$$

由sigmoid严格单调性，

$$
g_j(x)\ge\tau_j\iff h_j(x)=\sum_k w_{jk}l_{jk}(x)+\gamma_j\ge0.
$$

@@COMPILED_TABLE@@

表值为便于阅读保留12位小数，实际模型保存原浮点数。**零是定义好的决策边界，不是重新搜索的分类阈值，也不表示人脸概率为0。** 成员网络本身的sigmoid仍用于生成家族概率；S01主融合路径不再显式计算辅助 $g_j$。

```python
clipped = np.asarray(families,dtype=np.float32).clip(1e-5,1-1e-5)
L = np.log(clipped/(1-clipped)).clip(-8,8).astype(np.float64)
margin = L @ weights + decision_bias
decision = margin >= 0
```

真实运行器在 $|h_j|$ 很接近零时，按相对数值容差回退到训练表述的比较，保留边界数值行为。模型同时保存校准来源信息用于回退与审计，并非只保存表中的12个标量。[实际运行器](code/c1_shared_runtime.py)

## 10. 条件执行的实现与数学性质

### 10.1 共享特征与短路代码

以下是与实际运行器控制逻辑一致的核心摘录，网络构建和完整边界处理见代码文件：

```python
encoded = shared_dense_prefix(preprocess_dense(images))
positive = score_initial_branch(images,encoded) >= .37
pending = np.flatnonzero(~positive)
if len(pending):
    v1 = rescue_vote(1,images,encoded,pending)
    v2 = rescue_vote(2,images,encoded,pending)
    same = v1 == v2
    positive[pending[same]] = v1[same]
    disputed = pending[~same]
    if len(disputed):
        positive[disputed] = rescue_vote(3,images,encoded,disputed)
labels = np.where(positive,1,-1)
```

若前两票相同，第三票无论为何都不改变多数结果；如果前两票不同，第三票直接决定多数。这是输入相关的执行路径，**不是根据图像质量学习权重的注意力路由**。

### 10.2 初判保留规则的作用与代价

初判正例集合包含于最终正例集合：

$$
\{x:B(x)=1\}\subseteq\{x:\widehat y(x)=+1\}.
$$

因此相对于单独初判分支，救援不会新增漏检，但也不能纠正初判已有的误报，并可能新增误报。总体准确率变化取决于救回真脸与新增误报的数量差。

非负 $w_{jk}$ 保证其他特征固定时，提高某家族证据不会降低 $h_j$；这是一项结构先验，不自动构成其不可替代性的实测证明。

### 10.3 源图流程中的执行路径分布

![条件执行路径统计](assets/17_S01条件执行路径.png)

**图10**　根据800张原图的外层折外分数回放条件逻辑，在标准16视图加权下，约50.325%由初判结束、48.946%由前两组救援一致结束、0.729%需要第三组救援。

每图逻辑成员调用数为

$$
N(x)=7+14\mathbb1[B(x)=0]
+7\mathbb1[B(x)=0\land v_1(x)\ne v_2(x)].
$$

该面板加权均值约14.0055。它是源图决策回放的逻辑计数，不是GPU耗时；共享Dense前缀后不同成员成本不同，不能把成员数量直接换算成延迟。

## 11. 超参数如何确定：完整账本

网络权重和融合系数属于**参数拟合**；检查点和阈值属于**内部选择**；结构、LR、增强范围等大多是**固定预设**；初判阈值等属于**有训练集来源的继承设置**。这些概念必须区分。下面按实际记录说明，不将未经搜索的参数写成已寻优。

@@HYPERPARAMETERS@@

## 12. S01分类结果可视化

### 12.1 原图混淆矩阵与正确样本

![S01混淆矩阵](assets/10_S01折外混淆矩阵.png)

**图11**　训练原图的流程折外评价：TP400、FN0、TN390、FP10，790/800正确，准确率98.75%；人脸召回100%，非人脸特异度97.5%，人脸预测精确率400/410≈97.561%。

![S01分类正确样本](assets/11_S01正确分类示例.png)

**图12**　按原图编号等间隔取正确样本，展示人脸与非人脸两类，不仅选择人脸成功案例。

### 12.2 全部原图错误

![S01原图全部错误](assets/12_S01全部原图错误.png)

**图13**　全部10个错误，依据原训练标签评价。图中给出外折、初判/救援路径和救援票数。它们均为非人脸误报，说明粗布局、边缘或纹理可能触发人脸证据；本报告没有人工改写训练标签以消除错误。

每张图的标签、外折、初判、票数和S01判断见[逐原图决策](evidence/S01训练原图折外决策.csv)。这些是流程评价中的对应嵌套配方，不能用最终部署组的阈值直接解释它们。

## 13. 鲁棒性分析与不确定性

### 13.1 指标与总体表现

$$
\mathrm{Acc}=\frac{\mathrm{TP}+\mathrm{TN}}N,\quad
\mathrm{Recall}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}},\quad
\mathrm{Specificity}=\frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FP}}.
$$

@@SUMMARY_TABLE@@

800原图两类平衡，所以按全部原图汇总的每视图BA与准确率相同；SEL可能为44／45类数量，阈值搜索因此明确采用BA。

### 13.2 逐退化结果

![逐视图鲁棒性](assets/13_S01逐视图鲁棒性.png)

**图14**　仅展示S01。浅色区为四个保留压力条件，各行纵轴覆盖全部数据点。标准加权平均较高，并不意味着每种强退化都接近满分。

@@VIEW_TABLE@@

![模糊强度响应](assets/14_S01模糊强度响应.png)

**图15**　在已测的高斯σ=0、1、1.5、2四条件中，准确率依次为98.75%、98.75%、97.25%、94.50%。σ=2时人脸召回94%，说明强模糊仍有损失；没有评估连续所有σ或更强退化，不能向外推断。

复合条件“高斯1.5＋噪声”的人脸召回为97.25%；“模糊＋低对比＋噪声”为98.50%。四压力视图中，模糊＋散斑的人脸召回为90%，是明显短板。不同条件涉及不同物理破坏，不能只用模糊强度单一数值排序。

### 13.3 同时展示退化成功与漏检

![退化识别成功和漏检](assets/16_S01退化成功与漏检.png)

**图16**　上两行是成功案例，下两行是漏检；展示原图与实际评分的退化张量。固定噪声按原外折顺序重新生成，并与评分缓存的图像张量哈希核对，避免显示的噪声图与被分类样本不一致。

### 13.4 条件区间及其边界

以原图为聚类单位，同一原图所有视图一起重采样，按外折×类别分层，5000次、随机种子20261116，得到：

@@CI_TABLE@@

这是**固定既有权重、阈值和流程后的条件区间**，不覆盖重训、超参数搜索、反复选型或数据域变化。不能把区间当作全项目无选择偏差的泛化保证。[计算记录](evidence/S01绝对指标条件区间.json)

## 14. 消融实验：检验各部分的实际作用

### 14.1 两类消融问题

**固定删除F组**保留其他融合系数和阈值；删除家族相当于去掉其标准化特征项。它检查部署配置的敏感性。**重新校准R组**保持匹配神经权重和原图角色，仅在META重新拟合、SEL重新选阈值；它检查删改后能否恢复分类能力。

下面以S01为唯一参考。删除某个救援家族时，初判分支中的该家族仍保留；单种子对照则同时改变初判与救援的小网络，Dense种子始终42。每组均使用相同来源外折、视图和评价权重。

本轮复用已训练的匹配权重，重新前向单成员并拟合小型校准器，没有为每个删改结构重训所有剩余骨干。因此结论是**在该权重与校准协议下的组件作用**，不是架构空间的不可替代性定理。

### 14.2 核心对照与条件区间

@@ABLATION_TABLE@@

![S01模块消融](assets/15_S01模块消融与条件区间.png)

**图17**　消融相对S01的百分点变化和原图聚类条件区间，标准与压力指标同时显示，避免只选择有利指标。

@@ABLATION_CI_TABLE@@

这些区间条件于既有模型，未校正多重探索，也不包含重训与选型方差。它们帮助判断点估计的量级与稳定性，不能强行把每个小差异解释为必要贡献。

### 14.3 可以支持哪些作用

**第二阶段与复合退化训练的支持较强。** 救援仅使用第一阶段权重并重新校准，标准准确率下降1.158个百分点、压力平均准确率下降2.281个百分点；使用基础退化路线时压力准确率下降2.625个百分点。它们支持第二阶段复合退化训练对当前流程的作用，但差异也包括额外更新和视图损失结构，不能将收益全部归因于某个噪声参数。

**ResNet8救援家族有明确的本面板贡献。** 去掉并重新校准后，标准准确率下降0.325个百分点、压力下降1.781个百分点，条件区间均偏负。HiRes19的删除主要损害标准准确率，压力均值未下降；Dense删除的压力指标下降较明显。这说明家族提供的价值存在侧重，不能简单说每个架构都专门负责所有模糊情况。

**初判与救援承担不同取舍。** 去掉救援，压力平均准确率降至94.25%；去掉初判，平均准确率有所上升，但人脸召回下降0.525个百分点，最差标准视图下降2.375个百分点。初判的作用是保留其正判断和提供早期退出，其误报也会被保留，不能称为所有指标都必需。

**三种子与可学习融合的不可替代性证据不足。** 单种子42的标准损失仅0.05个百分点，等权logit替代的损失仅0.02083个百分点，两者标准和压力条件区间均跨0。因此现有数据不足以证明三个种子或可学习融合系数必不可少。保留最终配置不等于应把这些不确定性写成已验证收益。

**非负约束与标准化有明确功能，但必要性尚未单独证明。** 非负约束规定证据方向；本轮允许负系数后，各拟合系数仍为正，全部源判断未变，说明约束没有在已拟合解上触界。去标准化并重新校准也没有显示一致劣势；标准化会影响L2所处尺度，但不能据此宣称没有它模型就无效。

**L2正则在标准面板有小幅支持。** 去掉后标准准确率下降约0.104个百分点，其条件区间偏负；压力平均下降0.125个百分点，但该条件区间跨0。它与限制小样本融合系数规模的设计动机一致，仍不能说明0.01已是最佳正则强度，或正则对每种退化都必需。

**共享与条件执行的作用由结构说明。** 共享模块提供公共特征；短路依据逻辑充分条件跳过后续成员。它们不是通过提高准确率来证明价值的模块。S01中这些实现规则的依据在第5、9、10节给出。

### 14.4 尚未单独验证的部分

没有新做去掉第一阶段、直接从初始化进行第二阶段的匹配总更新实验；也没有逐项重训消融每个GroupNorm、膨胀率、池化区间和增强算子。因此不能在本报告宣称上述每个设置均已被证明必要。

### 14.5 固定删除与重新校准的完整结果

下表保留全部F/R记录，分别说明固定删改与恢复校准。它们是S01的消融变体，不作为额外最终模型选型。

@@FULL_ABLATION_TABLE@@

## 15. 模型优点、局限与适用边界

### 15.1 优点

- **输入与结构互补**：原生低分辨率残差、高网格布局和监督迁移/强度路径提供不同类型的证据。
- **复合退化训练有实测支持**：在固定原图角色和内部评价下，对救援阶段训练路线的消融显示较明显压力损失。
- **共享特征与条件执行明确**：公共冻结参数只保存一份；执行路径由已经足够决定最终标签的逻辑条件确定。
- **分数融合容量有限**：每组仅拟合三个系数与截距，具有正则和边界，校准来源可追踪。
- **数据角色可核对**：原图先分角色、视图随后生成，ES、META、SEL各自用途清楚，代码和权重有哈希记录。

### 15.2 局限

- **初判误报无法由救援纠正**，高召回逻辑可能增加非人脸误报；原图流程评价仍有10个误报。
- **强模糊与复合噪声仍有漏检**，19×19的信息损失可能不可逆；某些退化人脸只剩不足以区分的结构。
- **融合权重对输入固定**，条件执行不是学习图像质量后挑选专家；某个家族持续自信犯错时，静态融合仍可能失败。
- **校准样本较少**，每组SEL只有89张独立原图，长小数不能体现阈值统计稳定性。
- **部署与流程评价样本规模不同**，又存在同一训练数据上的历史开发与反复探索，不能把内部成绩当作完整独立泛化结论。
- **模拟退化覆盖有限**，未全面检验姿态、遮挡、非对齐和所有噪声机制；没有给最终逻辑输出认证概率解释。
- **模型仍有28个评分成员**，共享参数与短路不能保证所有设备上相同的吞吐或延迟；本文没有提供受控硬件benchmark。
- **部分组件必要性证据不足**，本报告保留这些结论，不以固定模型选择替代实验论证。

## 16. 关键代码、复核材料与使用

| 材料 | 内容 |
| --- | --- |
| [S01实际运行器](code/c1_shared_runtime.py) | 共享前缀、独立尾部、float32/64数值规则和条件执行 |
| [网络与预处理](code/networks.py) | ResNet8、HiRes19及输入转换 |
| [专家训练代码](code/a1_a4_experiment.py) | 两阶段权重、复合退化与记录；文件还含未启用的其他分支 |
| [融合与角色代码](code/c_rescue_experiment.py) | META拟合、原图角色与配对；文件还含其他实验 |
| [精确阈值搜索](code/b1_b4_experiment.py) | 候选分区扫描与同分规则 |
| [最终模型身份](evidence/S01部署模型身份.json) | 实际权重位置、哈希、参数量与部署边际 |
| [逐视图结果](evidence/S01逐退化视图指标.csv) | 全部20条件的准确率、召回率与特异度 |
| [完整超参数账本](evidence/超参数完整账本.md) | 固定、拟合、内部选择、继承设置的区分 |
| [字体核验](evidence/报告图像字体核验.json) | 数据与曲线图的微软雅黑来源；结构图另附字体记录 |

以下为真实运行器提供的Python接口。模型文件、`metadata.json`、运行器和`networks.py`放在同一模型目录；图像恢复方式必须与训练一致：

```python
from c1_shared_runtime import SharedRuntime
model = SharedRuntime(model_dir,device='cpu')  # 或 device='mps'
labels, stats = model.predict(images_cpu_float32_N1x19x19)
```

报告图文包不含大型神经权重；运行器代码快照保留导出功能的工作区路径，推理从显式传入的模型目录读取便携文件。报告生成需要原工作区数据与已有单成员源评分缓存。所有训练与评价数字均可通过随附记录追溯，不用全量拟合后的训练准确率冒充折外结果。

## 17. 参考依据

本模型具体参数、成绩和训练过程以实际代码、权重、分折和日志为依据。文献支持机制与术语，不替代本任务的实验：

- He等，[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)：残差学习。
- Huang等，[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)：密集特征复用。
- Wu、He，[Group Normalization](https://arxiv.org/abs/1803.08494)：按通道分组归一化。
- Guo等，[On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)：分类分数与概率校准的区别；S01没有把最终逻辑输出称为已认证的概率。
'''
replacements={'ROLE_TABLE':role_table,'EPOCH_TABLE':epoch_table,'THRESHOLD_TABLE':threshold_table,'COMPILED_TABLE':compiled_table,'SUMMARY_TABLE':summary_table,'VIEW_TABLE':view_table,'CI_TABLE':ci_table,'ABLATION_TABLE':ab_table,'ABLATION_CI_TABLE':ab_ci_table,'FULL_ABLATION_TABLE':full_ab_table,'HYPERPARAMETERS':hyper}
for k,v in replacements.items():report=report.replace('@@'+k+'@@',v)
assert '@@' not in report
# The final manuscript contains no predecessor-model comparison or deployment-image analysis.
assert not any(t in report for t in ['简化前','简化后','测试集','平台成绩','result_C1_SCORE'])
(P/'S01最终模型报告_便携版.md').write_text(report,encoding='utf-8')
absolute=re.sub(r'\]\(([^)]+)\)',lambda mt:mt.group(0) if mt.group(1).startswith(('http://','https://','#','/')) else ']('+str(P/mt.group(1))+')',report)
(P/'S01最终模型详细报告.md').write_text(absolute,encoding='utf-8')
shutil.copyfile(Path(__file__),P/'code'/Path(__file__).name)
print('WROTE',len(report),'characters',len(report.splitlines()),'lines')
