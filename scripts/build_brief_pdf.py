"""Generate the 12-page main report and 5-page evidence appendix.
Supply Microsoft YaHei fonts with --font and --font-bold; fonts are not bundled.
"""
from pathlib import Path
import argparse,json,csv,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from PIL import Image
from report_graphics import draw_math, math_drawing, architecture, conditional
P=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--font',required=True);a.add_argument('--font-bold',required=True);args=a.parse_args()
pdfmetrics.registerFont(TTFont('YaHei',args.font,subfontIndex=0));pdfmetrics.registerFont(TTFont('YaHeiBold',args.font_bold,subfontIndex=0))
pdfmetrics.registerFontFamily('YaHei',normal='YaHei',bold='YaHeiBold',italic='YaHei',boldItalic='YaHeiBold')
TMP=P/'outputs/pdf_assets';TMP.mkdir(parents=True,exist_ok=True)
FONT=FontProperties(fname=args.font)
SUMMARY=json.loads((P/'report_summary.json').read_text());EV=P/'evidence';AS=P/'docs/assets'
PDF=P/'docs/SCRE-Face_experiment_brief.pdf';W,H=landscape(A4);C=canvas.Canvas(str(PDF),pagesize=(W,H));C.setTitle('SCRE-Face 实验简报');C.setAuthor('')
PAGE=0;mathid=0
def text(s,x,y,w,size=11,leading=None,color='#182B3D',bold=False):
 style=ParagraphStyle('p',fontName='YaHeiBold' if bold else 'YaHei',fontSize=size,leading=leading or size*1.55,textColor=colors.HexColor(color),wordWrap='CJK')
 q=Paragraph(s,style);qw,qh=q.wrap(w,1000);assert y-qh>=40,(PAGE,s[:45],y,qh)
 q.drawOn(C,x,y-qh);return y-qh-8
def page(title,appendix=False):
 global PAGE;PAGE+=1
 C.setFillColor(colors.HexColor('#162B3F'));C.setFont('YaHeiBold',19);C.drawString(42,H-41,title)
 C.setFont('YaHei',8);C.setFillColor(colors.HexColor('#536375'))
 C.drawString(42,22,'SCRE-Face  |  S01固定实现  |  '+('证据附录' if appendix else '实验正文'))
 C.drawRightString(W-42,22,f'{PAGE} / 17')
 return H-75
def end():C.showPage()
def image(path,x,y,mw,mh):
 im=Image.open(path);scale=min(mw/im.width,mh/im.height);w=im.width*scale;h=im.height*scale
 C.drawImage(str(path),x+(mw-w)/2,y-h,width=w,height=h,mask='auto');return y-h-6
def eq(tex,x,y,w=750):
 # Natural-size STIX outlines stay sharp at any zoom level; no bitmap upscaling.
 height=55
 C.setFillColor(colors.HexColor('#F5F7FA'));C.roundRect(x,y-height,w,height,5,fill=1,stroke=0)
 d=math_drawing(tex,17);actual_h=min(d.height,43)
 draw_math(C,tex,x+w/2,y-(height-actual_h)/2,w-32,17,43)
 return y-height-10
def table(headers,rows,x,y,widths,size=9.2):
 sty=ParagraphStyle('cell',fontName='YaHei',fontSize=size,leading=size*1.35,wordWrap='CJK')
 def cell(v):
  s=str(v)
  return math_drawing(s[1:-1],size+1) if s.startswith('$') and s.endswith('$') else Paragraph(html.escape(s),sty)
 data=[[cell(v) for v in row] for row in [headers]+rows]
 t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.45,colors.HexColor('#CBD4DC')),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E4EDF5')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6)]))
 tw,th=t.wrap(750,1000);assert y-th>=42,('table overflow',PAGE,th,y);t.drawOn(C,x,y-th);return y-th-12

y=page('SCRE-Face 低分辨率模糊人脸分类实验')
y=text('共享特征条件救援集成：以原图和退化样本进行两阶段训练，以分离的校准角色拟合分数融合。输入19×19灰度图，输出人脸或非人脸；不进行身份识别。',42,y,750,13)
y=table(['结果层次','指标','解释'],[['平台返回','Acc = 0.964','用户报告；420张无标签测试图像，未提供逐图真值'],['训练原图流程折外','790/800 = 98.75%','TP400、TN390、FN0、FP10'],['标准16视图','加权准确率98.0833%','原图权重0.5，其他15视图各1/30'],['四压力视图','平均准确率97.6875%','当轮不参与检查点、META或SEL选择']],[42][0],y,[150,205,395],11)
y=text('核心结构：28个评分成员，物理存储15,294,620个神经参数；共享DenseNet冻结特征，按决策条件跳过无须执行的救援分支。',42,y,750,12,bold=True)
y=text('阅读范围：正文共12页（包含本页），附录5页。全部细节、日志与代码索引在GitHub包中。此次发布只整理、复算和核验，没有重新训练、改阈值或人工修正测试预测。平台与内部成绩分开解释。',42,y,750,11)
y=text('结论边界：强模糊、复合噪声和初判误报仍是短板。消融显示各部件存在不同取舍，不能宣称全部模块不可替代。项目曾多轮参考平台和困难样本反馈，0.964不构成一次性未触碰盲测声明。',42,y,750,11)
end()

y=page('01 数据与任务定义')
y=text('训练集800张，两类各400；测试集420张且无标签。原文件每行361个像素，float64、范围[0,1]。恢复为N×1×19×19 float32时，reshape后交换两个空间轴；不再除255。未发现像素完全相同的训练原图，但缺少身份或拍摄来源元数据。',42,y,750,11)
y=image(AS/'03_数据统计.png',42,y,750,320)
text('图1 训练集类别均值、标准差与强度纹理统计。人脸具有较一致的粗布局；这些是总体描述，不能作为每张图都对齐或清晰的证据。数据原压缩包、MAT及哈希随仓库提供。',42,y,750,10)
end()

y=page('02 最终结构与参数共享')
y=architecture(C,42,y)
text('图2 四路分类共享冻结前缀特征。R/H家族先平均三个种子概率；四个Dense尾部拥有独立block4和分类头，共享norm5参数不等于共享分支激活。',42,y,750,9)
end()

y=page('03 三家族的输入与互补动机')
y=table(['家族','数量／分支','输入与设计','优势与局限'],[['ResNet8','3','逐图标准化；残差与两次下采样','小容量、低分辨率原生输入；弱结构可能随下采样丢失'],['HiRes19','3','保持19×19；膨胀率1/2/3；全局+3×3空间池化','保留粗布局；自定义小CNN，不是原HRNet或频谱分解'],['DenseNet121','1','原强度放大至128；复制RGB并做ImageNet归一化','监督预训练、特征复用；放大不增加细节，也可能过拟合']],[42][0],y,[105,90,270,285],10)
y=eq(r'\widetilde{x}=\operatorname{clip}\!\left(\frac{x-\mu_x}{\max(\sigma_x,\,0.01)},\,-3,\,3\right)',42,y)
y=text('小网络标准差按逐图总体定义（correction=0）。Dense路径使用原始[0,1]强度、双线性放大、均值(0.485,0.456,0.406)与标准差(0.229,0.224,0.225)。不对Dense输入先做小网标准化。',42,y,750)
y=text('每条分支含七个成员；四条分支合计24个小网和4个Dense分类成员。共享前缀及norm5共4,795,776个参数，四份独立Dense block4及头、小网合计10,498,844个参数。部署时全部冻结。',42,y,750)
text('模型动机必须由匹配消融支持：架构名字不同不保证错误互补；残差、密集连接和抗混叠也不自动等于去模糊。本最终模型没有逐图可学习注意力路由。',42,y,750)
end()

y=page('04 两阶段训练与退化监督')
y=text('<b>阶段一</b> 使用原图内容与常规几何、翻转、强度增强，建立基本人脸结构判别。<b>阶段二</b> 从对应阶段一检查点继续，以一个常规视图和两个独立退化视图共同监督；两类采用同样的退化生成政策。',42,y,750,12)
y=eq(r'\ell(z,y)=\log(1+e^z)-yz,\qquad y\in\{0,1\}',42,y)
y=eq(r'\mathcal{L}_2=\frac{1}{3}\left[\ell\!\left(f(c),y\right)+\ell\!\left(f(d_1),y\right)+\ell\!\left(f(d_2),y\right)\right]',42,y)
y=table(['设置','ResNet8／HiRes19','DenseNet121'],[['优化器','AdamW，weight decay 0.001','同左；只更新block4与分类头，BN运行统计固定'],['阶段一／二上限','80／40轮','30／20轮'],['阶段一／二初始LR','0.001／0.0001','0.0001／0.00003'],['原图batch上限','64','32'],['学习率调度','按阶段上限余弦下降','同左']],[42][0],y,[175,250,325],10)
text('退化包含高斯与运动模糊、降采样、对比度变化、噪声、JPEG及滤波/噪声顺序变化。阶段二改变训练分布以鼓励结构利用，不能恢复已丢失细节。LR、增强范围和训练上限多为固定设置；最佳检查点依据ES验证BCE选择，未把每个超参数都系统搜索。',42,y,750)
end()

y=page('05 原图角色隔离与检查点选择')
y=table(['角色','用途','与其他角色的关系'],[['FIT','神经网络参数拟合','先分原图，再生成变体'],['ES','阶段一原图BCE；阶段二固定16视图加权BCE选检查点','同一配对内与FIT、META、SEL、OUT互斥'],['META','分数标准化与融合系数拟合','不使用SEL选择阈值'],['SEL','冻结评分后选择救援分类阈值','不更新神经网络权重'],['OUT','流程外层评价','当轮不参与检查点与校准选择']],[42][0],y,[90,320,340],10.5)
y=text('三折外层评价，每个外层训练池再构建三个校准配对。典型嵌套规模：FIT284、ES71/72、META118、SEL59/60、OUT266/267。最终部署配对：FIT426/426/427，ES107，META178/178/177，SEL89。各配对原图ID完整记录随包提供。',42,y,750)
y=text('初判分支最终使用800张原图固定轮数拟合；轮数继承内部选择后按家族中位数确定，不伪造全800拟合过程的逐轮验证曲线。救援保留各自ES所选检查点。不同交叉验证配对可复用原图承担不同角色。',42,y,750)
text('解释验证损失：少量高置信错误可令BCE上升，即使准确率不变。检查点选择依据验证目标而非训练loss一直下降；更长执行轮数不等于应采用最后一轮。',42,y,750)
end()

y=page('06 非负logit融合与救援边界')
y=eq(r'\bar p_{jk}=\operatorname{clip}(p_{jk},\,10^{-5},\,1-10^{-5}),\qquad L_{jk}=\operatorname{clip}\!\left(\log\frac{\bar p_{jk}}{1-\bar p_{jk}},\,-8,\,8\right)',42,y)
y=text('META按原图和视图权重计算均值与尺度（尺度下限0.05），对三个家族logit进行标准化。类别加权BCE加0.01倍系数L2惩罚；系数限定[0,4]、截距[-4,4]，L-BFGS-B解析梯度求解。系数和不要求为1，部署时对所有输入固定。',42,y,750)
y=eq(r'h_j(x)=\sum_{k\in\{R,H,D\}}w_{jk}L_{jk}(x)+\gamma_j,\qquad v_j=\mathbf{1}\!\left\{h_j(x)\geq0\right\}',42,y)
rows=[]
for i,g in enumerate(SUMMARY['compiled_gates'],1):rows.append([i,*[f'{v:.9f}' for v in g['weights']],f"{g['decision_bias']:.9f}"])
y=table(['救援组','$w_R$','$w_H$','$w_D$',r'$\gamma$'],rows,42,y,[75,170,170,170,165],10)
y=text('SEL选择原评分阈值后，将标准化、融合系数、截距与logit阈值吸收到线性边际。零是代数变换后的边界，不是又搜索的新概率阈值。实际模型保留float32评分与float64融合顺序，并在临界边际回退原比较。表中只显示九位小数，推理使用保存的浮点值。',42,y,750)
text('救援原评分阈值为约0.496424、0.421976、0.283380；仅在各组SEL上搜索，使用标准16视图加权平衡准确率。三个阈值不能平均为统一全局阈值。',42,y,750)
end()

y=page('07 初判0.37如何被选中')
y=text('历史七网络配方：训练原图分层640/160（种子20261112），验证两类各80。八视图为原图、高斯σ1/1.5/2、降采样12/7、水平运动5、对比度0.3。固定评分后遍历0.01至0.99共99阈值。',42,y,750)
y=eq(r'J(t)=\frac{1}{2}A_0(t)+\frac{1}{14}\sum_{v=1}^{7}A_v(t),\qquad t^{\star}=\underset{t\in\{0.01,\,0.02,\,\ldots,\,0.99\}}{\operatorname{arg\,max}}\ J(t)',42,y)
y=table(['阈值','原图错误／160','退化错误／1120','加权准确率'],[['0.30',1,11,'99.1964%'],['0.35／0.36',0,7,'99.6875%'],['0.37',0,6,'99.7321%'],['0.38',0,7,'99.6875%'],['0.50',0,13,'99.4196%']],[42][0],y,[130,195,220,205],10.5)
y=text('0.37为该配方99候选中唯一最高点；同分规则（先近0.5、再取小阈值）未触发。训练#778非人脸的σ2评分约0.365149：0.37修正0.36时误报；#362人脸评分约0.375999：0.38新增漏检。',42,y,750)
text('160张验证原图也用于检查点与配方选择，所以99.7321%属于开发成绩。0.37对邻近点的优势只有一个退化视图判断。当前初判重训后继承该值，未对当前完整级联重新联合寻优。完整复算见docs/threshold_search.md与scripts/replay_threshold.py。',42,y,750)
end()

y=page('08 条件执行与最终分类')
y=conditional(C,42,y)
y=eq(r'\widehat y=+1\quad\Longleftrightarrow\quad (p_B\geq0.37)\ \vee\ (v_1+v_2+v_3\geq2)',42,y)
text('图3 初判正例直接保留；负例先算救援1/2，同票直接结束、异票才算第3组。共享前缀每图计算一次。该规则不能纠正初判误报，也不是按图像质量学习动态权重。',42,y,750,10)
end()

y=page('09 分类结果与平台成绩')
image(AS/'10_S01折外混淆矩阵.png',455,y,330,325)
yy=table(['训练集流程指标','值'],[['原图准确率','790/800 = 98.75%'],['人脸召回率','400/400 = 100%'],['非人脸特异度','390/400 = 97.5%'],['误报／漏检','10／0']],[42][0],y,[170,225],11)
yy=text('图4为训练原图流程折外结果。全部10个错误均为非人脸误报；不将训练集全量拟合准确率冒充该结果。',42,yy,395,11)
yy=text('测试集无标签：平台Acc 0.964由用户报告。本包保存模型生成的420行预测与哈希，不能据此在本地计算测试混淆矩阵或逐图正确率。',42,yy,395,11)
text('发布时实际在CPU重新加载共享权重并对420张图像推理，与归档预测分类差异为0；该核验只证明推理一致性，不是重新获得平台准确率。测试真值未包含于仓库，预测文件不能被当成标签。',42,156,750,11)
end()

y=page('10 模糊与噪声鲁棒性')
image(AS/'14_S01模糊强度响应.png',42,y,470,325)
yy=text('高斯σ从0至2：准确率98.75%、98.75%、97.25%、94.50%。σ2时人脸召回94%，强模糊仍有漏检。',535,y,260,11)
yy=text('四压力均值97.6875%；其中模糊+散斑的人脸召回只有90%，是明确短板。对角运动5的召回91.75%，说明不同退化机制不能仅用一个模糊强度排序。',535,yy,260,11)
yy=text('四压力视图不参加当轮检查点、META和SEL；但与增强有物理相似性，不能称所有机制完全未见。',535,yy,260,11)
text('条件95%区间（原图聚类、分层5000次重采样）：标准加权约[97.4542%,98.6583%]，压力均值约[97.0000%,98.3125%]。区间条件于既有权重、阈值与流程，不覆盖重新训练、长期探索或未知域偏移。全20条件见附录A。',42,143,750,11)
end()

y=page('11 消融结果与使用边界')
ab=list(csv.DictReader((EV/'源图全部消融结果.csv').open(encoding='utf-8-sig')));by={r['name']:r for r in ab}
keys=['F01_NO_C0','F02_NO_RESCUE','R11_NO_RESNET8','R12_NO_HIRES19','R13_NO_DENSENET121','R22_STAGE1','R23_OLD_DEGRADE','R26_EQUAL_LOGIT']
rows=[['完整S01','最终模型','98.083%','97.688%']]+[[k,by[k]['title'],f"{100*float(by[k]['accuracy']):.3f}%",f"{100*float(by[k]['held_accuracy']):.3f}%"] for k in keys]
y=table(['组','处理','标准加权Acc','压力平均Acc'],rows,42,y,[155,355,120,120],9.3)
y=text('F为固定其他系数/阈值删除；R为删除后仅在对应META/SEL重新校准。救援第二阶段与复合退化、ResNet8家族有较明显支持；Dense删除对压力表现的影响更大。',42,y,750,10.5)
y=text('负面结果必须保留：去初判可提高平均准确率，但改变召回与最差视图；等权logit替代的平均损失很小；多种子、标准化或非负约束未被证明普遍不可替代。消融没有为每个改动从零重训全部骨干。完整36项记录见附录B，细节和区间见evidence/。',42,y,750,10.5)
end()

y=page('附录A 全20条件的内部流程指标',True)
rows=[[v['name'],'标准' if v['kind']=='standard' else '压力',f"{100*v['accuracy']:.4f}%",f"{100*v['face_recall']:.4f}%",f"{100*v['nonface_specificity']:.4f}%"] for v in SUMMARY['views']]
table(['条件','类型','准确率','人脸召回','非人脸特异度'],rows,42,y,[280,65,135,135,135],8.9)
end()
for part,items in enumerate([ab[:18],ab[18:]],1):
 y=page(f'附录B 完整固定删除与重校准记录 {part}/2',True)
 rows=[[r['name'],r['title'],f"{100*float(r['accuracy']):.3f}",f"{100*float(r['recall']):.3f}",f"{100*float(r['specificity']):.3f}",f"{100*float(r['held_accuracy']):.3f}"] for r in items]
 y=table(['组','处理','标准Acc%','召回%','特异度%','压力Acc%'],rows,42,y,[155,300,74,73,74,74],8.3)
 text('以上均为训练集内部流程评价；F/R不能混同。原图聚类条件区间与全部逐图决策可在仓库证据文件中复查。',42,y,750,9)
 end()

y=page('附录C 部署轮次与原图角色',True)
epochs=json.loads((EV/'28网络部署权重与轮次.json').read_text());groups={}
for r in epochs:
 b=r['branch'];groups.setdefault((r['kind'],r['seed']),{})[b]=f"{r['stage1_epoch']} → {r['stage2_epoch']}"
branches=list(dict.fromkeys(r['branch'] for r in epochs))
rows=[[k,s,*[v[b] for b in branches]] for (k,s),v in groups.items()]
y=table(['家族','种子','初判阶段1→2','救援1 阶段1→2','救援2 阶段1→2','救援3 阶段1→2'],rows,42,y,[125,60,140,140,140,145],10)
y=text('初判是全800固定长度训练；救援所列为ES选择的检查点轮次，不等于只执行这些轮数。小网救援实际执行80/40轮，Dense执行30/20轮。原图角色与种子、优化器及超参数来源的完整账本见evidence/超参数完整账本.md。',42,y,750,10.5)
y=table(['部署配对','FIT','ES','META','SEL'],[['1',426,107,178,89],['2',426,107,178,89],['3',427,107,177,89]],42,y,[190,140,140,140,140],10)
text('同一配对先划分原图再生成退化；角色两两互斥。初判阈值0.37为历史内部开发设置继承，三组救援阈值分别依据SEL确定。没有声称所有架构宽度、增强强度、正则系数与投票数均经过最优搜索。',42,y,750,10)
end()

y=page('附录D 证据索引与复现说明',True)
y=table(['证据','位置／用途'],[['冻结模型及数值规则','models/s01/；src/scre_face/runtime.py'],['训练原图与无标签测试图','data/train.zip、test.zip及解压MAT'],['分折、训练日志与超参数','evidence/原图划分完整记录.json、配对曲线真实记录.csv、超参数完整账本.md'],['内部逐图决策与消融','evidence/S01源图折外决策.npz、源图全部消融结果.csv'],['阈值历史缓存与复算','evidence/historical_threshold/；scripts/replay_threshold.py'],['平台分数与预测来源','results/platform_result.json；test_predictions.txt'],['完整方法和历史代码','docs/full_report.md；archive/code/（有上游工作区依赖）']],[42][0],y,[190,560],9.6)
y=text('复现级别：冻结模型推理、预测一致性、保存决策的指标复算和历史阈值复算可独立运行。完整从零训练和全部消融重训仍需要恢复历史上游依赖；本包没有将快照误称为一键完整训练项目。发布材料不包含人工测试标签。',42,y,750,10.5)
y=text('背景参考：He等，Deep Residual Learning for Image Recognition，arXiv:1512.03385；Huang等，Densely Connected Convolutional Networks，arXiv:1608.06993；Wu与He，Group Normalization，arXiv:1803.08494；Guo等，On Calibration of Modern Neural Networks，arXiv:1706.04599。架构与数据结论以本项目实际代码及记录为依据，文献不替代任务实验。',42,y,750,9.5)
text('发布记录：测试平台Acc 0.964来自用户报告；无本地真值，未独立重新评分。长期使用平台反馈可能影响项目级选择，内部区间也未覆盖所有探索。报告正文12页、附录5页，总17页；数据与字体许可未在本材料中重新授予。',42,y,750,9.5)
end()
assert PAGE==17,PAGE
C.save();print(PDF)
