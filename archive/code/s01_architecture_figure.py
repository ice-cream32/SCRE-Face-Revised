"""Draw the final S01 structure and conditional execution, without data reads."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.text import Text


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/S01最终模型完整报告'
IMAGE = OUT / 'assets/01_S01整体结构.png'
AUDIT = OUT / 'evidence/01_结构图字体核验.json'
FONT = Path('${ORIGINAL_HOME}/Library/Fonts/MedicalMVP-MicrosoftYaHei.ttc')
BOLD = FONT.with_name('MedicalMVP-MicrosoftYaHei-Bold.ttc')
for path in [FONT, BOLD]:
    fm.fontManager.addfont(str(path))
FAMILY = fm.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'font.family': FAMILY, 'axes.unicode_minus': False,
                     'font.size': 16, 'figure.dpi': 110, 'savefig.dpi': 180})

BLUE = '#225B91'
GREEN = '#287F6A'
ORANGE = '#B56D24'
PURPLE = '#755293'
INK = '#243649'
MUTED = '#536476'


def box(ax, x, y, w, h, text='', edge=BLUE, fill='#F2F7FB', fontsize=16,
        weight='normal', linewidth=1.7, radius=.10):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.03,rounding_size={radius}',
                          edgecolor=edge, facecolor=fill, lw=linewidth)
    ax.add_patch(patch)
    if text:
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', color=INK,
                fontsize=fontsize, fontweight=weight, linespacing=1.35)
    return patch


def arrow(ax, points, label=None, label_at=None, color=MUTED, width=1.8):
    # Orthogonal polylines keep paths away from text boxes.
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=width,
                solid_capstyle='round')
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                                 mutation_scale=15, color=color, lw=width))
    if label:
        ax.text(*label_at, label, ha='center', va='center', fontsize=15, color=color,
                bbox={'facecolor':'white', 'edgecolor':'none', 'pad':2.5})


def diamond(ax, cx, cy, w, h, text):
    ax.add_patch(Polygon([(cx-w/2,cy), (cx,cy+h/2), (cx+w/2,cy), (cx,cy-h/2)],
                         closed=True, edgecolor=BLUE, facecolor='#F2F7FB', lw=1.8))
    ax.text(cx,cy,text,ha='center',va='center',fontsize=16,color=INK,linespacing=1.25)


fig = plt.figure(figsize=(24,18.4), facecolor='white')
top = fig.add_axes([.035,.425,.93,.50])
top.set(xlim=(0,24), ylim=(0,14))
top.axis('off')
fig.text(.5,.965,'S01 最终模型：共享特征、四路判别与条件执行',
         ha='center',va='center',fontsize=27,fontweight='bold',color=INK)
fig.text(.5,.938,'28 个分类成员 ＝ 24 个小网络 ＋ 4 个 Dense 分类成员；神经参数 15,294,620',
         ha='center',va='center',fontsize=17,color=MUTED)

top.text(.1,13.6,'A  模型组成与分支评分',fontsize=21,fontweight='bold',color=INK)
box(top,.30,11.25,4.8,1.35,'输入图像 x\n1 × 19 × 19 灰度，[0,1]',fontsize=17,weight='bold')
box(top,6.20,12.0,6.0,1.15,'小网络输入 A_S(x)\n逐图标准化与裁剪',edge=GREEN,fill='#F0F8F4',fontsize=16)
box(top,6.20,10.35,6.0,1.15,'Dense 输入 A_D(x)\n128×128、三通道、ImageNet 归一化',fontsize=15.5)
arrow(top,[(5.1,12.2),(5.62,12.2),(5.62,12.58),(6.2,12.58)])
arrow(top,[(5.1,11.65),(5.62,11.65),(5.62,10.93),(6.2,10.93)])
box(top,13.10,10.35,5.00,1.15,'共享冻结前缀 E\nz = E(A_D(x))：512 × 4 × 4',fontsize=16,weight='bold')
arrow(top,[(12.2,10.93),(13.1,10.93)])
box(top,18.90,10.35,4.80,1.15,'N5：共享 norm5 参数\n四条分支各自计算激活',edge=ORANGE,fill='#FCF5EC',fontsize=15.5)
top.text(13.1,12.56,'每图只计算一次 E，按图像 ID 复用 z',fontsize=16,color=BLUE,
         ha='left',va='center',fontweight='bold')
top.text(6.20,9.82,'所有 R/H 成员接收 A_S(x)；每条 Dense 尾接收同一 z。',
         fontsize=15,color=MUTED,ha='left',va='center')

# One clean fan-out bus distributes z to the Dense row of each branch.
bus_y = 9.30
top.plot([15.6,15.6],[10.35,bus_y],color=BLUE,lw=1.9)
top.plot([5.45,23.45],[bus_y,bus_y],color=BLUE,lw=1.9)
top.text(.30,9.25,'共享特征 z',fontsize=15,color=BLUE,va='center')

for j,x in enumerate([.30,6.30,12.30,18.30]):
    color = BLUE if j==0 else PURPLE
    title = '保护分支 B' if j==0 else f'救援分支 {j}'
    # Full branch border; parallel member families are listed, not chained.
    box(top,x,.65,5.30,8.05,edge=color,fill='#FFFFFF',linewidth=2.1)
    top.text(x+2.55,8.32,title+' · 7 成员',ha='center',va='center',
             fontsize=18,fontweight='bold',color=color)
    box(top,x+.20,4.45,4.45,3.35,edge='#CBD5DF',fill='#FBFCFE',linewidth=1.2)
    top.plot([x+.22,x+4.63],[6.67,6.67],color='#D9E2EA',lw=1)
    top.plot([x+.22,x+4.63],[5.61,5.61],color='#D9E2EA',lw=1)
    sub = 'B' if j==0 else str(j)
    top.text(x+2.43,7.25,'ResNet8 × 3（42 / 7 / 2026）\n成员 σ 后概率平均 → p_'+sub+'R',
             ha='center',va='center',fontsize=14.1,linespacing=1.35,color=GREEN)
    top.text(x+2.43,6.15,'HiRes19 × 3（42 / 7 / 2026）\n成员 σ 后概率平均 → p_'+sub+'H',
             ha='center',va='center',fontsize=14.1,linespacing=1.35,color=GREEN)
    top.text(x+2.43,5.02,'Dense 尾：block4_'+sub+' → N5 → ReLU\n→ GAP → head_'+sub+' → σ → p_'+sub+'D',
             ha='center',va='center',fontsize=13.7,linespacing=1.35,color=ORANGE)
    # Shared z enters only the Dense row, through the free right-side gutter.
    arrow(top,[(x+5.15,bus_y),(x+5.15,5.02),(x+4.67,5.02)],color=BLUE,width=1.7)
    arrow(top,[(x+2.43,4.44),(x+2.43,4.02)],color=color)
    if j==0:
        score='三家族等权概率均值\np_B = (p_BR + p_BH + p_BD) / 3\nv_B = 1[p_B ≥ 0.37]'
    else:
        score='家族分数 → 受限 logit L_'+str(j)+'\nh_'+str(j)+' = w_'+str(j)+' · L_'+str(j)+' + γ_'+str(j)+'\nv_'+str(j)+' = 1[h_'+str(j)+' ≥ 0]'
    box(top,x+.20,1.65,4.45,2.35,score,edge=color,
        fill='#F2F7FB' if j==0 else '#F7F3FA',fontsize=14.7)
    top.text(x+2.55,1.12,'三个家族先各自产生一个分数',ha='center',va='center',
             fontsize=13.5,color=MUTED)

top.text(12,.10,'四个 block4 / head 分别训练；N5 参数共享不等于分支激活共享。救援 w 与 γ 为部署固定参数。',
         ha='center',va='center',fontsize=15,color=MUTED)

bottom = fig.add_axes([.035,.055,.93,.30])
bottom.set(xlim=(0,24),ylim=(0,8.3))
bottom.axis('off')
bottom.text(.1,8.0,'B  逐图条件执行：只计算影响当前决策的分支',
            fontsize=21,fontweight='bold',color=INK)
box(bottom,.30,4.40,3.70,1.35,'执行保护分支\n计算并缓存 z，得到 p_B',fontsize=15.5)
diamond(bottom,6.2,5.075,3.1,2.0,'p_B ≥ 0.37？')
arrow(bottom,[(4.0,5.075),(4.65,5.075)])
box(bottom,4.65,6.72,3.1,.88,'返回：人脸（+1）',edge=GREEN,fill='#F0F8F4',fontsize=15.5)
arrow(bottom,[(6.2,6.075),(6.2,6.72)],label='是',label_at=(6.48,6.40),color=GREEN)
box(bottom,8.55,4.40,4.20,1.35,'只对保护分支负例\n计算救援投票 v_1、v_2',edge=PURPLE,fill='#F7F3FA',fontsize=15.5)
arrow(bottom,[(7.75,5.075),(8.55,5.075)],label='否',label_at=(8.10,5.40))
diamond(bottom,15.2,5.075,3.10,2.0,'v_1 = v_2？')
arrow(bottom,[(12.75,5.075),(13.65,5.075)])
box(bottom,13.25,6.72,3.9,.88,'返回两者共同的投票',edge=GREEN,fill='#F0F8F4',fontsize=15.5)
arrow(bottom,[(15.2,6.075),(15.2,6.72)],label='是',label_at=(15.48,6.40),color=GREEN)
box(bottom,13.10,1.90,4.2,1.25,'两票分歧：计算 v_3\n复用 z，只执行第 3 救援分支',edge=PURPLE,fill='#F7F3FA',fontsize=15)
arrow(bottom,[(15.2,4.075),(15.2,3.15)],label='否',label_at=(15.50,3.62))
box(bottom,18.55,1.90,4.60,1.25,'返回 v_3 对应类别\n等价于三组投票至少 2 票为正',edge=GREEN,fill='#F0F8F4',fontsize=15)
arrow(bottom,[(17.30,2.525),(18.55,2.525)],color=GREEN)
bottom.text(.30,2.85,'最终规则：',ha='left',va='center',fontsize=17,fontweight='bold',color=INK)
bottom.text(.30,2.08,'v_B = 1 或 v_1 + v_2 + v_3 ≥ 2 → 人脸',ha='left',va='center',fontsize=17,color=INK)
bottom.text(.30,1.34,'其他情况 → 非人脸',ha='left',va='center',fontsize=17,color=INK)
bottom.text(12,.25,'投票 1 对应人脸（+1）；投票 0 对应非人脸（−1）。σ 仅用于成员概率，救援融合直接比较线性边际。',
            ha='center',va='center',fontsize=15,color=MUTED)

IMAGE.parent.mkdir(parents=True,exist_ok=True)
AUDIT.parent.mkdir(parents=True,exist_ok=True)
texts_for_coverage=[t for t in fig.findobj(match=Text) if t.get_text()]
missing_glyphs=[]
for text in texts_for_coverage:
    font=BOLD if text.get_fontweight() in ('bold',700) else FONT
    supported=fm.get_font(str(font)).get_charmap()
    missing_glyphs.extend(sorted({ch for ch in text.get_text() if not ch.isspace() and ord(ch) not in supported}))
assert not missing_glyphs, missing_glyphs
fig.canvas.draw()
texts=[]
for item in fig.findobj(match=Text):
    if not item.get_text():
        continue
    font = BOLD if item.get_fontweight() in ('bold',700) else FONT
    item.set_fontproperties(fm.FontProperties(fname=str(font),size=item.get_fontsize()))
    texts.append({'text':item.get_text(),'font_file':str(font),'point_size':item.get_fontsize()})
fig.savefig(IMAGE,dpi=180,facecolor='white')
plt.close(fig)
audit={'figure':str(IMAGE),'font_family':FAMILY,'font_regular':str(FONT),'font_bold':str(BOLD),
       'font_regular_sha256':hashlib.sha256(FONT.read_bytes()).hexdigest(),
       'font_bold_sha256':hashlib.sha256(BOLD.read_bytes()).hexdigest(),
       'text_objects':texts,'minimum_point_size':min(item['point_size'] for item in texts),
       'all_text_objects_use_Microsoft_YaHei':all('MicrosoftYaHei' in item['font_file'] for item in texts),
       'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'image_sha256':hashlib.sha256(IMAGE.read_bytes()).hexdigest(),
       'data_or_prediction_files_read':False,
       'missing_glyphs':missing_glyphs,
       'architecture':'one shared Dense prefix/norm5;4 independent Dense tails;24 small networks;4 seven-member branches',
       'visual_qa':'Pending direct generated-image inspection.'}
AUDIT.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'image':str(IMAGE),'audit':str(AUDIT),'font':FAMILY,
                  'minimum_point_size':audit['minimum_point_size']},ensure_ascii=False))
