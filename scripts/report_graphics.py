"""Vector mathematics and diagrams; Chinese text uses registered YaHei fonts."""
from io import BytesIO
from functools import lru_cache
import math
import re
from matplotlib.mathtext import math_to_image
from matplotlib.font_manager import FontProperties
from reportlab.lib import colors
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg

INK = '#21364B'
BLUE = '#285F91'
GREEN = '#2B806D'
PURPLE = '#785595'
GRAY = '#627489'

@lru_cache(maxsize=256)
def math_drawing(tex, size=16):
    stream = BytesIO()
    math_to_image('$' + tex + '$', stream,
                  prop=FontProperties(size=size, math_fontfamily='stix'),
                  format='svg', color=INK)
    # math_to_image emits an opaque page background; keep only glyph outlines.
    svg = stream.getvalue().decode('utf-8')
    svg = re.sub(r'<g id="patch_1">.*?</g>', '', svg, flags=re.S)
    return svg2rlg(BytesIO(svg.encode('utf-8')))

def draw_math(c, tex, center_x, top, width, size=16, max_height=80):
    """Place outlines at their natural point size; shrink only if required."""
    d = math_drawing(tex, size)
    scale = min(1, width / d.width, max_height / d.height)
    c.saveState()
    c.translate(center_x - d.width * scale / 2, top - d.height * scale)
    c.scale(scale, scale)
    renderPDF.draw(d, c, 0, 0)
    c.restoreState()
    return d.height * scale

class Diagram:
    def __init__(self, c):
        self.c = c

    def label(self, value, x, y, size=9.5, bold=False, color=INK):
        self.c.setFillColor(colors.HexColor(color))
        self.c.setFont('YaHeiBold' if bold else 'YaHei', size)
        self.c.drawCentredString(x, -y, value)

    def formula(self, value, x, top, width=150, size=11, height=30):
        return draw_math(self.c, value, x, -top, width, size, height)

    def box(self, x, y, w, h, edge=BLUE, fill='#F4F8FC'):
        self.c.setStrokeColor(colors.HexColor(edge))
        self.c.setFillColor(colors.HexColor(fill))
        self.c.setLineWidth(.85)
        self.c.roundRect(x, -y-h, w, h, 5, fill=1, stroke=1)

    def arrow(self, points, color=GRAY):
        c = self.c
        c.setStrokeColor(colors.HexColor(color))
        c.setFillColor(colors.HexColor(color))
        c.setLineWidth(.9)
        p = c.beginPath()
        p.moveTo(points[0][0], -points[0][1])
        for x,y in points[1:]: p.lineTo(x,-y)
        c.drawPath(p)
        (x0,y0),(x,y) = points[-2:]
        theta = math.atan2(y-y0,x-x0)
        q=c.beginPath();q.moveTo(x,-y)
        for angle in [theta+2.65,theta-2.65]:
            q.lineTo(x+5*math.cos(angle),-y-5*math.sin(angle))
        q.close();c.drawPath(q,fill=1,stroke=0)

    def diamond(self, x,y,w,h,tex):
        c=self.c;p=c.beginPath();p.moveTo(x-w/2,-y)
        p.lineTo(x,-y+h/2);p.lineTo(x+w/2,-y);p.lineTo(x,-y-h/2);p.close()
        c.setStrokeColor(colors.HexColor(BLUE));c.setFillColor(colors.HexColor('#F4F8FC'))
        c.drawPath(p,fill=1,stroke=1)
        self.formula(tex,x,y-7,w-16,12,20)

def architecture(c, x, top):
    c.saveState();c.translate(x,top);d=Diagram(c)
    d.box(0,36,128,72)
    d.label('输入图像',64,55,10,True)
    d.formula(r'x\in[0,1]^{1\times19\times19}',64,65,118,11)
    d.label('灰度 · 原始强度',64,97,9)
    d.box(150,26,168,44,GREEN,'#F1F8F5')
    d.formula(r'A_S(x)',234,32,150,12)
    d.label('逐图标准化与裁剪',234,62)
    d.box(150,84,168,44)
    d.formula(r'A_D(x)',234,89,150,12)
    d.label('128×128 · 三通道归一化',234,120,9)
    d.arrow([(128,55),(140,55),(140,48),(150,48)])
    d.arrow([(128,91),(140,91),(140,106),(150,106)])
    d.box(340,84,166,44)
    d.label('共享冻结前缀',423,99,10,True)
    d.formula(r'z=E(A_D(x))\in\mathbb{R}^{512\times4\times4}',423,107,156,10)
    d.arrow([(318,106),(340,106)])
    d.box(528,84,222,44,'#B77A39','#FCF7EF')
    d.label('共享 norm5 参数',639,101,10,True)
    d.label('各分支分别计算激活',639,119,9.5)
    d.label('每张图只计算一次共享前缀',544,44,10,True,BLUE)
    d.label('四个 block4 与分类头各有独立权重',544,63,9.5)
    d.label('共享特征',44,153,9.5,True,BLUE)
    c.setStrokeColor(colors.HexColor(BLUE));c.setLineWidth(.9)
    c.line(423,-128,423,-164);c.line(168,-164,744,-164)
    for j,bx in enumerate([0,192,384,576]):
        key='B' if j==0 else str(j)
        edge=BLUE if j==0 else PURPLE
        d.box(bx,178,174,229,edge,'#FFFFFF')
        d.label('保护分支 B' if j==0 else f'救援分支 {j}',bx+87,197,11,True,edge)
        for yy,family,fam in [(210,'ResNet8 ×3','R'),(249,'HiRes19 ×3','H')]:
            d.box(bx+9,yy,144,33,GREEN,'#F1F8F5')
            d.label(family,bx+81,yy+12,9.5)
            d.formula(r'p_{'+key+fam+r'}',bx+81,yy+17,133,11,14)
        d.box(bx+9,288,144,45,'#B77A39','#FCF7EF')
        d.label('Dense 尾：block4 → norm5',bx+81,301,8.3)
        d.label('ReLU → GAP → 分类头',bx+81,314,8.8)
        d.formula(r'\sigma\ \longrightarrow\ p_{'+key+r'D}',bx+81,318,133,10,12)
        d.arrow([(bx+168,164),(bx+168,310),(bx+153,310)],BLUE)
        d.arrow([(bx+81,333),(bx+81,342)],edge)
        d.box(bx+9,343,144,55,edge,'#F4F8FC' if j==0 else '#F8F5FB')
        if j==0:
            d.formula(r'p_B=\frac{p_{BR}+p_{BH}+p_{BD}}{3}',bx+81,349,134,11,23)
            d.formula(r'v_B=\mathbf{1}\{p_B\geq0.37\}',bx+81,378,134,10.5,15)
        else:
            d.formula(r'h_'+key+r'=\mathbf{w}_'+key+r'^{\mathsf{T}}\mathbf{L}_'+key+r'+\gamma_'+key,bx+81,353,134,11,18)
            d.formula(r'v_'+key+r'=\mathbf{1}\{h_'+key+r'\geq0\}',bx+81,379,134,10.5,15)
    d.label('R/H 成员先经 sigmoid，再按种子平均；所有 R/H 成员接收标准化输入。',375,424,9,color=GRAY)
    c.restoreState()
    return top-438

def conditional(c,x,top):
    c.saveState();c.translate(x,top);d=Diagram(c)
    d.box(0,125,126,70)
    d.label('执行保护分支',63,143,10,True)
    d.label('计算并缓存共享特征',63,160,9)
    d.formula(r'z,\ p_B',63,169,112,12)
    d.diamond(208,160,100,70,r'p_B\geq0.37')
    d.arrow([(126,160),(158,160)])
    d.box(142,40,132,46,GREEN,'#F1F8F5')
    d.label('返回人脸',208,58,10,True,GREEN)
    d.formula(r'\widehat y=+1',208,66,120,12,16)
    d.arrow([(208,125),(208,86)],GREEN);d.label('是',222,107,9,color=GREEN)
    d.arrow([(258,160),(290,160)]);d.label('否',274,150,9,color=GRAY)
    d.box(290,125,164,70,PURPLE,'#F8F5FB')
    d.label('仅对保护分支负例',372,143,10,True,PURPLE)
    d.label('复用特征，计算前两组投票',372,161,9)
    d.formula(r'v_1,\ v_2',372,170,150,12,18)
    d.arrow([(454,160),(478,160)])
    d.diamond(528,160,100,70,r'v_1=v_2')
    d.box(462,40,132,46,GREEN,'#F1F8F5')
    d.label('返回共同投票',528,59,10,True,GREEN)
    d.formula(r'\widehat y=2v_1-1',528,67,120,11,16)
    d.arrow([(528,125),(528,86)],GREEN);d.label('是',543,107,9,color=GREEN)
    d.arrow([(528,195),(528,246)]);d.label('否',543,224,9,color=GRAY)
    d.box(459,246,138,54,PURPLE,'#F8F5FB')
    d.label('分歧时运行第 3 组',528,263,9.5,True,PURPLE)
    d.formula(r'z\ \longrightarrow\ v_3',528,275,125,12,17)
    d.box(621,246,129,54,GREEN,'#F1F8F5')
    d.label('返回第三组投票',685.5,263,9.5,True,GREEN)
    d.formula(r'\widehat y=2v_3-1',685.5,275,118,11,17)
    d.arrow([(597,273),(621,273)],GREEN)
    d.label('同票即停止；异票才计算第三组，等价于三组多数投票。',217,265,10,True)
    d.label('共享特征每图只计算一次，投票 1 / 0 分别对应人脸 / 非人脸。',217,286,9,color=GRAY)
    c.restoreState()
    return top-320
