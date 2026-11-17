"""Replay historical 99-point search on 160 source-validation originals only."""
from pathlib import Path
import json
import numpy as np
r=Path(__file__).resolve().parents[1];z=np.load(r/'evidence/historical_threshold/validation_scores.npz');p=np.mean([z[k] for k in ['resnet8','densenet121','hires19']],axis=0).reshape(8,160);y=z['labels']==1;t=np.arange(1,100)/100;w=np.array([.5]+[.5/7]*7)
acc=((p[None]>=t[:,None,None])==y[None,None,:]).mean(2)@w
j=min(range(99),key=lambda k:(-round(float(acc[k]),12),abs(t[k]-.5),t[k]));assert t[j]==.37
print(json.dumps({'threshold':float(t[j]),'weighted_validation_accuracy':float(acc[j]),'grid_size':99,'independent_originals':160,'note':'Historical development score; current S01 inherits .37.'},indent=2))
