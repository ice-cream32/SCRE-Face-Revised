"""Predict every input row using S01. No label or reference file is read."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
from scipy.io import loadmat
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from scre_face import SharedRuntime
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('data/test_data.mat'));p.add_argument('--key',default='test_data');p.add_argument('--output',type=Path,default=Path('outputs/result_S01.txt'));p.add_argument('--device',choices=['cpu','mps'],default='cpu');p.add_argument('--batch-size',type=int,default=64)
a=p.parse_args();torch.set_num_threads(4);matrix=loadmat(a.input)[a.key]
if matrix.ndim!=2 or matrix.shape[1]!=361:raise ValueError('Expected N x 361 matrix')
images=torch.from_numpy(matrix.reshape(-1,19,19).transpose(0,2,1).copy()).float()[:,None]
labels,stats=SharedRuntime(device=a.device).predict(images,a.batch_size)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(''.join(f'{i} {int(y)}\n' for i,y in enumerate(labels,1)),encoding='ascii')
a.output.with_suffix('.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2));print(json.dumps(stats,ensure_ascii=False,indent=2))
