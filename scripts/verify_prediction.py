"""Check format and byte-independent classification equality; never patches labels."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('file',type=Path);p.add_argument('--reference',type=Path,default=Path('results/test_predictions.txt'));p.add_argument('--count',type=int,default=420);a=p.parse_args()
x=np.loadtxt(a.file,dtype=np.int64);assert x.shape==(a.count,2);assert np.array_equal(x[:,0],np.arange(1,a.count+1));assert np.isin(x[:,1],[-1,1]).all()
r=np.loadtxt(a.reference,dtype=np.int64);assert r.shape==x.shape
diff=x[x[:,1]!=r[:,1],0].tolist();print(json.dumps({'rows':len(x),'different_ids':diff,'identical':not diff}));assert not diff
