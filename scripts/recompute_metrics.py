"""Replay stored outer-fold decisions. This does not retrain or re-infer networks."""
from pathlib import Path
import json
import numpy as np
root=Path(__file__).resolve().parents[1];z=np.load(root/'evidence/S01源图折外决策.npz');y=z['labels']==1;d=z['decisions'];w=np.array([.5]+[.5/15]*15)
assert d.shape==(20,800);assert np.array_equal(d,z['initial_decisions']|(z['rescue_votes'].sum(0)>=2))
acc=(d==y[None]).mean(1);rec=d[:,y].mean(1);spec=(~d[:,~y]).mean(1)
result={'clean_accuracy':float(acc[0]),'weighted_accuracy':float(w@acc[:16]),'weighted_recall':float(w@rec[:16]),'weighted_specificity':float(w@spec[:16]),'held_accuracy':float(acc[16:].mean()),'TP':int((d[0]&y).sum()),'TN':int((~d[0]&~y).sum()),'FN':int((~d[0]&y).sum()),'FP':int((d[0]&~y).sum())}
expected=json.loads((root/'report_summary.json').read_text())['metrics']
for k,v in result.items():assert abs(v-expected[k])<1e-12,(k,v,expected[k])
print(json.dumps(result,indent=2))
