import json,numpy as np
from pathlib import Path
from scipy.stats import ttest_ind
from brass.orbits.size_comparison import paired_groups,holm
from brass.orbits.io import stable_seed,write_json
root=Path('results/orbits/2026-09-19-size-32b')
rows=json.loads((root/'pre_scoring_analysis/paired_features.json').read_text())
a=np.array(list(paired_groups(rows,'response_drift',cohort='attack').values()))
b=np.array(list(paired_groups(rows,'response_drift',cohort='benign').values()))
results=[]
for name,x,y in [('32b_attack_minus_benign',a[:,1],b[:,1]),('change_in_attack_benign_gap_32b_minus_7b',a[:,1]-a[:,0],b[:,1]-b[:,0])]:
 rng=np.random.default_rng(stable_seed(235711,name))
 boot=rng.choice(x,(10000,len(x)),replace=True).mean(1)-rng.choice(y,(10000,len(y)),replace=True).mean(1)
 results.append({'label':name,'difference':float(x.mean()-y.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'p':float(ttest_ind(x,y,equal_var=False).pvalue),'attack_groups':len(x),'benign_groups':len(y)})
holm(results)
record={'status':'exploratory_post_outcome_not_primary','metric':'response drift after two cycles','method':'Use the primary paired-complete-case task groups; independent task-group bootstrap for cohort contrasts, 10000 draws; Welch tests; Holm across these two exploratory contrasts. Not adjusted for task/corpus or initial success.','contrasts':results}
write_json(root/'exploratory_attack_benign_contrasts.json',record)
print(json.dumps(record,indent=2))
