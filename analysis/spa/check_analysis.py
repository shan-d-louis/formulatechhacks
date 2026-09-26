"""Independent recount: python check_analysis.py /path/to/spa_telemetry.csv."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
p=pd.read_csv(sys.argv[1]); root=Path(__file__).parent
s=json.loads((root/'summary.json').read_text()); t=p.timestamp.to_numpy(); expected=[]
def runs(mask):
    start=None
    result=[]
    for i,active in enumerate(mask):
        if start is not None and (not active or (i and t[i]-t[i-1]>.1)):
            result.append((start,i-1));start=None
        if active and start is None: start=i
    if start is not None:result.append((start,len(mask)-1))
    return result
braking=(p.brake.to_numpy()>.2)&(p.speed_kmh.to_numpy()>50)
assert sum(t[b]-t[a]>=.2 for a,b in runs(braking))==s['braking_episodes']==92
for wheel,radius in s['effective_radius_m'].items():
    ratio=p['wheel_speed_'+wheel].to_numpy()*radius/(p.speed_kmh.to_numpy()/3.6)-1
    for a,b in runs(braking&(ratio<-.2)):
        if t[b]-t[a]>=.08:expected.append((wheel,t[a]-t[0],t[b]-t[0]))
events=pd.read_csv(root/'candidate_events.csv')
assert len(expected)==len(events)==5
for w,a,b in expected:
    assert ((events.wheel==w)&np.isclose(events.start_s,a)&np.isclose(events.end_s,b)).sum()==1
windows=pd.read_csv(root/'event_windows.csv')
assert windows.event_id.nunique()==5
assert len(p)==s['rows'] and len(p.columns)==s['columns']
print('PASS: independent event recount, braking count, exported window IDs and dataset dimensions')
