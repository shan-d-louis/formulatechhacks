"""Run: python analyze_spa.py path/to/spa_telemetry.csv. Requires pandas, numpy."""
import sys, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
src=Path(sys.argv[1]); out=Path(__file__).resolve().parent
p=pd.read_csv(src)
required=['timestamp','speed_kmh','brake','throttle','g_lat']+['wheel_speed_'+w for w in ['fl','fr','rl','rr']]
if not set(required).issubset(p.columns) or len(p)<2:
    raise ValueError('Missing required channels or insufficient rows')
if not np.isfinite(p[required].to_numpy(dtype=float)).all():
    raise ValueError('Required channels contain missing or nonfinite values')
t=p.timestamp-p.timestamp.iloc[0]; dt=t.diff()
if (dt.dropna()<=0).any():
    raise ValueError('Timestamps must strictly increase')
# ponytail: exploratory thresholds, review video and independent sessions before using as labels.
def episodes(mask, minimum):
    groups=(mask.ne(mask.shift()) | dt.gt(.1)).cumsum()
    return [(g.index[0],g.index[-1]) for _,g in p[mask].groupby(groups[mask]) if t.loc[g.index[-1]]-t.loc[g.index[0]]>=minimum]
base=(p.speed_kmh>80)&(p.brake<.01)&(p.throttle<.15)&(p.g_lat.abs()<.2)
radii={w:float(((p.speed_kmh/3.6)/p['wheel_speed_'+w]).where(base & (p['wheel_speed_'+w]>1)).median()) for w in ['fl','fr','rl','rr']}
if not all(np.isfinite(r) and r>0 for r in radii.values()):
    raise ValueError('Insufficient rolling-radius calibration data')
slip=pd.DataFrame({w:p['wheel_speed_'+w]*r/(p.speed_kmh.where(p.speed_kmh>0)/3.6)-1 for w,r in radii.items()})
rows=[]
for w in radii:
 for a,b in episodes((p.brake>.2)&(p.speed_kmh>50)&(slip[w]<-.2),.08):
  rows.append(dict(wheel=w,start_s=float(t[a]),end_s=float(t[b]),duration_s=float(t[b]-t[a]),min_proxy_slip=float(slip.loc[a:b,w].min()),start_speed_kmh=float(p.speed_kmh[a]),peak_brake=float(p.brake.loc[a:b].max())))
events=pd.DataFrame(rows,columns=['wheel','start_s','end_s','duration_s','min_proxy_slip','start_speed_kmh','peak_brake']).sort_values('start_s')
events.to_csv(out/'candidate_events.csv',index=False)
quality=p.describe().T;quality['unique_values']=p.nunique();quality.to_csv(out/'channel_summary.csv')
summary=dict(rows=len(p),columns=len(p.columns),duration_s=float(t.iloc[-1]),mean_hz=float((len(p)-1)/t.iloc[-1]),median_dt_ms=float(dt.median()*1000),max_gap_s=float(dt.max()),gaps_over_100ms=int((dt>.1).sum()),nonpositive_deltas=int((dt<=0).sum()),missing_cells=int(p.isna().sum().sum()),constant_columns=p.columns[p.nunique()==1].tolist(),calibration_samples=int(base.sum()),effective_radius_m=radii,braking_episodes=len(episodes((p.brake>.2)&(p.speed_kmh>50),.2)),candidate_wheel_events=len(events),sha256=hashlib.sha256(src.read_bytes()).hexdigest())
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2));print(events.to_string(index=False))

# Export full-rate context around every candidate, retaining overlaps by event ID.
windows=[]
for event_id,e in enumerate(events.itertuples(),1):
    mask=(t>=e.start_s-2)&(t<=e.end_s+2)
    window=p.loc[mask,required+['tyre_core_'+w for w in radii]+['psi_'+w for w in radii]].copy()
    window.insert(0,'event_id',event_id)
    window.insert(1,'elapsed_s',t[mask])
    for w in radii: window['slip_proxy_'+w]=slip.loc[mask,w]
    windows.append(window)
if windows: pd.concat(windows).to_csv(out/'event_windows.csv',index=False)
sensitivity=[]
for cutoff in [-.1,-.2,-.3]:
    for duration in [.08,.15]:
        count=sum(len(episodes((p.brake>.2)&(p.speed_kmh>50)&(slip[w]<cutoff),duration)) for w in radii)
        sensitivity.append(dict(proxy_threshold=cutoff,minimum_duration_s=duration,wheel_events=count))
pd.DataFrame(sensitivity).to_csv(out/'threshold_sensitivity.csv',index=False)
