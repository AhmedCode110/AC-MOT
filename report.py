"""Retain every candidate and freeze only development-qualified configurations."""
import argparse
import json
from pathlib import Path
import numpy as np
from core import atomic_json,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=json.loads((a.run/'configuration.json').read_text())
    data=json.loads((a.evaluation/'metrics.json').read_text())
    baseline='A0_pinned_default';reference='A3_adaptive_reference'
    if baseline not in data or reference not in data: raise ValueError('Both control systems are required')
    a.output.mkdir(parents=True,exist_ok=False)
    def scores(r):
        return dict(HOTA=float(np.mean(r['HOTA']['combined']['HOTA'])),
                    MOTA=float(r['CLEAR']['combined']['MOTA']),IDS=int(r['CLEAR']['combined']['IDSW']))
    bs=scores(data[baseline]);rs=scores(data[reference]);rows=[]
    rng=np.random.default_rng(20260906)
    for name,r in data.items():
        sc=scores(r);seqs=sorted(r['CLEAR']['per_sequence'])
        # Cluster clips with the same UAV source prefix to avoid treating them as independent videos.
        groups={s.split('_')[0] for s in seqs}
        clusters=[np.array([float(r['CLEAR']['per_sequence'][s]['MOTA'])-
                    float(data[baseline]['CLEAR']['per_sequence'][s]['MOTA']) for s in seqs if s.split('_')[0]==g]) for g in sorted(groups)]
        samples=[float(np.concatenate([clusters[i] for i in rng.integers(0,len(clusters),len(clusters))]).mean()) for _ in range(2000)]
        qualified=all(sc['MOTA']>=v['MOTA'] and sc['IDS']<=v['IDS'] and sc['HOTA']>=v['HOTA'] for v in [bs,rs])
        rows.append(dict(system=name,**sc,qualified=qualified,
            macro_MOTA_delta_ci95=np.percentile(samples,[2.5,97.5]).tolist(),
            uncertainty_scope='Exploratory paired source-cluster bootstrap of macro MOTA delta vs A0; not combined-score CI or selection-adjusted'))
    atomic_json(a.output/'all_candidates.json',rows)
    eligible=[r for r in rows if r['qualified'] and r['system'] not in [baseline,reference]]
    if cfg['split']=='development' and eligible:
        winner=max(eligible,key=lambda r:(r['HOTA'],r['MOTA'],-r['IDS']))['system']
        selected=[c for c in cfg['systems'] if c['name'] in [baseline,reference,winner]]
        atomic_json(a.output/'frozen.json',dict(development_split='development',systems=selected,
            weights_sha256=cfg['weights_sha256'],environment=cfg['environment'],
            selection_rule='HOTA maximum subject to MOTA/HOTA not lower and IDS not higher than both A0 and reference A3',
            winner=winner,evaluation_sha256=sha(a.evaluation/'metrics.json'),
            warning='Development selection, not evidence of held-out improvement'))
        print('Development candidate frozen:',winner)
    else:
        print('No candidate frozen. No jointly qualifying candidate, or this is test data.')

if __name__=='__main__':main()
