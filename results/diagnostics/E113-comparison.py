import json
from pathlib import Path
import numpy as np


def estimate(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1 and len(values) > 1 and np.isfinite(values).all()
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))

assert estimate([0., 2.]) == (1., 1.)
comparisons=[]
for seed in range(3):
    a_path, = Path('runs').glob(f'*_E112-v3-A{seed}')
    d_path, = Path('runs').glob(f'*_E113-direct-s{seed}')
    a_summary=json.loads((a_path/'summary.json').read_text())
    d_summary=json.loads((d_path/'summary.json').read_text())
    cfg=json.loads((d_path/'config.json').read_text())
    assert a_summary['status']==d_summary['status']=='finished'
    assert d_summary['generation']==200 and cfg['args']['seed']==seed
    assert cfg['controller']=='direct-rnn-v1' and cfg['args']['condition_split']=='final'
    assert a_summary['eval_seed']==d_summary['eval_seed']==2026111901
    with np.load(a_path/'events.npz',allow_pickle=False) as a, np.load(d_path/'events.npz',allow_pickle=False) as d:
        for key in ('wkeys','dkeys','rules'): np.testing.assert_array_equal(a[key],d[key])
        av,dv=(np.asarray(z['stationary_genome_fitness'],dtype=np.float64) for z in (a,d))
        assert av.shape==dv.shape==(256,)
        am,ase=estimate(av);dm,dse=estimate(dv);delta,se=estimate(av-dv)
        assert am==a_summary['evaluations']['stationary']['genome']['fitness']
        assert dm==d_summary['final_direct']['fitness']
        comparisons.append(dict(seed=seed,a_run=a_path.as_posix(),direct_run=d_path.as_posix(),
                                a_fitness=am,a_se=ase,direct_fitness=dm,direct_se=dse,
                                paired_difference=delta,paired_se=se,
                                a_clear_advantage=delta>2*se,
                                interpretation='A advantage' if delta>2*se else 'direct advantage' if delta < -2*se else 'unresolved',
                                per_world_difference=(av-dv).tolist()))
result=dict(experiment='E113',n_worlds=256,eval_seed=2026111901,primary='stationary fitness',
            rule='Continue only if A_i - direct_i > 2 paired world SE in all three seeds',
            continue_beyond_phase2=all(r['a_clear_advantage'] for r in comparisons),comparisons=comparisons)
path=Path('results/diagnostics/E113-comparison.json')
assert not path.exists(), 'refuse to overwrite prior comparison'
path.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({**result,'comparisons':[{k:v for k,v in r.items() if k!='per_world_difference'} for r in comparisons]},indent=2))
