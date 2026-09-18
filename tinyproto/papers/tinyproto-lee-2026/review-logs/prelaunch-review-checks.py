"""Read-only prelaunch audit; observations reproduce gaps, not successful training.

Run from the repository root through scripts/run_local_checked.py.
No Kaggle requests, CUDA use, data decode, or notebook regeneration.
"""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts import gen_notebooks as G
from scripts.validate_notebooks import validate
from src.sweep import candidate_name, plan_sweep, selected_mu, selection_signature

base = ROOT / 'papers/tinyproto-lee-2026/notebook'
failures = []
for d in sorted(base.iterdir()):
    if not d.is_dir():
        continue
    validate(d, failures)
    meta = json.loads((d / 'kernel-metadata.json').read_text())
    nb = json.loads((d / meta['code_file']).read_text())
    assert meta['docker_image'] == G.RUNTIME['docker_image']
    assert meta['id'].split('/')[0] == 'odixe0502'
    assert nb['metadata']['kaggle']['accelerator'] == 'nvidiaTeslaT4'
    assert nb['metadata']['kaggle']['isGpuEnabled'] is True
    slug = d.name
    n = next((n for n in (20, 50, 100) if f'{n}client' in slug), 20)
    assert meta['dataset_sources'] == [G.DATASETS[n], G.CENTRALIZED]
    if 'train-' in slug:
        expected = G.train_notebook(n, 50, 256 if n == 100 else 512, 11.0, 'inv_mean_nij', '1.0')
        sweep_slug = 'tinyproto-fp-mu-sweep' + (f'-{n}client' if n != 20 else '')
        assert meta['kernel_sources'] == [f'odixe0502/{sweep_slug}']
    elif 'mu-sweep' in slug:
        expected = G.mu_notebook(n)
    else:
        expected = G.calib_notebook()
    actual_cells = [(c['cell_type'], ''.join(c['source'])) for c in nb['cells']]
    expected_cells = [(c.cell_type, c.source) for c in expected.cells]
    assert actual_cells == expected_cells, slug
    assert all(not c.get('outputs') and c.get('execution_count') is None
               for c in nb['cells'] if c['cell_type'] == 'code')
    print('PASS exact generated cells, embedded modules, runtime, sources:', slug)
assert not failures, failures

g = {'print': lambda *a, **kw: None}
exec(G.MU_CFG, g)
cfg = dict(g['CFG'], model_cfg={}, feature_cols=['f'], class_names=['c'],
           data_fingerprint='fixture-footer', scaler={'mean': 0})
grid = g['MU_GRID']
doc = {'scenario': cfg['scenario'], 'status': 'complete',
       'selection_signature': selection_signature(cfg), 'grid': grid, 'rounds': 4,
       'validation_fingerprint': {'fraction': .02, 'seed': 42},
       'results': {str(k): {'rounds': 4, 'last_proto_f1_macro': .5, 'mu_absolute': 1e-5}
                   for k in grid},
       'winner': {'k': .03, 'mu_absolute': 1e-5}}
assert selected_mu(doc, cfg)[0] == 1e-5
mutations = {
    'only 1 sweep round': lambda d: d.update(rounds=1),
    'only 1 grid candidate': lambda d: d.update(grid=[.03]),
    'no candidate results': lambda d: d.update(results={}),
    'different validation fraction': lambda d: d['validation_fingerprint'].update(fraction=.5),
    'different validation seed': lambda d: d['validation_fingerprint'].update(seed=123),
    'winner not in grid': lambda d: d['winner'].update(k=999),
}
for label, mutate in mutations.items():
    altered = copy.deepcopy(doc)
    mutate(altered)
    assert selected_mu(altered, cfg)[0] == 1e-5
    print('OBSERVED selected_mu accepts:', label)
assert selected_mu(doc, dict(cfg, scaler={'mean': 999}))[0] == 1e-5
print('OBSERVED selection signature does not cover scaler')

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    previous = copy.deepcopy(doc)
    previous.update(status='incomplete')
    (root / 'mu_sweep.json').write_text(json.dumps(previous))
    # Simulate an attachment missing the first candidate directory, while another has markers.
    done = root / 'runs' / candidate_name(cfg['scenario'], .1) / 'complete'
    done.mkdir(parents=True)
    (done / 'round_001.done').write_text('{}')
    plan = plan_sweep(grid, cfg['scenario'], [root])
    started = {k for k, v in plan.items() if v['committed_rounds'] > 0}
    assert started and '0.03' not in started
    print('OBSERVED SWEEP_RESUME any-started gate accepts partial attachment;',
          'missing previous candidate 0.03 will be scheduled fresh')

with tempfile.TemporaryDirectory() as td:
    original = base / 'tinyproto-fp-mu-sweep'
    target = Path(td) / original.name
    target.mkdir()
    meta = json.loads((original / 'kernel-metadata.json').read_text())
    meta.pop('docker_image')
    nb = json.loads((original / meta['code_file']).read_text())
    nb['metadata'].pop('kaggle')
    (target / 'kernel-metadata.json').write_text(json.dumps(meta))
    (target / meta['code_file']).write_text(json.dumps(nb))
    gaps = []
    validate(target, gaps)
    assert not gaps, gaps
    print('OBSERVED current validator accepts missing docker_image AND notebook GPU metadata')

# Audit the actual static worker assignment without importing torch or decoding parquet.
tree = ast.parse((ROOT / 'src/data.py').read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'plan_gpus')
scope = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<plan_gpus>', 'exec'), scope)
for n in (20, 50, 100):
    stats = json.loads((ROOT.parent / f'dataset/fl_client/alpha05/{n}_client/client_stats.json').read_text())
    rows = {r['client_id']: r['rows'] for r in stats['clients']}
    assignment = scope['plan_gpus'](rows, 2)
    counts = [len(b) for b in assignment]
    print('MEASURED sidecar assignment:', n, 'clients/GPU=', counts,
          'eval factor vs perfect half=', max(counts)/(n/2))

print('AUDIT CHECKS COMPLETE; observations above are remaining review gaps, not fixed code.')
