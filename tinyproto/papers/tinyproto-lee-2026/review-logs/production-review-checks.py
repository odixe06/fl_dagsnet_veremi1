"""Read-only production audit; run through scripts/run_local_checked.py, CPU only.

Temporary notebook mutations exercise validator gaps. No credentials are executed/printed,
no source notebooks are rewritten, no Kaggle or W&B requests are made.
"""
import ast
import contextlib
import io
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts import gen_notebooks as G
from scripts.validate_notebooks import validate
from src import ckpt as C
from src.data import footer_fingerprint, plan_gpus
from src.model import CFG as MODEL_CFG, FlatPacker, build_model
from src.sweep import selected_mu

PAPER = ROOT / 'papers/tinyproto-lee-2026'
DATA = ROOT.parent / 'dataset'
TMP = Path('/tmp/claude-1000/-home-odixe-nckh-tinyproto')


def cfg_of(nb):
    for cell in nb['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and source.startswith('CFG = '):
            return ast.literal_eval(ast.parse(source).body[0].value)
    raise AssertionError('CFG not found')


def audit_notebook(directory):
    meta = json.loads((directory / 'kernel-metadata.json').read_text())
    nb = json.loads((directory / meta['code_file']).read_text())
    failures = []
    validate(directory, failures)
    print('NOTEBOOK', meta['id'], 'problems=', failures)
    if 'train-' in meta['id']:
        cfg = cfg_of(nb)
        print('  config', {k: cfg[k] for k in ('rounds', 'batch', 'max_hours',
              'rounds_this_session', 'require_resume', 'require_mu_selection')})
        print('  dataset_sources=', meta['dataset_sources'],
              'kernel_sources=', meta.get('kernel_sources'))
    return nb


for directory in sorted((PAPER / 'notebook').iterdir()):
    if (directory / 'kernel-metadata.json').exists():
        audit_notebook(directory)
production = {}
for directory in sorted((PAPER / 'production').iterdir()):
    if (directory / 'kernel-metadata.json').exists():
        nb = audit_notebook(directory)
        cfg = cfg_of(nb)
        sources = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
        assert G.CONFIG_BUILD in sources and G.LAUNCH_CELL in sources
        assert all(c.get('execution_count') is None and not c.get('outputs')
                   for c in nb['cells'] if c['cell_type'] == 'code')
        embedded = {Path(s.split('\n', 1)[0].split()[-1]).name
                    for s in sources if s.startswith('%%writefile')}
        assert embedded == set(G.MODULES)
        meta = json.loads((directory / 'kernel-metadata.json').read_text())
        production[directory.name] = (cfg, meta)
assert len(production) == 7
for n, owners, budgets in ((20, ['minhtran0601'], [11]),
                           (50, ['odixe0502']*2, [11, 7.5]),
                           (100, ['khanhmay0304']*3+['minhtrit06'], [11, 11, 8, 11])):
    for session, (owner, budget) in enumerate(zip(owners, budgets), 1):
        slug = f'tinyproto-fp-train-{n}client-s{session}'
        cfg, meta = production[slug]
        assert meta['id'] == f'{owner}/{slug}'
        assert cfg['max_hours'] == budget and cfg['require_resume'] == (session > 1)
        assert meta['dataset_sources'] == [G.DATASETS[n], G.CENTRALIZED,
                                          f'odixe0502/tinyproto-fp-mu-{n}client']
        prev = [f'{owners[session-2]}/tinyproto-fp-train-{n}client-s{session-1}'] if session>1 else []
        assert meta['kernel_sources'] == prev
        assert cfg['rounds'] == 50 and cfg['save_preds_rounds'] == [50]
    print('PRODUCTION CHAIN PASS', n, 'owners/budgets/attachments/glue/source/clean outputs')

central = DATA / 'centralized'
features = json.loads((central / 'feature_schema.json').read_text())['feature_columns']
classes = json.loads((central / 'label_mapping.json').read_text())['classes']
scaler = central / 'scaler.json'
for n in (20, 50, 100):
    directory = PAPER / 'production' / f'tinyproto-fp-train-{n}client-s1'
    nb = json.loads((directory / f'{directory.name}.ipynb').read_text())
    cfg = cfg_of(nb)
    mu_dir = PAPER / 'mu-datasets' / f'tinyproto-fp-mu-{n}client'
    suffix = '' if n == 20 else f'-{n}client'
    saved = PAPER / 'review-logs' / f'mu-sweep{suffix}-result.json'
    doc = json.loads(saved.read_text())
    assert doc == json.loads((mu_dir / 'mu_sweep.json').read_text())
    print('MU dataset JSON matches saved summary:', n,
          '(byte-identical:', saved.read_bytes() == (mu_dir / 'mu_sweep.json').read_bytes(), ')')
    client_root = DATA / 'fl_client/alpha05' / f'{n}_client'
    stats = json.loads((client_root / 'client_stats.json').read_text())
    rows = {c['client_id']: c['rows'] for c in stats['clients']}
    with tempfile.TemporaryDirectory() as td:
        env = dict(CFG=cfg, CLASS_NAMES=classes, FEATURE_COLS=features, MODEL_CFG=MODEL_CFG,
                   FlatPacker=FlatPacker, build_model=build_model, Path=Path, time=time,
                   ASSIGNMENT=plan_gpus(rows, 2), CLIENT_ROOT=client_root,
                   TEST_DIR=central / 'test', SCALER_PATH=scaler,
                   footer_fingerprint=footer_fingerprint, json=json, shutil=shutil,
                   TEST_ROWS=10761343, TOTAL=43045415)
        code = G.CONFIG_BUILD.replace('from proj.', 'from src.')
        code = code.replace('Path("/kaggle/input")', f'Path({str(mu_dir)!r})')
        code = code.replace('"/kaggle/working"', repr(td))
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, '<actual CONFIG_BUILD with local mounts>', 'exec'), env)
    effective = env['cfg']
    assert effective['mu_kind'] == 'absolute'
    assert effective['mu_value'] == doc['winner']['mu_absolute']
    for k, result in doc['results'].items():
        assert result['rounds'] == len(result['history']) == 4
        assert math.isclose(result['mu_absolute'], float(k) / result['mean_nij'], rel_tol=1e-12)
        assert result['last_proto_f1_macro'] == float(result['history'][-1]['proto_f1_macro'])
    ranked = sorted(doc['results'], key=lambda k: doc['results'][k]['last_proto_f1_macro'], reverse=True)
    gap = doc['results'][ranked[0]]['last_proto_f1_macro'] - doc['results'][ranked[1]]['last_proto_f1_macro']
    round1 = [float(r['history'][0]['proto_f1_macro']) for r in doc['results'].values()]
    print('MU PASS', n, 'k=', doc['winner']['k'], 'absolute=', effective['mu_value'],
          'runner_up_gap=', gap, 'round1_range=', max(round1)-min(round1))
    effective.update(artifact_version=2, scaler=json.loads(scaler.read_text()),
                     validation_fingerprint='full-train-fixed-test')
    baseline = C.fingerprint(effective)
    assert C.fingerprint(dict(effective, require_resume=True, max_hours=8)) == baseline
    assert C.fingerprint(dict(effective, mu_value=effective['mu_value']*2)) != baseline
    print('  resume fingerprint stable across session budget:', baseline)
    for name, (session_cfg, _) in production.items():
        if session_cfg['scenario'] != cfg['scenario']:
            continue
        assert {k for k in cfg if cfg[k] != session_cfg[k]} <= {'max_hours', 'require_resume'}
        changed = dict(effective, max_hours=session_cfg['max_hours'],
                       require_resume=session_cfg['require_resume'])
        assert C.fingerprint(changed) == baseline
        assert selected_mu(doc, changed)[0] == effective['mu_value']
        print('  actual session mu + fingerprint PASS:', name)
    # Confirm provenance against downloaded configs/metrics without loading model tensors.
    sources = list(TMP.glob(f'*/scratchpad/mu-sweep{suffix}/runs/*/config.json'))
    assert len(sources) == 5, (n, len(sources))
    for cp in sources:
        pulled = json.loads(cp.read_text())
        assert selected_mu(doc, pulled)[0] == effective['mu_value']
        k = str(pulled['mu_value'])
        metrics = json.loads((cp.parent/'metrics/round_004.json').read_text())
        actual_f1 = metrics['aggregate']['proto']['mean_over_clients']['f1_macro']
        assert actual_f1 == doc['results'][k]['last_proto_f1_macro']
    print('  5 downloaded configs and round-4 metrics match summary')

# Metadata-only gaps: valid generated notebooks with intentionally wrong attachments.
def check_variant(title, resume, sources):
    nb = G.train_notebook(100, 50, 256, 11, 'inv_mean_nij', '1.0', require_resume=resume)
    with tempfile.TemporaryDirectory() as td:
        directory = G.write(nb, title, Path(td), 'minhtrit06', sources[0], sources[1])
        failures = []
        validate(directory, failures)
    print('VALIDATOR GAP', title, 'require_resume=', resume,
          'sources=', sources, 'problems=', failures)

ds = [G.DATASETS[100], G.CENTRALIZED, 'odixe0502/tinyproto-fp-mu-100client']
check_variant('TinyProto FP train 100client s2', False, (ds, []))
check_variant('TinyProto FP train 100client s2', True,
              (ds, ['minhtrit06/tinyproto-fp-train-100client-s2']))
check_variant('TinyProto FP train 100client s2', True,
              (ds, ['odixe0502/tinyproto-fp-train-50client-s1']))
check_variant('TinyProto FP train 100client s1', False,
              ([G.DATASETS[100], G.CENTRALIZED, 'odixe0502/unrelated-dataset'], []))

# Exercise the literal budget condition with a constant-cost scenario; not a speed benchmark.
elapsed, worst, completed = 217.0, 2500.0, 0
while elapsed + max(60.0, 1.15 * worst) < 11 * 3600:
    elapsed += 2443.0
    worst = max(worst, 2443.0)
    completed += 1
print('BUDGET SIMULATION constant 2443s/round, setup 217s:', completed,
      'rounds/session;', (44*2443+3*217)/3600, 'hours for 44 rounds + 3 setups')
remaining, counts = 50, []
for hours in (11, 11, 8, 11):
    elapsed, count = 217.0, 0
    while remaining and elapsed + 1.15 * 2500 < hours * 3600:
        elapsed += 2443
        count += 1
        remaining -= 1
    counts.append(count)
print('ACTUAL 100-client budget schedule, same illustrative costs:', counts,
      'rounds, remaining=', remaining, '(projection only)')

# The AMP decision threshold currently grows with the error it is meant to detect.
import torch
from src.evaluate import decisive_agreement
ref, got = torch.tensor([[2., 0.]]), torch.tensor([[0., 1000.]])
delta = float((ref-got).abs().max())
print('AMP GATE GAP wrong argmax with large finite error:',
      'delta=', delta, '(bad, decisive)=', decisive_agreement(ref, got, delta))

# Execute only the actual driver's telemetry block, with a logger that raises.
tree = ast.parse((ROOT / 'src/driver.py').read_text())
block = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
             and ast.unparse(node.test) == 'wandb_run is not None')
class FailedTelemetry:
    def log(self, *args, **kwargs):
        raise RuntimeError('simulated telemetry failure')
env = dict(wandb_run=FailedTelemetry(), RULES=['proto'], METRIC_KEYS=['f1_macro'],
           agg={'proto': {'mean_over_clients': {'f1_macro': .5}}}, extra={}, rnd=1)
try:
    exec(compile(ast.Module(body=[block], type_ignores=[]), '<driver telemetry>', 'exec'), env)
except RuntimeError:
    print('TELEMETRY GAP: exception escapes actual driver logging block')
print('Review complete. GAP observations above are findings, not implemented fixes.')
