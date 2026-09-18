"""Sequential CPU regression suite. Invoke through run_local_checked.py on WSL."""
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'papers/tinyproto-lee-2026/review-logs'
LOG.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')

def run(name,*args):
    print('START',name,flush=True)
    with (LOG/f'{name}.log').open('w') as f:
        result=subprocess.run([sys.executable,*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        print((LOG/f'{name}.log').read_text()[-4000:],flush=True)
        raise SystemExit(f'FAIL {name}: exit {result.returncode}')
    print('PASS',name,flush=True)

run('fixture-final','tests/make_fixture.py','/tmp/tinyproto_fixture','--rows','2048','--test-rows','1024')
run('unit-final','tests/unit_test.py')
run('regressions-final','tests/review_regressions.py')
run('generate-final','scripts/gen_notebooks.py')
run('validate-final','scripts/validate_notebooks.py')
for slug in ('tinyproto-fp-calibration','tinyproto-fp-mu-sweep','tinyproto-fp-mu-sweep-50client',
             'tinyproto-fp-mu-sweep-100client','tinyproto-fp-train-20client',
             'tinyproto-fp-train-50client','tinyproto-fp-train-100client'):
    run(slug+'-final','tests/notebook_sim.py',slug,'--work','/tmp/tp_final_sim')
run('verify-final','scripts/verify_run.py','/tmp/tp_final_sim/runs/sim_100client','--require-complete')
run('resume-final','tests/resume_test.py')
run('tamper-final','tests/tamper_test.py','/tmp/tp_resume1/runs/e2e')
print('REVIEW SUITE PASSED',flush=True)
