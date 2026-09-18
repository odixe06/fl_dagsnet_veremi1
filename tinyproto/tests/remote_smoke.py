"""Bounded real-data 2xT4 verification. Scores from these samples are NOT research results."""
from pathlib import Path
import json
import shutil
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path('/kaggle/working')
sys.path.insert(0,str(ROOT))
from src.data import find_client_root, find_test_root
from src.driver import run
from tests.run_e2e import make_cfg
from scripts.verify_run import verify


def sample_data(n, out):
    root=find_client_root([Path('/kaggle/input')],n)
    test=find_test_root([Path('/kaggle/input')])
    meta=json.loads((ROOT/'knowledge/meta.json').read_text())
    cols=meta['feature_cols']+['label']
    source_clients=[0,n//3,2*n//3,n-1]
    for cid, source_id in enumerate(source_clients):
        parts=[]; remaining=8192
        for f in sorted((root/'train'/f'client_id={source_id:03d}').glob('*.parquet')):
            for b in pq.ParquetFile(f).iter_batches(batch_size=8192,columns=cols):
                take=min(remaining,b.num_rows); parts.append(pa.Table.from_batches([b]).slice(0,take))
                remaining-=take
                if remaining==0: break
            if remaining==0: break
        d=out/'train'/f'client_id={cid:03d}'; d.mkdir(parents=True,exist_ok=True)
        pq.write_table(pa.concat_tables(parts),d/'part.parquet')
    test_parts=[]
    for f in sorted(test.glob('*.parquet')):
        pf=pq.ParquetFile(f)
        targets=np.sort(np.random.default_rng(42).choice(pf.metadata.num_rows,min(256,pf.metadata.num_rows),replace=False))
        offset=0
        for b in pf.iter_batches(batch_size=65536,columns=cols,use_threads=False):
            selected=targets[(targets>=offset)&(targets<offset+b.num_rows)]-offset
            if len(selected): test_parts.append(pa.Table.from_batches([b]).take(selected))
            offset+=b.num_rows
    (out/'test').mkdir(exist_ok=True)
    pq.write_table(pa.concat_tables(test_parts),out/'test/part.parquet')
    print('REAL-DATA SAMPLE',n,'source clients',source_clients,'test rows',sum(t.num_rows for t in test_parts),flush=True)


def main():
    report={'scope':'sampled real data, 4 clients/scenario; not full-data calibration', 'scenarios':{}}
    for n in (20,50,100):
        fixture=Path('/kaggle/temp')/f'review_{n}'
        sample_data(n,fixture)
        cfg,paths=make_cfg(fixture,4,3,['cuda:0','cuda:1'], batch=256 if n==100 else 512,
                           eval_batch=2048,compile_=True,run_name=f'review_real_{n}',
                           extra={'scenario':f'{n}client-sample','max_hours':0.75})
        if n==20:
            run(dict(cfg,rounds_this_session=2),paths,ROOT)
            result=run(dict(cfg,require_resume=True),paths,ROOT)
        else:
            result=run(cfg,paths,ROOT)
        assert result['last_round']==3
        assert verify(Path(result['run_dir']),require_complete=True)==0
        audit=json.loads((Path(result['run_dir'])/'data_audit.json').read_text())
        report['scenarios'][str(n)]={'last_round':3,'batch':cfg['batch'],
                                    'compile':[r['compile'] for r in audit['ready']]}
        (ROOT/'remote_validation.json').write_text(json.dumps(report,indent=2))
    report['status']='passed'
    (ROOT/'remote_validation.json').write_text(json.dumps(report,indent=2))
    print('REMOTE VALIDATION PASSED',flush=True)

if __name__=='__main__':
    main()
