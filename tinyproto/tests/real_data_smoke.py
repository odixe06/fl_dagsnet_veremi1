"""Bounded real-data CPU forward/backward and remote-test sample preflight."""
import json
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.model import build_model
from src.protos import ProtoAccumulator, ProtoRegularizer
from src.cps import build_masks
ROOT=Path(__file__).resolve().parents[1]
features=json.loads((ROOT/'knowledge/meta.json').read_text())['feature_cols']
torch.set_num_threads(1)
for n in (20,50,100):
    f=next((Path('/home/odixe/nckh/dataset/fl_client/alpha05')/f'{n}_client/train/client_id=000').glob('*.parquet'))
    b=next(pq.ParquetFile(f).iter_batches(batch_size=64,columns=features+['label']))
    x=torch.from_numpy(np.column_stack([b.column(c).to_numpy() for c in features]).astype(np.float32))
    y=torch.from_numpy(b.column('label').to_numpy().astype(np.int64))
    torch.manual_seed(42); model=build_model(); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    h,logits=model.forward_both(x); loss=torch.nn.functional.cross_entropy(logits,y)
    loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    assert torch.isfinite(loss) and torch.isfinite(norm)
    opt.step(); model.eval()
    with torch.no_grad():
        h,_=model.forward_both(x); acc=ProtoAccumulator(16,256,'cpu'); acc.update(h,y); p,c=acc.finish()
    mask=torch.from_numpy(build_masks(256,16,50).astype(np.float32))
    reg=ProtoRegularizer(p*mask,c>0,mask,torch.tensor(1.),50)
    model.train(); opt.zero_grad(); h,lo=model.forward_both(x)
    total=torch.nn.functional.cross_entropy(lo,y)+reg(h,y); total.backward()
    assert torch.isfinite(total) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print(n,'clients: 64 real rows; CE + prototype regularizer forward/backward PASS',flush=True)
raw=[]
for f in sorted(Path('/home/odixe/nckh/dataset/centralized/test').glob('*.parquet')):
    pf=pq.ParquetFile(f)
    targets=np.sort(np.random.default_rng(42).choice(pf.metadata.num_rows,min(256,pf.metadata.num_rows),replace=False))
    offset=0
    for b in pf.iter_batches(batch_size=65536,columns=['f_snd_spd'],use_threads=False):
        selected=targets[(targets>=offset)&(targets<offset+b.num_rows)]-offset
        if len(selected): raw.append(b.column(0).to_numpy()[selected])
        offset+=b.num_rows
mean=float(np.concatenate(raw).mean())
print('Remote test sample: rows',sum(map(len,raw)),'raw f_snd_spd mean',mean)
assert 6.5<mean<8.5, 'sample does not meet full-test sentinel; adjust remote sample design'
