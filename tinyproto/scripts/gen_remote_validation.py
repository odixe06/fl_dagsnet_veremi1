"""Create the bounded Kaggle validation notebook from the same embedded production modules."""
import json
import argparse
from pathlib import Path
import nbformat as nbf
from gen_notebooks import ROOT, ENV_CELL, module_cells, metadata, DATASETS, CENTRALIZED

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--owner',default='odixe0502')
    args=ap.parse_args()
    title='TinyProto FP 2T4 validation'; slug='tinyproto-fp-2t4-validation'
    out=ROOT/'papers/tinyproto-lee-2026/validation'/slug; out.mkdir(parents=True,exist_ok=True)
    setup='''from pathlib import Path
import json, sys
root=Path('/kaggle/working')
(root/'src').symlink_to(root/'proj',target_is_directory=True) if not (root/'src').exists() else None
for sub in ('tests','scripts','knowledge'):
    (root/sub).mkdir(exist_ok=True)
    if sub!='knowledge': (root/sub/'__init__.py').write_text('')
'''
    for name in ('meta.json','scaler.json'):
        setup+=f"(root/'knowledge/{name}').write_text({(ROOT/'knowledge'/name).read_text()!r})\n"
    nb=nbf.v4.new_notebook()
    nb.cells=[nbf.v4.new_markdown_cell('# Bounded 2×T4 validation\n\nReal-data samples only, four clients per scenario, three rounds; verifies workers, AMP, compile fallback, artifacts and 20-client-sample resume. Not a full-data calibration or production result.'),nbf.v4.new_code_cell(ENV_CELL),*module_cells(),nbf.v4.new_code_cell(setup)]
    for f in ('tests/run_e2e.py','scripts/verify_run.py','tests/remote_smoke.py'):
        nb.cells.append(nbf.v4.new_code_cell(f'%%writefile /kaggle/working/{f}\n'+(ROOT/f).read_text()))
    nb.cells.append(nbf.v4.new_code_cell("import subprocess, sys\nsubprocess.run([sys.executable, '/kaggle/working/tests/remote_smoke.py'], check=True)"))
    nb.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
    nb.metadata['language_info']={'name':'python','version':'3.12'}
    nb.metadata['kaggle']={'accelerator':'nvidiaTeslaT4','isGpuEnabled':True,
                           'isInternetEnabled':True,'language':'python','sourceType':'notebook'}
    nbf.write(nb,out/f'{slug}.ipynb')
    meta = metadata(title,args.owner,[DATASETS[n] for n in (20,50,100)]+[CENTRALIZED])
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(out)
if __name__=='__main__': main()
