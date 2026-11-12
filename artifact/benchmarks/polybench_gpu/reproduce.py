"""Figure 4: fresh V3 builds, native CUDA baselines, full outputs and timings."""
import argparse
import json
from pathlib import Path
import shutil
from common import ARTIFACT, OUT, PYTHON, command, save
from build import CASES

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['build','run','summarize','all'],default='all')
    p.add_argument('--cases',nargs='+',choices=CASES,default=CASES);a=p.parse_args()
    if a.stage in ('build','all'):
        save(OUT/'cases.json',a.cases)
        for name in ['iteration_driver.c','paper_targets.csv']:shutil.copy2(HERE/name,OUT/name)
        command([PYTHON,HERE/'build.py',*a.cases],OUT/'build.log')
        for name in a.cases:
            record=json.loads((OUT/name/'build.json').read_text())
            if record['source_build_rc'] or record['gpu_build_rc']:
                raise RuntimeError(f'{name}: compiler build failed')
        command([PYTHON,HERE/'build_native.py',*a.cases],OUT/'build-native.log')
        special=[n for n in a.cases if n in ('correlation','covariance')]
        if special:command([PYTHON,HERE/'build_paper_baselines.py',*special],OUT/'build-paper.log')
        command([PYTHON,HERE/'save_provenance.py'],OUT/'snapshot.log')
    if a.stage in ('run','summarize'):a.cases=json.loads((OUT/'cases.json').read_text())
    if a.stage in ('run','all'):
        for name in a.cases:
            for variant in ['reference', 'paper-baseline' if name in ('correlation','covariance') else 'native', 'raised']:
                command([PYTHON,HERE/'run.py','--remote-helper',ARTIFACT/'remote.py',name,variant],OUT/'run.log')
            command([PYTHON,HERE/'run.py','--remote-helper',ARTIFACT/'remote.py','--trace',name,'raised'],OUT/'run.log')
    if a.stage in ('run','summarize','all'):
        command([PYTHON,HERE/'audit.py'],OUT/'audit.log')
        command([PYTHON,HERE/'report.py'],OUT/'report.log')
        print('Results:', OUT/'results.csv')

if __name__=='__main__':main()
