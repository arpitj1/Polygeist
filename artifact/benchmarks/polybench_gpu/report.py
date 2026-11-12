#!/usr/bin/env python3
"""Report only fresh current-v3 evidence and the exact paper baselines."""
import csv,json,statistics
from pathlib import Path
from common import OUT as D
rows=json.loads((D/'audit.json').read_text());summary=[]
lines=['Current-v3 PolyBench GPU Figure4 verification — 2026-11-11','',
'FP64 LARGE; canonical C initialization and complete output arrays.',
'Five warmups, five samples, minimum CUDA-event compute device interval.',
'Allocation and initial/final transfers are excluded from device compute; asynchronous submission wall time, transfers and synchronized end-to-end host time are retained separately.',
'Fresh source-imported v3 specifications, shared Linalg/region matching, typed ABI lowering, and generic GPU residual-loop lowering.',
'Explicit ABI harnesses; isolated-kernel reproduction, not untouched whole-program compilation.','',
'kernel       paper ms     raised ms    native ms    speedup   paper speedup  status']
for row in rows:
 name=row['kernel'];base=row.get('paper_comparison_baseline','native');v=row.get('variants',{});r=v.get('raised',{});b=v.get(base,{})
 run=r.get('run',{});brun=b.get('run',{});ms=run.get('compute_device_ms',{}).get('min_ms');native=brun.get('compute_device_ms',{}).get('min_ms')
 valid=r.get('accepted',False) and b.get('accepted',False) and row.get('vendor_valid',False)
 proof_path=D/name/'export/match.json';proof=json.loads(proof_path.read_text()) if proof_path.exists() else {}
 x=dict(kernel=name,correctness_passed=r.get('correctness',{}).get('passed',False),validated=valid,
        baseline=base,raised_device_ms=ms,baseline_device_ms=native,paper_raised_ms=r.get('paper_ms'),
        device_speedup=row.get('device_speedup'),paper_device_speedup=row.get('paper_speedup'),
        raised_submission_wall_ms=run.get('compute_wall_ms',{}).get('min_ms'),
        raised_memory_device_ms=run.get('memory_device_ms',{}).get('min_ms'),
        raised_end_to_end_ms=run.get('e2e_host_ms',{}).get('min_ms'),
        specifications=','.join(proof.get('emitted',[])))
 summary.append(x)
 if None in (ms,native,x['device_speedup'],x['paper_device_speedup'],x['paper_raised_ms']):
  lines.append(f'{name:12} INCOMPLETE — see audit.json');continue
 lines.append(f"{name:12} {x['paper_raised_ms']:10.4f} {ms:12.4f} {native:12.4f} {x['device_speedup']:8.3f}x {x['paper_device_speedup']:8.3f}x  {'PASS' if valid else 'FAIL'}")
lines += ['','Correctness: every canonical output element must satisfy abs(actual-reference)<=1e-8+1e-8*abs(reference). Missing/wrong-size/nonfinite output arrays are rejected.',
'Vendor execution is checked in a separate run of the exact timed binary. Its timing is not used for the headline comparison. Runtime traces must show zero staged host/device bytes.',
'Correlation/covariance baselines are the paper’s locally authored CUDA adaptations; other computational CUDA bodies come from pinned PolyBenchGPU 5584aaa7d0be810ff5eb0b61c49fb64ecc81ba4c.',
'No historical lowered kernels or executable timings are used for current-v3 results.',
'Board clocks/power/load/temperatures and raw samples are retained per run. Fan remains automatic; fixed-fan publication protocol is therefore not fully satisfied.',
'Generic integration: CUDA ABI declarations carry polygeist.gpu_runtime; GPU residency/timing consume that promise. This does not authorize graph capture.',
'Shared implementation checks: six GPU compiler regression files plus the shared Python pipeline/ABI/import suite; exact final counts are in retained logs.',
'Artifact run: see provenance.json for source revision.']
(D/'report.txt').write_text('\n'.join(lines)+'\n')
(D/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
if summary:
 with (D/'results.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
print('Validated',sum(x['validated'] for x in summary),'of',len(summary))
