import csv, json, sys
from pathlib import Path

root = Path('/home/arjaiswal/Polygeist')
sys.path.insert(0, str(root / 'scripts/correctness'))
from aten_frozen_runtime_campaign import save
base = root / 'issues/aten_paper29_followup_20261112'
oldbase = root / 'issues/aten_paper86_correctness_20261112'
paper_path = root / 'issues/aten_c_kernels/paper_analysis/aten_gpu_nonpointwise_115.csv'
paper = {r['kernel']: r for r in csv.DictReader(paper_path.open())}
initial = json.loads((oldbase / 'runtime-current/runtime-results.json').read_text())
rows = {}
for item in initial:
    name = item['kernel']
    path = Path(item['runtime_result_source'])
    rows[name] = dict(kernel=name, complete_match=True,
        correctness_passed=item['correctness_passed'], correctness=item['correctness'],
        evidence=str(path), initial_evidence=str(path), timing=item.get('timing', {}),
        shape_record=item.get('shape_record'), paper_shape_exact=item['paper_shape_exact'],
        binary_sha256=item['timing_binary_sha256'], baseline_scope=item.get('baseline_scope'),
        baseline_caveat=item.get('baseline_caveat'), cudnn_plan_cache=False)

def select(directory, names):
    for name in names:
        path = directory / name / 'gpu-result.json'
        result = json.loads(path.read_text())
        built = json.loads((directory / name / 'build.json').read_text())
        previous = rows.get(name, {})
        row = dict(kernel=name, complete_match=True,
            correctness_passed=result['fully_validated'], correctness=result,
            evidence=str(path), initial_evidence=previous.get('initial_evidence', str(path)),
            superseded_evidence=previous.get('evidence'), timing=result.get('timing', {}),
            shape_record=built.get('shape_record'),
            paper_shape_exact=not (built.get('shape_record') or {}).get('correction', False),
            binary_sha256=result['binary_sha256'], baseline_scope=result.get('scope'),
            cudnn_plan_cache=result.get('cudnn_plan_cache', False))
        diagnostic = directory / name / 'diagnostic-timing.json'
        if diagnostic.exists():
            row['diagnostic_timing_evidence'] = str(diagnostic)
        comparison_path = directory / name / 'paper-comparison.json'
        if comparison_path.exists():
            comparison = json.loads(comparison_path.read_text())
            row['infrastructure_failure'] = comparison.get('infrastructure_error', False)
            row['baseline_scope'] = comparison.get('baseline_scope', row['baseline_scope'])
            row['workspace_cache'] = comparison.get('workspace_cache', False)
            if comparison.get('native_inputs_match_paper') is False:
                row['native_inputs_match_paper'] = False
                row['paper_speedup_comparable'] = False
                row['paper_native_input_note'] = comparison.get('paper_native_scope_note')
            if 'native_baseline_matches_paper_scope' in comparison:
                row['native_baseline_matches_paper_scope'] = comparison['native_baseline_matches_paper_scope']
                row['paper_native_scope_note'] = comparison.get('paper_native_scope_note')
        rows[name] = row

select(base / 'gpu-additions', ['aten_outer', 'aten_conv_transpose3d_grad_weight_cpu',
                              'aten_slow_conv3d_backward_weight_cpu'])
select(base / 'gpu-backward-data', ['aten_conv_transpose3d_cpu', 'aten_slow_conv3d_backward_input_cpu'])
select(base / 'gpu-backward-data-next', ['aten_conv_transpose2d', 'aten_conv_tbc_backward_cpu'])
select(base / 'gpu-whole-references', ['aten_sum'])
select(base / 'gpu-row-scan-final', ['aten_cumprod_cpu'])
# The controlled cached variants are current steady-state evidence, while the
# original uncached campaign remains independently accessible in its directory.
select(base / 'transpose-cache-diagnostic/cache-on', ['aten_conv_transpose2d', 'aten_conv_transpose3d_cpu'])
newcache = base / 'gpu-forward-cache-five'
for name in ['aten_conv2d', 'aten_dilated_convolution_cpu', 'aten_conv_transpose3d_backward_cpu']:
    if (newcache / name / 'gpu-result.json').exists():
        select(newcache, [name])
for name in ['aten_convert_csr_to_coo_cpu', 'aten_sparse_matmul_csr_to_coo_cpu']:
    directory = base / 'gpu-csr-expand'
    if (directory / name / 'gpu-result.json').exists():
        select(directory, [name])
    repaired = base / 'gpu-csr-expand-enabled'
    if (repaired / name / 'gpu-result.json').exists():
        select(repaired, [name])
for name in ['aten_sparse_addmv_csr_cpu', 'aten_sparse_csr_addmm_cpu', 'aten_sparse_matmul_cpu']:
    directory = base / 'gpu-csr-arithmetic'
    if (directory / name / 'gpu-result.json').exists():
        select(directory, [name])
for name in ['aten_sparse_addmm_cpu', 'aten_hspmm_cpu', 'aten_sspaddmm_cpu']:
    directory = base / 'gpu-coo-arithmetic'
    if (directory / name / 'gpu-result.json').exists():
        select(directory, [name])
    followup = base / 'coo-native-attribution' / name / 'paper-comparison.json'
    if followup.exists() and name in rows:
        rows[name]['native_attribution_followup'] = dict(
            evidence=str(followup), result=json.loads(followup.read_text()))
histogram = base / 'gpu-histogram'
if (histogram / 'aten_embedding_bag_counts_cpu/gpu-result.json').exists():
    select(histogram, ['aten_embedding_bag_counts_cpu'])

paper13 = root / 'issues/aten_paper13_followup_20261112'
for folder, name in [('gpu-nested-all', 'aten_nested_all_cpu'), ('gpu-conv1d', 'aten_conv1d')]:
    directory = paper13 / folder
    if (directory / name / 'gpu-result.json').exists():
        select(directory, [name])
        comparison = json.loads((directory / name / 'paper-comparison.json').read_text())
        if comparison.get('native_inputs_match_paper') is False:
            rows[name]['native_inputs_match_paper'] = False
            rows[name]['paper_speedup_comparable'] = False
            rows[name]['paper_native_input_note'] = comparison.get('paper_native_scope_note')

cached_conv1d = paper13 / 'gpu-conv1d-cache-relink'
if (cached_conv1d / 'aten_conv1d/gpu-result.json').exists():
    select(cached_conv1d, ['aten_conv1d'])

descriptor_copy = paper13 / 'descriptor-copy-optimized'
for name in ['aten_adaptive_avg_pool2d', 'aten_pixel_shuffle_cpu_backend', 'aten_batch_norm']:
    if (descriptor_copy / name / 'gpu-result.json').exists():
        select(descriptor_copy, [name])

sparse_depthwise = paper13 / 'gpu-sparse-depthwise-stable'
for name in ['aten_sampled_addmm_sparse_csr_cpu', 'aten_depthwise_conv3x3_cpu']:
    if (sparse_depthwise / name / 'gpu-result.json').exists():
        select(sparse_depthwise, [name])

workspace_cache = paper13 / 'workspace-cache'
for name in ['aten_adaptive_avg_pool2d', 'aten_pixel_shuffle_cpu_backend', 'aten_batch_norm']:
    if (workspace_cache / name / 'paper-comparison.json').exists():
        select(workspace_cache, [name])

depthwise_cache = paper13 / 'gpu-depthwise-cache-relink'
if (depthwise_cache / 'aten_depthwise_conv3x3_cpu/gpu-result.json').exists():
    select(depthwise_cache, ['aten_depthwise_conv3x3_cpu'])

descriptor_unrolled = paper13 / 'descriptor-copy-unrolled'
for name in ['aten_adaptive_avg_pool2d', 'aten_pixel_shuffle_cpu_backend', 'aten_batch_norm']:
    if (descriptor_unrolled / name / 'paper-comparison.json').exists():
        select(descriptor_unrolled, [name])

descriptor_tiled = paper13 / 'descriptor-copy-tiled'
if (descriptor_tiled / 'aten_pixel_shuffle_cpu_backend/paper-comparison.json').exists():
    select(descriptor_tiled, ['aten_pixel_shuffle_cpu_backend'])

assert len(paper) == 115
complete_count = len(rows)
strict_count = sum(r['correctness_passed'] for r in rows.values())
for name, historical in paper.items():
    row = rows.setdefault(name, dict(kernel=name, complete_match=False,
        correctness_passed=False, status='matching_pending', timing={}))
    row['paper'] = historical
    row['status'] = ('strict_pass' if row['correctness_passed'] else 'strict_comparison_failed') if row['complete_match'] else 'matching_pending'
    if row.get('infrastructure_failure'):
        row['status'] = 'infrastructure_failed_before_validation'
    timing = row['timing']
    if timing.get('passed'):
        count = sum(s['path'] == 'raised' for s in timing['samples'])
        paper_count = 20 if '20' in historical['timing_protocol'] else 5
        stat = 'minimum_ms' if count == paper_count else 'minimum_first_five_ms'
        current = timing[stat]
        row['comparison'] = dict(selected_statistic=stat, actual_samples=count,
            selected_samples=count if stat == 'minimum_ms' else 5,
            paper_samples=paper_count,
            statistic_matches_paper=(count if stat == 'minimum_ms' else 5) == paper_count,
            raised_ms=current['raised'], native_ms=current['native_aten_cuda'],
            raised_over_paper=current['raised'] * 1000 / float(historical['raised_us']),
            current_native_over_raised=current['native_aten_cuda'] / current['raised'],
            validated=row['correctness_passed'])

failures = [dict(kernel='aten_sparse_norm_cpu', cause='Long sequential FP32 accumulation differs from the parallel library accumulation; both GPU results are closer to the independent higher-precision reference.',
    observed=dict(sequential_c=1141.379150390625, raised=1141.093017578125,
                  native=1141.0928955078125, higher_precision=1141.092951463651),
    evidence=str(oldbase / 'sparse-norm-rounding.json'),
    proposed_fix='If the unchanged sequential-C gate remains required, supply and import an order-preserving reduction reference/backend. This will sacrifice parallel reduction performance. A higher-precision oracle is useful separate evidence, not a reason to relabel this strict failure.'),
 dict(kernel='aten_outer', cause='The extracted C initializes output to positive zero and adds the product. Native at::outer_out directly multiplies, preserving negative zero. Under the existing sign-sensitive gate these differ even though all finite numerical values agree.',
    observed=dict(values=4194304, signed_zero_disagreements=41120, raised_vs_c_bad=0, maximum_absolute_error=0),
    evidence=str(base / 'gpu-additions/aten_outer/gpu-result.json'),
    proposed_fix='Keep the direct native outer result as a failed semantic comparison. To match the extracted C exactly, a separate actual-ATen baseline must implement the same zero-initialize then addition, with that changed baseline scope explicitly documented. Do not normalize zeros or silently change the source/gate.'),
 dict(kernel='aten_conv_transpose3d_grad_weight_cpu', cause='Raised and native cuDNN outputs agree with each other, but differ from the long sequential FP32 C reduction. Accumulation order is the leading explanation; saved evidence does not yet include an independent higher-precision oracle for each failing output.',
    observed=dict(values=3456, raised_vs_native_bad=0, c_disagreements=94, maximum_absolute_error=0.0025100708),
    evidence=str(base / 'gpu-additions/aten_conv_transpose3d_grad_weight_cpu/gpu-result.json'),
    proposed_fix='First quantify each failing result against a higher-precision oracle without changing acceptance. To satisfy the existing sequential-C gate, investigate supported cuDNN algorithms with the required arithmetic ordering or a generic order-preserving reduction backend; deterministic alone does not imply the same order.'),
 dict(kernel='aten_slow_conv3d_backward_weight_cpu', cause='Raised and native cuDNN outputs agree, but differ from sequential FP32 C accumulation; same numerical-contract issue as the transpose weight gradient.',
    observed=dict(values=3456, raised_vs_native_bad=0, c_disagreements=10, maximum_absolute_error=0.000274658203),
    evidence=str(base / 'gpu-additions/aten_slow_conv3d_backward_weight_cpu/gpu-result.json'),
    proposed_fix='Use a higher-precision oracle to diagnose the discrepancies, while preserving this failed gate. Strict sequential-C reproduction requires an algorithm/reference that preserves the required accumulation behavior, not a looser tolerance.')]
oracle = base / 'weight-gradient-oracle/exact-oracle-comparison.json'
if oracle.exists():
    findings = json.loads(oracle.read_text())['cases']
    for failure in failures:
        if failure['kernel'] in findings:
            name = failure['kernel']
            data = findings[name]['modes']['1']
            failure['cause'] = ('Raised/native outputs are bit-identical, but differ from sequential C. Exact-integer accumulation diagnoses the rounding error without changing the gate. All supported legacy FP32/FMA cuDNN backward-filter algorithms were tested and none passed the existing C gate.')
            failure['exact_oracle_evidence'] = str(oracle)
            failure['supported_algorithm_evidence'] = str(base / 'weight-gradient-oracle/algorithm-probe/comparison.json')
            failure['exact_oracle_failing_subset'] = data['strict_C_failing_outputs']
            failure['exact_sum_C_tolerance_failures'] = data['exact_sum_C_tolerance_failures']
            failure['proposed_fix'] = 'Keep the strict failure. No tested supported FP32/FMA algorithm meets the existing C/native contract. Exact sequential-C reproduction requires an order-preserving fallback; changing to a mathematical-error or native-equivalence contract requires an explicit methodological decision, not a silent tolerance change.'

for failure in failures:
    failure['strict_status'] = 'failed_preserved'
    failure['memory_guards_and_inputs'] = 'pass'
save(base / 'runtime-failure-diagnoses.json', failures)

valid = [r for r in rows.values() if r['correctness_passed'] and r.get('paper_shape_exact') and r.get('comparison', {}).get('validated')]
summary = dict(paper_cases=115, complete_matches=complete_count, strict_passes=strict_count,
    strict_failures=[r['kernel'] for r in rows.values() if r['complete_match'] and not r['correctness_passed'] and not r.get('infrastructure_failure')],
    infrastructure_failures=[r['kernel'] for r in rows.values() if r.get('infrastructure_failure')],
    known_native_input_differences=[r['kernel'] for r in rows.values() if r.get('native_inputs_match_paper') is False],
    known_native_baseline_scope_differences=[r['kernel'] for r in rows.values()
        if r.get('native_baseline_matches_paper_scope') is False],
    isolated_native_attribution_followups=[dict(kernel=r['kernel'],
        fully_validated=r['native_attribution_followup']['result']['fully_validated'],
        paper_speedup_comparable=r['native_attribution_followup']['result']['paper_speedup_comparable'],
        evidence=r['native_attribution_followup']['evidence'])
        for r in rows.values() if 'native_attribution_followup' in r],
    matching_pending=[r['kernel'] for r in rows.values() if not r['complete_match']],
    exact_shape_validated_timings=len(valid),
    exact_shape_and_statistic_validated_timings=sum(r['comparison']['statistic_matches_paper'] for r in valid),
    within_five_percent_or_faster=sum(r['comparison']['raised_over_paper'] <= 1.05 for r in valid),
    slower_five_to_twenty_percent=sum(1.05 < r['comparison']['raised_over_paper'] <= 1.2 for r in valid),
    slower_over_twenty_percent=sum(r['comparison']['raised_over_paper'] > 1.2 for r in valid),
    scope='Exact recorded paper inputs, except the explicitly recorded conv2d-columns K correction; strict failures remain failed. Current cached steady-state timings are selected where available, without altering the original uncached campaigns. Historical paper cache state is unknown. Each row states its measured and paper sample counts.',
    future_protocol='Five warmups then five alternating synchronized pairs. Completed twenty-pair evidence retained.',
    source_paper=str(paper_path))
save(base / 'runtime-validation-ledger.json', dict(summary=summary, cases=[rows[n] for n in sorted(rows)]))
save(base / 'runtime-validation-status.json', summary)
print(json.dumps({k:v for k,v in summary.items() if k not in ('matching_pending','scope','strict_failures','source_paper')}))
print('UNMATCHED', ', '.join(summary['matching_pending']))
