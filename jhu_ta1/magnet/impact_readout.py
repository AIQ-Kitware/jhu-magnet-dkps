"""Read saved benchmark estimates into native quality and measured-cost tables."""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import kwconf
import numpy as np

from jhu_ta1.magnet._helm_pair_data import file_hash

METHODS = {'sample_mean': 'sample_score', 'submitted_blend': 'prediction', 'raw_dkps': 'raw_prediction'}


def paired_error_summary(differences, key):
    """Paired bootstrap over query draws for one fixed target, not model transfer."""
    values = np.asarray(differences, dtype=float)
    seed = int.from_bytes(hashlib.sha256(json.dumps(key).encode()).digest()[:8], 'big')
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    return {'n_paired_draws': len(values), 'mean_abs_error_difference': float(values.mean()),
            'paired_bootstrap_lower': float(np.quantile(means, .025)),
            'paired_bootstrap_upper': float(np.quantile(means, .975)),
            'fraction_lower_abs_error': float((values < 0).mean()),
            'fraction_equal_abs_error': float((values == 0).mean()),
            'bootstrap_replicates': 2000, 'bootstrap_seed': seed,
            'uncertainty_scope': 'paired query draws conditional on one fixed target and pool'}


def wilson(successes, count):
    """95% interval over independently sampled query draws for one fixed target."""
    z = NormalDist().inv_cdf(.975)
    p, z2 = successes / count, z * z
    center = (p + z2 / (2 * count)) / (1 + z2 / count)
    radius = z * math.sqrt(p * (1 - p) / count + z2 / (4 * count * count)) / (1 + z2 / count)
    return max(0., center - radius), min(1., center + radius)


def precision_frontier(quality, required_fraction):
    """Compare only actually tested budgets on each target, without interpolation."""
    grouped = defaultdict(dict)
    for row in quality:
        grouped[row['dataset'], row['model']].setdefault(row['method'], []).append(row)
    result = []
    for (dataset, model), methods in sorted(grouped.items()):
        baseline = methods.get('sample_mean', [])
        baseline_budgets = {r['query_budget'] for r in baseline}
        for method, rows in sorted(methods.items()):
            if method == 'sample_mean':
                continue
            # Compare identical tested grids. Keep structural incompleteness explicit.
            common = baseline_budgets & {r['query_budget'] for r in rows}
            b = [r['query_budget'] for r in baseline if r['query_budget'] in common
                 and r['fraction_within_tolerance'] >= required_fraction]
            m = [r['query_budget'] for r in rows if r['query_budget'] in common
                 and r['fraction_within_tolerance'] >= required_fraction]
            matched = bool(b and m)
            result.append({'dataset': dataset, 'model': model, 'method': method,
                           'metric': rows[0]['metric'], 'tolerance': rows[0]['tolerance'],
                           'required_fraction': required_fraction,
                           'criterion': 'observed_fraction_over_query_draws',
                           'tested_budgets': sorted(common),
                           'sampling_budget': min(b) if b else None,
                           'method_budget': min(m) if m else None,
                           'query_ratio': min(b) / min(m) if matched else None,
                           'status': 'both_reached' if matched else 'sampling_not_reached' if m else
                                     'method_not_reached' if b else 'neither_reached'})
    return result


def _resolve(primary, value):
    path = Path(value)
    return path if path.is_absolute() else primary.parent / path


def recover_requests(protocol, output, hashes):
    """Retain reported HELM client durations for declared targets, without inference."""
    counts = {'selected_runs': 0, 'unavailable_runs': 0, 'observed_requests': 0}
    seen = set()
    with output.open('w') as stream:
        for source in protocol.get('sources', []):
            dataset = source['dataset']
            targets = set(protocol.get('targets_by_dataset', {}).get(dataset, protocol.get('targets', [])))
            for run in source.get('runs', []):
                if run['model'] not in targets or run['run'] in seen:
                    continue
                seen.add(run['run'])
                counts['selected_runs'] += 1
                path = Path(run['run']) / 'scenario_state.json'
                if not path.is_file():
                    counts['unavailable_runs'] += 1
                    continue
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                if run.get('scenario_sha256') and run['scenario_sha256'] != digest:
                    raise ValueError(f'Saved request source changed: {path}')
                hashes[str(path)] = digest
                for request in json.loads(raw).get('request_states', []):
                    result, instance = request.get('result') or {}, request['instance']
                    duration = result.get('request_time')
                    duration = (float(duration) if isinstance(duration, (int, float))
                                and not isinstance(duration, bool) and math.isfinite(duration) and duration >= 0 else None)
                    row = {'dataset': dataset, 'model': run['model'], 'instance_id': instance['id'],
                           'split': instance.get('split'), 'train_trial_index': request.get('train_trial_index', 0),
                           'perturbed': bool(instance.get('perturbation')),
                           'request_wall_seconds_reported': duration, 'cached_reported': result.get('cached'),
                           'success_reported': result.get('success'), 'request_datetime': result.get('request_datetime'),
                           'source_scenario': str(path)}
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    counts['observed_requests'] += 1
    return counts


def readout(source_fpath, out_fpath, *, tolerance=.05, required_fraction=.95, precompute_fpath=None,
            recover_request_times=False):
    if not math.isfinite(tolerance) or tolerance < 0 or not 0 < required_fraction <= 1:
        raise ValueError('Need a finite nonnegative tolerance and fraction in (0,1]')
    source_path, output = Path(source_fpath).resolve(), Path(out_fpath).resolve()
    source = json.loads(source_path.read_text())
    protocol = source.get('protocol', {})
    source_info = {s['dataset']: s for s in protocol.get('sources', [])}
    trial_path = _resolve(source_path, protocol.get('trial_file', source_path.with_suffix('.trials.jsonl').name))
    output.parent.mkdir(parents=True, exist_ok=True)
    groups, timings, paired = defaultdict(list), defaultdict(list), defaultdict(list)
    truths, reference_means, reference_counts = {}, defaultdict(list), defaultdict(list)
    seen, n_trials = set(), 0
    hashes = {str(source_path): file_hash(source_path)}
    batch_path = _resolve(source_path, protocol.get('measurement_files', {}).get('batches', {}).get(
        'path', source_path.with_suffix('.batches.jsonl').name))
    batches = {}
    if batch_path.is_file():
        hashes[str(batch_path)] = file_hash(batch_path)
        for line in batch_path.open():
            row = json.loads(line)
            if row['batch_id'] in batches:
                raise ValueError('Duplicate shared representation batch')
            batches[row['batch_id']] = row
    trial_digest = hashlib.sha256()
    timing_path = output.with_suffix('.trial_times.jsonl')
    with trial_path.open('rb') as stream, timing_path.open('w') as costs:
        for line in stream:
            trial_digest.update(line)
            row = json.loads(line)
            key = row['dataset'], row['model'], row['n_eval'], row['seed']
            if key in seen:
                raise ValueError('Duplicate target/query-draw trial')
            seen.add(key)
            n_trials += 1
            predictions = {method: row[field] for method, field in METHODS.items() if field in row}
            target_key = row['dataset'], row['model']
            if target_key in truths and truths[target_key] != row['truth']:
                raise ValueError('Target truth changed between query draws')
            truths[target_key] = row['truth']
            bank = source_info.get(row['dataset'], {}).get('full_scores')
            if bank is not None and row.get('reference_models'):
                # Same-query control, with the submitted blend weight and eligible bank.
                alpha = source['result']['metrics']['alpha']
                reference_mean = float(np.mean([bank[m] for m in row['reference_models']]))
                predictions['reference_mean_blend'] = alpha * row['sample_score'] + (1 - alpha) * reference_mean
                reference_means[target_key].append(reference_mean)
                reference_counts[target_key].append(len(row['reference_models']))
            for comparator in ('sample_mean', 'reference_mean_blend'):
                if comparator in predictions:
                    for method in predictions:
                        if method != comparator:
                            paired[row['dataset'], row['model'], row['n_eval'], method, comparator].append(
                                abs(predictions[method] - row['truth']) - abs(predictions[comparator] - row['truth']))
            for method, prediction in predictions.items():
                signed = float(prediction - row['truth'])
                if not math.isfinite(signed):
                    raise ValueError('Non-finite saved prediction or truth')
                groups[row['dataset'], row['model'], row['n_eval'], method].append(signed)
            elapsed = row.get('diagnostics', {}).get('elapsed')
            if elapsed:
                batch = batches.get(row.get('batch_id'), {})
                shared = batch.get('stages', {}).get('shared_representation', {})
                cost = {'dataset': row['dataset'], 'model': row['model'], 'query_budget': row['n_eval'],
                        'seed': row['seed'], 'batch_id': row.get('batch_id'),
                        'fit_wall_seconds': elapsed['wall_seconds'],
                        'fit_process_cpu_seconds': elapsed['process_cpu_seconds'],
                        'shared_representation_wall_seconds': shared.get('wall_seconds'),
                        'shared_target_fits': batch.get('num_target_fits'),
                        'reference_count': len(row['reference_models']),
                        'representation': batch.get('representation')}
                # Shared batch time is repeated as a join reference, never added per target.
                costs.write(json.dumps(cost, allow_nan=False) + '\n')
                timings[row['dataset'], row['n_eval']].append(cost['fit_wall_seconds'])
    hashes[str(trial_path)] = trial_digest.hexdigest()
    request_path = output.with_suffix('.requests.jsonl')
    if recover_request_times:
        request_coverage = recover_requests(protocol, request_path, hashes)
    else:
        request_path.write_text('')
        request_coverage = {'status': 'not_requested'}
    quality = []
    for (dataset, model, budget, method), values in sorted(groups.items()):
        signed = np.asarray(values)
        errors = abs(signed)
        successes = int((errors <= tolerance).sum())
        lower, upper = wilson(successes, len(errors))
        predictions = signed + truths[dataset, model]
        quality.append({'dataset': dataset, 'model': model, 'query_budget': budget, 'method': method,
                        'metric': source_info.get(dataset, {}).get('metric', 'unknown'),
                        'n_draws': len(errors), 'mae': float(errors.mean()), 'signed_bias': float(signed.mean()),
                        'error_q95': float(np.quantile(errors, .95)), 'tolerance': tolerance,
                        'fraction_within_tolerance': successes / len(errors),
                        'within_wilson_lower': lower, 'within_wilson_upper': upper,
                        'prediction_mean': float(predictions.mean()),
                        'prediction_std': float(predictions.std()),
                        'prediction_min': float(predictions.min()), 'prediction_max': float(predictions.max()),
                        'boundary_prediction_fraction': float(((predictions <= 0) | (predictions >= 1)).mean())})
    paired_path = output.with_suffix('.paired.jsonl')
    with paired_path.open('w') as stream:
        for key, differences in sorted(paired.items()):
            dataset, model, budget, method, comparator = key
            row = {'dataset': dataset, 'model': model, 'query_budget': budget,
                   'method': method, 'comparator': comparator, **paired_error_summary(differences, key)}
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    profiles = []
    for (dataset, model), truth in sorted(truths.items()):
        means = reference_means[dataset, model]
        counts = reference_counts[dataset, model]
        profiles.append({'dataset': dataset, 'model': model, 'full_score': truth,
                         'aligned_pool_size': source_info.get(dataset, {}).get('pool_size'),
                         'reference_score_mean': float(np.mean(means)) if means else None,
                         'reference_score_mean_min': min(means) if means else None,
                         'reference_score_mean_max': max(means) if means else None,
                         'reference_only_mae': float(np.mean(np.abs(np.asarray(means) - truth))) if means else None,
                         'reference_only_fraction_within_tolerance': float((np.abs(np.asarray(means) - truth) <= tolerance).mean()) if means else None,
                         'reference_only_target_queries': 0 if means else None,
                         'reference_count_min': min(counts) if counts else None,
                         'reference_count_max': max(counts) if counts else None,
                         'scope': 'saved finite-pool truth for evaluation; eligible reference scores only for controls'})
    quality_path, frontier_path = output.with_suffix('.quality.jsonl'), output.with_suffix('.frontier.jsonl')
    for path, rows in [(quality_path, quality), (frontier_path, precision_frontier(quality, required_fraction))]:
        with path.open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
    grouped_quality = defaultdict(list)
    for row in quality:
        grouped_quality[row['dataset'], row['query_budget'], row['method']].append(row)
    summaries = [{'dataset': d, 'query_budget': b, 'method': m, 'metric': rows[0]['metric'],
                  'n_targets': len(rows), 'n_draws': sum(r['n_draws'] for r in rows),
                  'mean_target_mae': float(np.mean([r['mae'] for r in rows])),
                  'mean_target_fraction_within': float(np.mean([r['fraction_within_tolerance'] for r in rows])),
                  'targets_reaching_fraction': sum(r['fraction_within_tolerance'] >= required_fraction for r in rows),
                  'tolerance': tolerance, 'required_fraction': required_fraction}
                 for (d, b, m), rows in sorted(grouped_quality.items())]
    measurements = source.get('measurements', {})
    stages = [{'scope': 'replay_audit', 'stage': name, **span}
              for name, span in measurements.get('stages', {}).items()]
    if precompute_fpath and str(precompute_fpath) != 'None':
        prep_path = Path(precompute_fpath).resolve()
        prep = json.loads(prep_path.read_text()).get('measurements', {})
        hashes[str(prep_path)] = file_hash(prep_path)
        stages.extend({'scope': 'reference_representation_preparation', 'stage': name, **span}
                      for name, span in prep.get('stages', {}).items())
        for name in ('embed_function', 'save_chunk', 'open_chunk', 'hash_chunk'):
            spans = [r['stages'][name] for r in prep.get('chunks', []) if name in r['stages']]
            if spans:
                stages.append({'scope': 'reference_representation_preparation', 'stage': name,
                               'wall_seconds': sum(s['wall_seconds'] for s in spans),
                               'process_cpu_seconds': sum(s['process_cpu_seconds'] for s in spans)})
    fit_summary = [{'dataset': d, 'query_budget': b, 'n_timed_fits': len(values),
                    'mean_fit_wall_seconds': float(np.mean(values)),
                    'median_fit_wall_seconds': float(np.median(values)),
                    'p95_fit_wall_seconds': float(np.quantile(values, .95))}
                   for (d, b), values in sorted(timings.items())]
    result = {'result': {'metrics': {
        'quality_summary': summaries, 'quality_by_model_fpath': str(quality_path),
        'paired_comparisons_fpath': str(paired_path), 'target_profiles': profiles,
        'precision_frontier_fpath': str(frontier_path), 'trial_times_fpath': str(timing_path),
        'request_times_fpath': str(request_path),
        'stage_timings': stages, 'fit_timings': fit_summary,
        'replay_scope': [{'observed_trials': n_trials, 'expected_trials': source.get('result', {}).get('metrics', {}).get('dkps_fits'),
                          'timed_fits': sum(len(v) for v in timings.values()),
                          'shared_batches': len(batches), 'replay_wall_seconds': measurements.get('elapsed_wall_seconds'),
                          'original_generation_seconds': None, 'original_grading_seconds': None,
                          'reference_bank_acquisition_seconds': None, 'net_savings_factor': None}]}},
        'source_sha256': hashes, 'request_source_coverage': request_coverage,
        'interpretation': {'quality': 'Per-target repeated-query estimates; Wilson intervals assume independent query draws for that fixed target. They do not bound transfer to new models.',
                           'frontier': 'Minimum tested budgets at an observed precision criterion, not a confidence guarantee, query interpolation or dollar/compute savings.',
                           'paired': 'Negative error differences favor the method. Bootstrap intervals condition on a fixed target/pool, are exploratory and unadjusted for multiple comparisons; they do not bound transfer across models.',
                           'cost': 'Replay audit and individual fit times exclude original generation/grading. Shared batch time must be counted once, not once per fit.'}}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


class ImpactReadoutCLI(kwconf.Config):
    source_fpath = kwconf.Value(None, required=True, tags=['in_path'])
    precompute_fpath = kwconf.Value(None, tags=['in_path'])
    recover_request_times = kwconf.Value(0, tags=['algo_param'])
    tolerance = kwconf.Value(.05, tags=['algo_param'])
    required_fraction = kwconf.Value(.95, tags=['algo_param'])
    readout_version = kwconf.Value(3, tags=['algo_param'])
    out_fpath = kwconf.Value('impact_readout.json', tags=['out_path', 'primary'])

    @classmethod
    def main(cls, argv=None, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        readout(config.source_fpath, config.out_fpath, tolerance=float(config.tolerance),
                required_fraction=float(config.required_fraction), precompute_fpath=config.precompute_fpath,
                recover_request_times=bool(int(config.recover_request_times)))


if __name__ == '__main__':
    ImpactReadoutCLI.main()
