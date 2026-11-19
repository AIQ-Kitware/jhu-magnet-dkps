import json
from pathlib import Path

import pytest
import yaml

from jhu_ta1.magnet.impact_readout import paired_error_summary, readout


def saved_predictions(folder, *, timed=False):
    source = folder / 'pair_coverage.json'
    rows = []
    for budget, sample in [(1, .8), (2, .7), (4, .52)]:
        for seed in (0, 1):
            row = {'dataset': 'fixture', 'model': 'heldout', 'n_eval': budget, 'seed': seed,
                   'truth': .5, 'sample_score': sample, 'prediction': sample,
                   'raw_prediction': .51, 'reference_models': ['reference'], 'batch_id': f'b{budget}-{seed}'}
            if timed:
                row['diagnostics'] = {'elapsed': {'wall_seconds': .2, 'process_cpu_seconds': .1}}
            rows.append(row)
    trials = source.with_suffix('.trials.jsonl')
    trials.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    data = {'result': {'metrics': {'dkps_fits': len(rows), 'alpha': .8}},
            'protocol': {'sources': [{'dataset': 'fixture', 'metric': 'accuracy',
                                      'full_scores': {'reference': .6}}]}}
    if timed:
        data['measurements'] = {'elapsed_wall_seconds': 2., 'stages': {'geometry': {'wall_seconds': 1.}}}
        batches = [{'batch_id': row['batch_id'], 'representation': 'one_hot', 'num_target_fits': 2,
                    'stages': {'shared_representation': {'wall_seconds': 3.}}} for row in rows]
        source.with_suffix('.batches.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in batches))
    source.write_text(json.dumps(data))
    return source


def test_precision_uses_only_tested_budgets_and_retains_unknown_costs(tmp_path):
    result = readout(saved_predictions(tmp_path), tmp_path / 'readout.json', required_fraction=1)
    metrics = result['result']['metrics']
    frontier = [json.loads(line) for line in Path(metrics['precision_frontier_fpath']).read_text().splitlines()]
    raw = next(r for r in frontier if r['method'] == 'raw_dkps')
    assert raw['sampling_budget'] == 4 and raw['method_budget'] == 1 and raw['query_ratio'] == 4
    quality = [json.loads(line) for line in Path(metrics['quality_by_model_fpath']).read_text().splitlines()]
    assert any(r['method'] == 'reference_mean_blend' for r in quality)
    assert all(r['within_wilson_lower'] < 1 for r in quality)
    assert metrics['replay_scope'][0]['net_savings_factor'] is None
    assert metrics['stage_timings'] == metrics['fit_timings'] == []
    assert Path(metrics['trial_times_fpath']).read_text() == ''
    # Neither estimator reaches this tighter observed criterion; no interpolated ratio.
    tighter = readout(tmp_path / 'pair_coverage.json', tmp_path / 'tight.json', tolerance=.001)
    frontier = [json.loads(line) for line in Path(tighter['result']['metrics']['precision_frontier_fpath']).read_text().splitlines()]
    assert all(r['query_ratio'] is None and r['status'] == 'neither_reached' for r in frontier)


def test_timing_table_keeps_shared_work_as_a_join_reference(tmp_path):
    result = readout(saved_predictions(tmp_path, timed=True), tmp_path / 'readout.json')
    metrics = result['result']['metrics']
    assert metrics['replay_scope'][0]['timed_fits'] == 6
    assert metrics['replay_scope'][0]['replay_wall_seconds'] == 2
    assert len(metrics['fit_timings']) == 3
    rows = [json.loads(line) for line in Path(metrics['trial_times_fpath']).read_text().splitlines()]
    assert all(r['fit_wall_seconds'] == .2 and r['shared_representation_wall_seconds'] == 3 for r in rows)
    assert all('total_cost' not in r for r in rows)


def test_duplicate_draws_cannot_inflate_precision(tmp_path):
    source = saved_predictions(tmp_path)
    trials = source.with_suffix('.trials.jsonl')
    trials.write_text(trials.read_text() * 2)
    with pytest.raises(ValueError, match='Duplicate'):
        readout(source, tmp_path / 'readout.json')


def test_paired_differences_use_the_same_draw_and_preserve_missing_bank(tmp_path):
    source = saved_predictions(tmp_path)
    result = readout(source, tmp_path / 'readout.json')
    metrics = result['result']['metrics']
    paired = [json.loads(line) for line in Path(metrics['paired_comparisons_fpath']).read_text().splitlines()]
    row = next(r for r in paired if r['query_budget'] == 1 and r['method'] == 'raw_dkps'
               and r['comparator'] == 'sample_mean')
    assert row['mean_abs_error_difference'] == pytest.approx(-.29)
    assert row['paired_bootstrap_upper'] < 0 and row['n_paired_draws'] == 2
    assert metrics['target_profiles'][0]['full_score'] == .5
    assert metrics['target_profiles'][0]['reference_score_mean'] == .6
    assert metrics['target_profiles'][0]['reference_only_mae'] == pytest.approx(.1)
    assert metrics['target_profiles'][0]['reference_only_fraction_within_tolerance'] == 0
    assert metrics['target_profiles'][0]['reference_only_target_queries'] == 0
    primary = json.loads(source.read_text())
    del primary['protocol']['sources'][0]['full_scores']
    source.write_text(json.dumps(primary))
    missing = readout(source, tmp_path / 'missing.json')['result']['metrics']
    assert missing['target_profiles'][0]['reference_score_mean'] is None
    assert 'reference_mean_blend' not in Path(missing['paired_comparisons_fpath']).read_text()


def test_paired_bootstrap_retains_correlation_and_is_reproducible():
    # Large individual errors can cancel in paired differences; do not resample
    # the two estimators independently or report model-transfer uncertainty.
    result = paired_error_summary([-.01] * 64, ('dataset', 'target', 8, 'method', 'control'))
    assert result == paired_error_summary([-.01] * 64, ('dataset', 'target', 8, 'method', 'control'))
    assert result['paired_bootstrap_lower'] == pytest.approx(-.01)
    assert result['paired_bootstrap_upper'] == pytest.approx(-.01)


def test_readout_revision_keeps_expensive_fit_identity(tmp_path):
    from kwdagger.pipeline import coerce_pipeline
    package = Path(__file__).resolve().parents[1] / 'jhu_ta1/magnet'
    current = yaml.safe_load((package / 'pair_coverage_supplied_pipeline.yaml').read_text())
    assert current['nodes']['impact_readout']['algo_params']['readout_version'] == 3
    old = yaml.safe_load(yaml.safe_dump(current))
    old['nodes']['impact_readout']['algo_params']['readout_version'] = 1
    pipelines = []
    for name, spec in [('old', old), ('new', current)]:
        path = tmp_path / f'{name}.yaml'
        path.write_text(yaml.safe_dump(spec))
        pipeline = coerce_pipeline(str(path))
        pipeline.configure({'pair_coverage.helm_suite_path': '/saved-suite',
                            'pair_coverage.dataset_manifest': '/selected-manifest.json'}, root_dpath=tmp_path / 'dag')
        pipelines.append(pipeline)
    assert pipelines[0].node_dict['pair_coverage'].process_id == pipelines[1].node_dict['pair_coverage'].process_id
    assert pipelines[0].node_dict['impact_readout'].process_id != pipelines[1].node_dict['impact_readout'].process_id


def test_original_request_times_retain_cache_flags_and_missing_values(tmp_path):
    source = saved_predictions(tmp_path)
    run = tmp_path / 'helm'
    run.mkdir()
    states = [{'instance': {'id': str(i), 'split': 'test'}, 'result': result}
              for i, result in enumerate([{'request_time': 1.5, 'cached': False, 'success': True},
                                           {'request_time': .2, 'cached': True},
                                           {'request_time': -1, 'success': False}, {}])]
    (run / 'scenario_state.json').write_text(json.dumps({'request_states': states}))
    primary = json.loads(source.read_text())
    primary['protocol']['targets'] = ['heldout']
    primary['protocol']['sources'][0]['runs'] = [{'run': str(run), 'model': 'heldout'}]
    source.write_text(json.dumps(primary))
    result = readout(source, tmp_path / 'readout.json', recover_request_times=True)
    rows = [json.loads(line) for line in Path(result['result']['metrics']['request_times_fpath']).read_text().splitlines()]
    assert [r['request_wall_seconds_reported'] for r in rows] == [1.5, .2, None, None]
    assert [r['cached_reported'] for r in rows] == [False, True, None, None]
    assert result['request_source_coverage']['observed_requests'] == 4


def test_native_measurement_card_and_portable_tables(tmp_path):
    from magnet.evaluation_new import NewEvaluationRecipe
    from magnet.results import export_result_bundle, load_evaluation_card, verify_result_bundle

    source = saved_predictions(tmp_path, timed=True)
    package = Path(__file__).resolve().parents[1] / 'jhu_ta1'
    card = yaml.safe_load((package / 'cards/jhu_run_predict_pair_coverage_kwdagger.yaml').read_text())
    card.pop('claim')
    card['symbols'] = {k: v for k, v in card['symbols'].items() if k.startswith('metrics.impact_readout.')}
    card['kwdagger'] = {'pipeline': str(package / 'magnet/saved_readout_pipeline.yaml'),
                       'readout_nodes': ['impact_readout'],
                       'matrix': {'impact_readout.source_fpath': str(source)}}
    recipe = tmp_path / 'recipe.yaml'
    recipe.write_text(yaml.safe_dump(card))
    evaluation = NewEvaluationRecipe(recipe, tmp_path / 'evaluation')
    evaluation.evaluate(backend='serial', print_commands=False)
    stored = load_evaluation_card(evaluation.output_path / evaluation._run_hash)
    assert stored.evaluation['state'] == 'measured'
    assert len(stored.table(node='impact_readout', expand='metrics.impact_readout.trial_times_fpath')) == 6
    assert all(v['missing'] == 0 for v in stored.symbol_catalog().values())
    exported = export_result_bundle(stored.root, tmp_path / 'portable', supporting='terminal')
    assert verify_result_bundle(exported)['valid']
    for path in (tmp_path / 'evaluation' / '_kwdagger').rglob('*.jsonl'):
        path.unlink()
    assert len(load_evaluation_card(exported).table(node='impact_readout', expand='metrics.impact_readout.trial_times_fpath')) == 6
