# Experimental replay measurement records

Working branch: `experiment/measurement-records`. Base: `839a99f`, the
implementation recorded by the expanded November evaluation, with native
measurement readout and preparation timings added November 18. The preceding
`8b81ed2` fixes the card's coverage symbol declaration. Both commits were
recovered from a shared producer checkout; they were absent from fetched refs.

Enable recording on the existing card with a MAGNET matrix override:

```yaml
matrix:
  pair_coverage.record_measurements: 1
```

Use a new `pair_coverage.run_id` when measuring a fresh execution rather than
reading an existing kwdagger result. The containerized supplied-suite runner
also accepts `--record_measurements`. Recording defaults to off; the estimator,
query sampling, reference exclusions and claim are unchanged.

## Retained evidence

| Record | Contents | Supported analysis |
|---|---|---|
| `pair_coverage.json`, `measurements` | Wall/process CPU times, stage totals, representation row counts, cache storage/access metadata, package versions, source hashes and CPU/thread context | Replay overhead and configuration comparisons |
| `pair_coverage.json`, `protocol.sources[].full_scores` | Full aligned-pool scores for each model, linked to existing source hashes | Reference-informed controls on the same eligible bank |
| `pair_coverage.trials.jsonl` | Existing predictions/truth/query/reference IDs; adds sampled target scores, metric, unclipped OLS prediction, per-fit stage times and batch ID | Paired error, bias, tails, clipping, tolerance rates and baseline replay |
| `pair_coverage.batches.jsonl` | One record per shared representation batch, its query/model IDs, representation route, array shapes/bytes, target-fit count and timing | Shared work without counting the batch once per target |

Full scores include audit-only target truth. Prediction still receives only
the selected target grades and eligible reference scores. Trial reference IDs
determine which full scores a replayed control may use.

The pipeline adds an independent CPU `impact_readout` node;
`kwdagger.readout_nodes` publishes its measurements in `evaluation.json`.
Ten symbols have the prefix `metrics.impact_readout.`:
`quality_summary`, `quality_by_model_fpath`, `precision_frontier_fpath`,
`stage_timings`, `fit_timings`, `trial_times_fpath`, `replay_scope` and
`request_times_fpath`, `paired_comparisons_fpath` and `target_profiles`.
Detailed quality, precision-budget, paired-error and timing tables
are file-backed JSONL. Native exports with `supporting=terminal` include them.
The submitted claim and estimator remain unchanged. Saved-readout recipes
can omit a claim; new reports lead with measurements.

Readout version 2 adds paired error differences against sample mean and the
same-weight reference-mean blend. Negative differences favor the method.
The 2,000-draw percentile bootstrap resamples differences from identical query
draws for one fixed target/pool, with a recorded deterministic seed. These are
exploratory intervals, unadjusted for multiple comparisons; common queries and
related models prevent treating all target rows as independent replications.
`target_profiles` exposes full target scores and eligible reference-bank means.
Quality rows also retain prediction ranges, variation and boundary fractions.
Version 3 reports the eligible bank mean's error and within-tolerance fraction
using **zero target queries**, alongside its score. This control distinguishes
raw prediction gains from proximity to a reusable bank mean. Bank preparation
is still paid, and random reference-bank selection can contribute variability.
These additions recompute only the CPU readout, leaving estimator/acquisition
identities unchanged. Missing historical reference scores remain unavailable.

```bash
magnet results symbols RECORD
magnet results table RECORD --node=impact_readout \
  --expand=metrics.impact_readout.fit_timings
magnet results table RECORD --node=impact_readout \
  --expand=metrics.impact_readout.quality_by_model_fpath \
  --where='{query_budget: 8}' --group_by=dataset,method --aggregate='{mae: mean}'
```

`saved_readout_pipeline.yaml` accepts an existing `pair_coverage.json` as
`impact_readout.source_fpath`, with no materialization, embedding or fitting
nodes. Fresh and saved paths publish the same symbols. Readout tolerance
and required observed fraction default to 0.05 native score units and 0.95.
Five accuracy points and five normalized BLEU points are different scientific
requirements; select datasets/metrics explicitly.

Per-target quality includes raw DKPS, the submitted blend and sample mean:
signed bias, MAE, tails and observed within-tolerance rates. Sources with
retained eligible reference scores also support the same-weight sample/
reference-mean control on identical draws. Older outputs without that bank
do not invent it. Wilson intervals assume independent query draws conditional
on one fixed target; they do not establish transfer reliability. The precision
frontier selects minimum tested budgets at the observed-fraction criterion,
retaining null ratios/status when either method does not reach it. It does
not interpolate budgets or label query ratios compute savings.

`protocol.measurement_files` still links/hashes trial and batch streams.
Flushes preserve completed batches after interruption; readouts retain
observed/expected trial counts.

## Measurement boundaries

- Stages separate saved-panel loading, exported-cache opening/validation,
  shared representation work, DKPS geometry, OLS fitting/prediction and
  sample-mean/blend computation. Stage sums omit loop, import and output
  overhead. Elapsed time covers the evaluation call, including recording
  overhead, but excludes subsequent provenance collection and final file writes.
- CPU time belongs to the evaluator process, including its threads; it is not
  GPU time or subprocess accounting. A first invocation may include library
  warmup. Keep cold and warm conditions explicit.
- `one_hot`, `exported_cache` and `embedding_helper` distinguish representation
  routes. The helper may have its own cache; requested rows do not establish
  newly encoded rows or API calls. Array bytes describe returned arrays, not
  peak process memory. Exported-cache mmap hits/misses include validation;
  chunk storage bytes and mmap counts do not measure physical disk reads.
- Full-bank acquisition, original generation/grading and embedding preparation
  are not measured by the replay evaluator. The optional observations below
  supply separate evidence; absent costs stay unknown.
- Audit runtime includes every budget and replicate. A single target estimate
  has a different cost boundary; shared representation work must be allocated
  explicitly when estimating repeated-use cost.

Predictor invocation timing includes imports/setup, geometry, regression and
blending, excluding shared representation. First calls may include warmup.
Audit elapsed time also includes loading, recording and loop overhead; adding
invocation totals to their component spans would double-count work.

## Representation preparation

Add `--measurement_fpath=/path/preparation.json` to the existing
`python -m jhu_ta1.magnet.precompute_embeddings` command to record stage times,
requested rows, logical bytes and chunk reuse. A sibling `.chunks.jsonl`
flushes completed chunks. Encoder initialization, embedding-function calls,
chunk opening/hashing/writing and panel/index work are separate. The helper
may have its own cache: a function call is not proof of new device computation.
Existing chunks are reused, including with `--force`; instrumentation changes
neither chunk names nor index schema. Pass the JSON to
`impact_readout.precompute_fpath` to publish preparation stages.

## Original request observations

Set `impact_readout.recover_request_times: 1` to recover HELM request metadata
for declared targets. `request_times_fpath` retains reported durations, cache/
status flags, timestamps and source hashes. Missing files are counted in
`request_source_coverage`; missing/invalid durations remain null. This does not
generate requests or measure judge/grading work. Rows describe source runs,
including split, train-trial and perturbation flags; they are not automatically
restricted to the aligned scoring pool. Filter/join item IDs before matched
cost comparisons. Cached durations are historical reported facts, not free
generation or fresh execution. Summed durations may overlap and are not job
elapsed time or GPU-active time.

## Remaining operational evidence

Measure generation/grading where original metadata is absent, and compare
setup plus marginal cost at a declared precision target. Retain “not reached”
when tested budgets are insufficient. Larger-budget, matched-judge and
deployment studies are separate workloads. No savings factor or monetary
projection is inferred from replay wall time.

None of these proposed analyses changes the published historical verdict.
