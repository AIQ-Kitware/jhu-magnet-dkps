# Experimental replay measurement records

Working branch: `experiment/measurement-records`. Base: `839a99f`, the
implementation recorded by the expanded November evaluation. The preceding
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

`result.metrics` exposes scalar replay times and row counts to MAGNET. They
appear in serialized `evidence`, separate from claim-consumed symbols. Detailed
measurements remain in the primary artifact; `protocol.measurement_files`
links and hashes both JSONL files. Include those files with the primary artifact
when exporting an experimental result. Flushes occur at batch boundaries so
completed trial batches remain available after interruption; an incomplete
stream is not a completed evaluation.

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
- Full-bank acquisition, original answer generation/grading and embedding
  precomputation costs remain unknown. This evaluator reads those inputs.
  It does not reconstruct their costs from response lengths or cache hits.
- Audit runtime includes every budget and replicate. A single target estimate
  has a different cost boundary; shared representation work must be allocated
  explicitly when estimating repeated-use cost.

## Next additions

1. Reduce retained trials into raw-DKPS/blend/sample error curves, signed bias,
   error tails and within-tolerance fractions by dataset, target and budget.
   Preserve benchmark-specific score units and distinguish observed fractions
   from confidence guarantees.
2. Replay `0.8*sample_mean + 0.2*mean(eligible_reference_full_scores)` on identical
   draws. This isolates inexpensive borrowing from the reference bank. Tuning
   that weight requires separate calibration targets.
3. Instrument embedding precomputation and generation/grading acquisition.
   Then compare setup plus marginal cost at a declared precision target,
   retaining “not reached” when tested budgets are insufficient.

None of these proposed analyses changes the published historical verdict.
