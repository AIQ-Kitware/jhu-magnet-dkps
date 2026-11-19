import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

from jhu_ta1.magnet._helm_pair_data import load_panels, load_panels_by_dataset
from jhu_ta1.magnet.pair_coverage import (
    evaluate,
    predict,
    summarize_errors,
    summarize_dataset_errors,
)


class LivePairTests(unittest.TestCase):
    def run_claim(self, values):
        root = Path(__file__).resolve().parents[1]
        card = yaml.safe_load(
            (
                root / "jhu_ta1/cards/jhu_run_predict_pair_coverage_kwdagger.yaml"
            ).read_text()
        )
        exec(
            card["claim"]["python"],
            dict(
                coverage_threshold=0.95,
                metrics=SimpleNamespace(pair_coverage=SimpleNamespace(**values)),
            ),
        )

    def test_pooled_claim_can_pass_with_a_failing_budget(self):
        errors = np.zeros((1, 4, 3, 20))
        errors[:, 0, :, :3] = 0.8
        result = summarize_errors(
            errors,
            np.full_like(errors, 0.1),
            ["dataset"],
            [f"model{i}" for i in range(20)],
            [1, 2, 4, 8],
        )
        metrics = result["result"]["metrics"]
        self.assertEqual(metrics["num_settings"], 80)
        self.assertEqual(metrics["settings_improved"], 77)
        self.assertEqual(metrics["setting_improvement_fraction"], 77 / 80)
        self.assertEqual(result["per_budget"][0]["pair_improvement_fraction"], 17 / 20)
        self.assertEqual(len(result["per_setting"]), 80)
        # Averaging errors across budgets would leave only 17/20 models
        # improved. Counting combinations must still pass the card.
        metrics.update(alpha=0.8, n_components_cmds=8, dkps_fits=240)
        self.run_claim(metrics)

    def test_invalid_budgets_fail_before_loading(self):
        for budgets in ([], [1, 1], [0], [-1]):
            with self.subTest(budgets=budgets):
                with self.assertRaisesRegex(ValueError, "distinct positive integers"):
                    evaluate("unused", {}, query_budgets=budgets)

    def test_dataset_specific_reduction_counts_settings_instead_of_panel_means(self):
        errors = [np.zeros((1, 3, 20)), np.full((1, 3, 1), 0.3)]
        sample = [np.full((1, 3, 20), 0.1), np.full((1, 3, 1), 0.1)]
        targets = {"a": [f"m{i}" for i in range(20)], "b": ["m0"]}
        result = summarize_dataset_errors(errors, sample, ["a", "b"], targets, [1])
        metrics = result["result"]["metrics"]
        self.assertEqual(metrics["num_pairs"], 21)
        self.assertEqual(metrics["num_settings"], 21)
        self.assertEqual(metrics["num_models"], 20)
        self.assertEqual(metrics["setting_improvement_fraction"], 20 / 21)
        self.assertEqual(result["per_budget"][0]["num_pairs"], 21)
        metrics.update(alpha=0.8, n_components_cmds=8, dkps_fits=63)
        self.run_claim(metrics)
        with self.assertRaisesRegex(ValueError, "same replicate count"):
            summarize_dataset_errors([errors[0], errors[1][:, :2]],
                                     [sample[0], sample[1][:, :2]], ["a", "b"], targets, [1])

    def test_common_grid_reduction_is_unchanged(self):
        rng = np.random.default_rng(42)
        errors = rng.random((2, 4, 3, 5))
        sample = rng.random(errors.shape)
        models = [f"m{i}" for i in range(5)]
        old = summarize_errors(errors, sample, ["a", "b"], models, [1, 2, 4, 8])
        new = summarize_dataset_errors(errors, sample, ["a", "b"],
                                       {"a": models, "b": models}, [1, 2, 4, 8])
        for key in old:
            self.assertEqual(old[key], new[key])

    def fixture(self, root):
        datasets = ["med_qa", "legalbench:subset=fixture"]
        for d, dataset in enumerate(datasets):
            for m in range(12):
                model = f"family{m // 4}/model{m}"
                directory = root / f"{dataset},model={model.replace('/', '_')}"
                directory.mkdir()
                requests, scores = [], []
                for i in range(8):
                    correct = i % 2
                    response = (i + m) % 2
                    labels = ["A", "B"] if d == 0 else ["yes", "no"]
                    instance = dict(
                        id=f"id{i}",
                        split="test",
                        input={"text": f"question {i}"},
                        references=[
                            {"output": {"text": labels[correct]}, "tags": ["correct"]}
                        ],
                    )
                    requests.append(
                        dict(
                            instance=instance,
                            train_trial_index=0,
                            result={"completions": [{"text": labels[response]}]},
                        )
                    )
                    score = float(response == correct)
                    stats = (
                        [
                            {
                                "name": {"name": "accuracy", "split": "test"},
                                "mean": score,
                                "count": 1,
                            }
                        ]
                        if d == 0
                        else {"accuracy": score}
                    )
                    scores.append(
                        dict(instance_id=f"id{i}", train_trial_index=0, stats=stats)
                    )
                (directory / "scenario_state.json").write_text(
                    json.dumps(
                        dict(adapter_spec={"model": model}, request_states=requests)
                    )
                )
                name = (
                    "per_instance_stats.json" if d == 0 else "display_predictions.json"
                )
                (directory / name).write_text(json.dumps(scores))
        return dict(
            datasets=[dict(dataset=name, metric="accuracy") for name in datasets]
        )

    def test_reduction_averages_only_replicates_and_excludes_ties(self):
        errors = np.array([np.zeros((1, 3, 2)), np.full((1, 3, 2), 0.3)])
        baseline = np.array([np.full((1, 3, 2), 0.4), np.full((1, 3, 2), 0.1)])
        result = summarize_errors(errors, baseline, ["a", "b"], ["m1", "m2"], [1])
        metrics = result["result"]["metrics"]
        self.assertEqual(metrics["num_settings"], 4)
        self.assertEqual(metrics["setting_improvement_fraction"], 0.5)
        self.assertEqual(
            [row["expected_gain"] > 0 for row in result["per_setting"]],
            [True, True, False, False],
        )
        tied = summarize_errors(baseline, baseline, ["a", "b"], ["m1", "m2"], [1])
        self.assertEqual(tied["result"]["metrics"]["settings_improved"], 0)

    def test_incomplete_or_nonfinite_errors_are_rejected(self):
        errors = np.zeros((1, 2, 3, 4))
        for invalid in (errors[:, :1], np.full_like(errors, np.nan)):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                summarize_errors(invalid, errors, ["a"], list("abcd"), [1, 2])
        with self.assertRaisesRegex(ValueError, "Invalid.*panel"):
            summarize_errors(errors, errors, ["a"], list("abcd"), [1])

    def test_loader_aligns_real_helm_formats_and_rejects_missing_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            panels, targets = load_panels(root, manifest)
            self.assertEqual([p.pool_size for p in panels], [8, 8])
            self.assertEqual(len(targets), 12)
            path = next(root.glob("med_qa*/per_instance_stats.json"))
            rows = json.loads(path.read_text())
            rows.pop()
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "Missing accuracy"):
                load_panels(root, manifest)

    def test_max_items_subsamples_the_aligned_pool_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            for entry in manifest["datasets"]:
                entry.update(max_items=3, item_seed=7)
            first, _ = load_panels(root, manifest)
            again, _ = load_panels(root, manifest)
            self.assertEqual([p.pool_size for p in first], [3, 3])
            self.assertEqual([p.item_ids for p in first], [p.item_ids for p in again])
            for panel in first:
                self.assertEqual(panel.item_ids, sorted(panel.item_ids))
                for model in panel.models:
                    self.assertEqual(len(panel.responses[model]), 3)
                    self.assertEqual(len(panel.scores[model]), 3)
                self.assertEqual(len(panel.references), 3)
                # The record of what each model actually held is the full pool.
                self.assertEqual(set(panel.original_pool_sizes.values()), {8})
            # A different seed picks a different subset; the full pool is the cap.
            for entry in manifest["datasets"]:
                entry["item_seed"] = 8
            other, _ = load_panels(root, manifest)
            self.assertNotEqual(
                [p.item_ids for p in first], [p.item_ids for p in other]
            )
            for entry in manifest["datasets"]:
                entry["max_items"] = 100
            capped, _ = load_panels(root, manifest)
            self.assertEqual([p.pool_size for p in capped], [8, 8])
            for entry in manifest["datasets"]:
                entry["max_items"] = 0
            with self.assertRaisesRegex(ValueError, "max_items must be positive"):
                load_panels(root, manifest)

    def test_loader_rejects_unaligned_item_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            path = next(root.glob("med_qa*/scenario_state.json"))
            data = json.loads(path.read_text())
            data["request_states"][0]["instance"]["input"]["text"] = (
                "different question"
            )
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "content differs"):
                load_panels(root, manifest)

    def test_dataset_specific_targets_execute_only_declared_combinations(self):
        import shutil

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            # An additional target has outputs on only the first dataset.
            shutil.rmtree(root / "legalbench:subset=fixture,model=family0_model0")
            panels, common = load_panels(root, manifest)
            self.assertNotIn("family0/model0", common)
            manifest["target_models_by_dataset"] = {
                "med_qa": ["family0/model0", "family1/model4"],
                "legalbench:subset=fixture": ["family1/model4"],
            }
            manifest["target_groups"] = {
                "existing": ["family1/model4"], "additional": ["family0/model0"],
            }
            stream = io.StringIO()
            result = evaluate(root, manifest, query_budgets=[1, 2],
                              num_replicates=2, dimensions=2, trial_stream=stream)
            metrics = result["result"]["metrics"]
            self.assertEqual(metrics["num_models"], 2)
            self.assertEqual(metrics["num_pairs"], 3)
            self.assertEqual(metrics["num_settings"], 6)
            self.assertEqual(metrics["dkps_fits"], 12)
            rows = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertEqual(len(rows), 12)
            self.assertEqual(sum(r["model"] == "family0/model0" for r in rows), 4)
            self.assertFalse(any(r["model"] == "family0/model0" and r["dataset"].startswith("legalbench")
                                 for r in rows))
            for trial in rows:
                self.assertTrue(all(m.split("/")[0] != trial["model"].split("/")[0]
                                    for m in trial["reference_models"]))
            self.assertEqual([r["num_settings"] for r in result["per_target_group"]], [4, 2])
            self.assertEqual(result["protocol"]["targets_by_dataset"], manifest["target_models_by_dataset"])
            # A requested missing target must be rejected, rather than silently
            # disappearing into the denominator of the improvement fraction.
            manifest["target_models_by_dataset"]["legalbench:subset=fixture"].append("family0/model0")
            with self.assertRaisesRegex(ValueError, "present on this dataset"):
                load_panels_by_dataset(root, manifest)
            del manifest["target_models_by_dataset"]["legalbench:subset=fixture"]
            with self.assertRaisesRegex(ValueError, "exactly one target roster"):
                load_panels_by_dataset(root, manifest)

    def test_live_evaluation_fits_dkps_and_holds_out_families(self):
        from dkps.dkps import DataKernelPerspectiveSpace

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            seen = []

            def inspect_predict(
                embeddings, reference_scores, target, target_query_scores, **kw
            ):
                self.assertEqual(len(target_query_scores), 2)
                self.assertTrue(
                    all(
                        m.split("/")[0] != target.split("/")[0]
                        for m in reference_scores
                    )
                )
                self.assertEqual(len(reference_scores), 8)
                seen.append(target)
                return predict(
                    embeddings, reference_scores, target, target_query_scores, **kw
                )

            original = DataKernelPerspectiveSpace.fit_transform
            fits = []

            def fit(instance, data, **kw):
                fits.append(tuple(data))
                self.assertEqual(len(data), 9)
                return original(instance, data, **kw)

            with (
                patch(
                    "jhu_ta1.magnet.pair_coverage.predict", side_effect=inspect_predict
                ),
                patch.object(DataKernelPerspectiveSpace, "fit_transform", fit),
            ):
                result = evaluate(
                    root, manifest, query_budgets=[2], num_replicates=3, dimensions=2
                )
            self.assertEqual(len(seen), 72)
            self.assertEqual(len(fits), 72)
            self.assertEqual(result["result"]["metrics"]["dkps_fits"], 72)
            self.assertEqual(result["result"]["metrics"]["embedding_batches"], 6)
            self.assertEqual(len(result["per_setting"]), 24)

    def test_unqueried_target_scores_cannot_change_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            manifest["datasets"] = manifest["datasets"][:1]
            manifest["target_models"] = ["family0/model0"]
            first, second = io.StringIO(), io.StringIO()
            evaluate(
                root,
                manifest,
                query_budgets=[1],
                num_replicates=2,
                dimensions=2,
                trial_stream=first,
            )
            trials = [json.loads(line) for line in first.getvalue().splitlines()]
            queried = {json.loads(q)[1] for row in trials for q in row["query_ids"]}
            path = root / "med_qa,model=family0_model0/per_instance_stats.json"
            data = json.loads(path.read_text())
            for row in data:
                if row["instance_id"] not in queried:
                    row["stats"][0]["mean"] = 0.0
            path.write_text(json.dumps(data))
            evaluate(
                root,
                manifest,
                query_budgets=[1],
                num_replicates=2,
                dimensions=2,
                trial_stream=second,
            )
            after = [json.loads(line) for line in second.getvalue().splitlines()]
            for a, b in zip(trials, after):
                self.assertEqual(a["prediction"], b["prediction"])
                self.assertEqual(a["raw_prediction"], b["raw_prediction"])
                self.assertEqual(a["sample_score"], b["sample_score"])
                self.assertNotEqual(a["truth"], b["truth"])

    def test_insufficient_references_fail_before_fitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            with patch("jhu_ta1.magnet.pair_coverage.predict") as estimator:
                with self.assertRaisesRegex(ValueError, "eligible references"):
                    evaluate(
                        root,
                        manifest,
                        query_budgets=[1],
                        num_replicates=2,
                        dimensions=9,
                    )
                estimator.assert_not_called()

    def test_multiple_budgets_execute_and_tag_every_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            manifest["target_models"] = ["family0/model0"]
            stream = io.StringIO()
            result = evaluate(
                root,
                manifest,
                query_budgets=[1, 2],
                num_replicates=2,
                dimensions=2,
                trial_stream=stream,
            )
            m = result["result"]["metrics"]
            self.assertEqual(m["num_settings"], 4)
            self.assertEqual(m["dkps_fits"], 8)
            rows = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertEqual(len(rows), 8)
            self.assertEqual({r["n_eval"] for r in rows}, {1, 2})
            self.assertTrue(all(len(r["query_ids"]) == r["n_eval"] for r in rows))

    def test_measurements_preserve_predictions_and_retained_inputs_support_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            manifest["target_models_by_dataset"] = {
                manifest["datasets"][0]["dataset"]: ["family0/model0"],
                manifest["datasets"][1]["dataset"]: ["family0/model0", "family1/model4"],
            }
            before, after, batches = io.StringIO(), io.StringIO(), io.StringIO()
            kwargs = dict(query_budgets=[1, 2], num_replicates=2, dimensions=2)
            original = evaluate(root, manifest, trial_stream=before, **kwargs)
            measured = evaluate(root, manifest, trial_stream=after,
                                record_measurements=True, batch_stream=batches, **kwargs)
            for key, value in original["result"]["metrics"].items():
                self.assertEqual(measured["result"]["metrics"][key], value)
            self.assertNotIn("measurements", original)
            trials = [json.loads(line) for line in after.getvalue().splitlines()]
            prior = [json.loads(line) for line in before.getvalue().splitlines()]
            batch_rows = [json.loads(line) for line in batches.getvalue().splitlines()]
            by_batch = {row["batch_id"]: row for row in batch_rows}
            sources = {row["dataset"]: row for row in measured["protocol"]["sources"]}
            for old, new in zip(prior, trials):
                for key, value in old.items():
                    self.assertEqual(new[key], value)
                self.assertEqual(np.mean(new["target_query_scores"]), new["sample_score"])
                bank = sources[new["dataset"]]["full_scores"]
                self.assertNotIn(new["model"], new["reference_models"])
                # A cheap reference-mean control can now be replayed from the
                # retained evidence without reopening the original HELM bank.
                control = .8 * new["sample_score"] + .2 * np.mean(
                    [bank[model] for model in new["reference_models"]])
                self.assertTrue(np.isfinite(control))
                self.assertEqual(by_batch[new["batch_id"]]["query_ids"], new["query_ids"])
                self.assertEqual(new["raw_prediction"],
                                 np.clip(new["diagnostics"]["unclipped_prediction"], 0, 1))
                self.assertGreaterEqual(new['diagnostics']['elapsed']['wall_seconds'], 0)
                self.assertEqual(new['diagnostics']['reference_count'], len(new['reference_models']))
            self.assertEqual(len(trials), 12)
            self.assertEqual(len(batch_rows), 8)  # shared across unequal target rosters
            report = measured["measurements"]
            self.assertEqual(report["representation_requested_rows"], 144)
            unique = {}
            for row in batch_rows:
                unique.setdefault(row["dataset"], set()).update(row["query_ids"])
                self.assertEqual(row["representation"], "one_hot")
                self.assertGreater(row["array_bytes"], 0)
            self.assertEqual(report["representation_unique_response_rows"],
                             sum(map(len, unique.values())) * 12)
            shared = sum(row["stages"]["shared_representation"]["wall_seconds"]
                         for row in batch_rows)
            self.assertAlmostEqual(report["stages"]["shared_representation"]["wall_seconds"], shared)
            self.assertIsNone(report["acquisition"]["original_generation_seconds"])
            self.assertIn("jhu_ta1.magnet.pair_coverage", report["runtime"]["sources"])
            for value in report["stages"].values():
                self.assertGreaterEqual(value["wall_seconds"], 0)
                self.assertGreaterEqual(value["process_cpu_seconds"], 0)

    def test_card_requires_complete_fixed_alpha_setting_coverage(self):
        root = Path(__file__).resolve().parents[1]
        card = yaml.safe_load(
            (
                root / "jhu_ta1/cards/jhu_run_predict_pair_coverage_kwdagger.yaml"
            ).read_text()
        )
        pipeline = yaml.safe_load(
            (root / "jhu_ta1/magnet/pair_coverage_pipeline.yaml").read_text()
        )
        algo_params = pipeline["nodes"]["pair_coverage"]["algo_params"]
        self.assertEqual(algo_params["alpha"], 0.8)
        # Budgets live in the pipeline as one scalar, so all of them land in the
        # same evidence row; the card must not split them across matrix rows.
        self.assertEqual(algo_params["query_budgets"], "1,2,4,8")
        self.assertNotIn("pair_coverage.query_budgets", card["kwdagger"]["matrix"])
        self.assertNotIn("pair_coverage.n_eval", card["kwdagger"]["matrix"])
        values = dict(
            alpha=0.8,
            n_components_cmds=8,
            budget_spec="1,2,4,8",
            num_models=25,
            num_datasets=4,
            num_replicates=10,
            num_pairs=100,
            num_budgets=4,
            num_settings=400,
            dkps_fits=4000,
            embedding_batches=160,
            settings_improved=384,
            setting_improvement_fraction=0.96,
        )
        self.run_claim(values)
        for update, message in [
            (dict(alpha=0.85), "Expected sample weight"),
            (dict(dkps_fits=0), "Expected one DKPS fit"),
            (dict(num_settings=100), "Every dataset/model/budget combination"),
            (dict(budget_spec="1,2,4,4"), "Budgets must be distinct"),
            (dict(settings_improved=401), "Improvement count must be between"),
            (
                dict(settings_improved=380, setting_improvement_fraction=0.95),
                "the claim requires more than",
            ),
        ]:
            with (
                self.subTest(update=update),
                self.assertRaisesRegex(AssertionError, message),
            ):
                self.run_claim(values | update)


if __name__ == "__main__":
    unittest.main()
