from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "infra"))

import cosmos_sharding


MODEL_REVISION = "2" * 40
GIT_SHA = "3" * 40
IMAGE_REF = "registry/image@sha256:" + "4" * 64
RUN_URI = "gs://bucket/runs/cosmos"


def make_preflight(counts: tuple[int, ...] = (2, 3, 4, 5, 6, 7)) -> dict:
    datasets = [
        {
            "dataset": f"dataset-{index}",
            "annotation_sha256": f"{index + 1:064x}",
            "category_count": 1,
            "image_count": image_count,
            "annotation_count": image_count * 2,
            "degenerate_annotation_count": 0,
        }
        for index, image_count in enumerate(counts)
    ]
    return {
        "status": "passed",
        "model_id": "nvidia/Cosmos3-Edge",
        "prompt_version": "cosmos3-edge-rf100-basic-v2",
        "dataset": {
            "expected_dataset_count": len(datasets),
            "dataset_count": len(datasets),
            "image_count": sum(counts),
            "datasets": datasets,
        },
    }


def make_summary(name: str, image_count: int, value: float = 0.1) -> dict:
    return {
        "dataset": name,
        "image_count": image_count,
        "completed_image_count": image_count,
        "successful_image_count": image_count,
        "new_error_count": 0,
        "model_failure_count": 0,
        "model_failure_counts": {"timeout": 0, "max_tokens": 0, "invalid_response": 0},
        "prediction_count": image_count,
        "diagnostics": {},
        "complete": True,
        "metrics": {"AP": value, "AP50": value * 2},
    }


def make_baseline(preflight: dict, names: tuple[str, ...] = ("dataset-0",)) -> dict:
    index = {item["dataset"]: item for item in preflight["dataset"]["datasets"]}
    summaries = [make_summary(name, index[name]["image_count"]) for name in names]
    return {
        "status": "running",
        "selected_dataset_count": len(index),
        "processed_dataset_count": len(summaries),
        "scored_dataset_count": len(summaries),
        "datasets": summaries,
    }


def make_plan(*, shard_count: int = 2) -> tuple[dict, dict, dict]:
    preflight = make_preflight()
    baseline = make_baseline(preflight)
    plan = cosmos_sharding.build_shard_plan(
        preflight=preflight,
        baseline_aggregate=baseline,
        gcs_run_uri=RUN_URI,
        shard_count=shard_count,
        image_ref=IMAGE_REF,
        benchmark_git_sha=GIT_SHA,
        model_revision=MODEL_REVISION,
        plan_id="test-plan",
        created_at="2026-01-01T00:00:00+00:00",
    )
    return plan, preflight, baseline


def shard_aggregate(plan: dict, shard: dict, value: float = 0.2) -> dict:
    summaries = [
        make_summary(item["dataset"], item["image_count"], value)
        for item in shard["datasets"]
    ]
    return {
        "status": "complete",
        "selected_dataset_count": len(summaries),
        "processed_dataset_count": len(summaries),
        "scored_dataset_count": len(summaries),
        "expected_dataset_count": len(summaries),
        "datasets": summaries,
    }


class FakeBlob:
    def __init__(self, client: "FakeStorage", bucket_name: str, name: str):
        self.client = client
        self.bucket_name = bucket_name
        self.name = name

    @property
    def key(self) -> tuple[str, str]:
        return self.bucket_name, self.name

    def upload_from_string(self, value, **kwargs) -> None:
        if kwargs.get("if_generation_match") == 0 and self.key in self.client.objects:
            raise RuntimeError("precondition failed")
        if isinstance(value, str):
            value = value.encode()
        self.client.objects[self.key] = bytes(value)
        self.client.operations.append(("upload", self.bucket_name, self.name))

    def download_as_bytes(self) -> bytes:
        return self.client.objects[self.key]

    def exists(self, client=None) -> bool:
        return self.key in self.client.objects


class FakeBucket:
    def __init__(self, client: "FakeStorage", name: str):
        self.client = client
        self.name = name

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self.client, self.name, name)

    def copy_blob(self, blob: FakeBlob, destination_bucket: "FakeBucket", name: str):
        self.client.objects[(destination_bucket.name, name)] = self.client.objects[blob.key]
        self.client.operations.append(("copy", destination_bucket.name, name))
        return destination_bucket.blob(name)


class FakeStorage:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.operations: list[tuple[str, str, str]] = []

    def bucket(self, name: str) -> FakeBucket:
        return FakeBucket(self, name)

    def list_blobs(self, bucket_name: str, prefix: str):
        return [
            FakeBlob(self, bucket_name, name)
            for bucket, name in sorted(self.objects)
            if bucket == bucket_name and name.startswith(prefix)
        ]

    def put_json(self, uri: str, value: dict) -> None:
        bucket, name = cosmos_sharding._parse_uri(uri)
        self.objects[(bucket, name)] = json.dumps(value).encode()

    def put_bytes(self, uri: str, value: bytes = b"data") -> None:
        bucket, name = cosmos_sharding._parse_uri(uri)
        self.objects[(bucket, name)] = value

    def has(self, uri: str) -> bool:
        bucket, name = cosmos_sharding._parse_uri(uri)
        return (bucket, name) in self.objects

    def read_json(self, uri: str) -> dict:
        bucket, name = cosmos_sharding._parse_uri(uri)
        return json.loads(self.objects[(bucket, name)])


class PlanTests(unittest.TestCase):
    def test_plan_is_balanced_disjoint_and_exact(self):
        plan, preflight, baseline = make_plan()
        baseline_names = set(plan["baseline"]["datasets"])
        assignments = [
            item["dataset"] for shard in plan["shards"] for item in shard["datasets"]
        ]
        all_names = {item["dataset"] for item in preflight["dataset"]["datasets"]}
        self.assertEqual(len(assignments), len(set(assignments)))
        self.assertEqual(set(assignments), all_names - baseline_names)
        self.assertFalse(baseline_names.intersection(assignments))
        self.assertEqual(plan["expected_image_count"], 27)
        loads = [shard["image_count"] for shard in plan["shards"]]
        self.assertLessEqual(max(loads) - min(loads), 3)
        self.assertEqual(len({shard["gcs_uri"] for shard in plan["shards"]}), 2)
        cosmos_sharding.validate_shard_plan(plan)

    def test_plan_generation_is_deterministic(self):
        preflight = make_preflight()
        baseline = make_baseline(preflight)
        kwargs = dict(
            preflight=preflight,
            baseline_aggregate=baseline,
            gcs_run_uri=RUN_URI,
            shard_count=2,
            image_ref=IMAGE_REF,
            benchmark_git_sha=GIT_SHA,
            model_revision=MODEL_REVISION,
            created_at="2026-01-01T00:00:00+00:00",
        )
        self.assertEqual(
            cosmos_sharding.build_shard_plan(**kwargs),
            cosmos_sharding.build_shard_plan(**kwargs),
        )

    def test_duplicate_or_missing_assignment_is_rejected(self):
        plan, _, _ = make_plan()
        duplicate = copy.deepcopy(plan)
        duplicate["shards"][1]["datasets"].append(
            copy.deepcopy(duplicate["shards"][0]["datasets"][0])
        )
        duplicate["shards"][1]["dataset_count"] += 1
        duplicate["shards"][1]["image_count"] += duplicate["shards"][0]["datasets"][0][
            "image_count"
        ]
        with self.assertRaisesRegex(ValueError, "multiple shards"):
            cosmos_sharding.validate_shard_plan(duplicate)

        missing = copy.deepcopy(plan)
        removed = missing["shards"][0]["datasets"].pop()
        missing["shards"][0]["dataset_count"] -= 1
        missing["shards"][0]["image_count"] -= removed["image_count"]
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            cosmos_sharding.validate_shard_plan(missing)

    def test_tampered_counts_metadata_and_uris_are_rejected(self):
        plan, _, _ = make_plan()
        cases = []
        bad = copy.deepcopy(plan)
        bad["expected_image_count"] += 1
        cases.append(bad)
        bad = copy.deepcopy(plan)
        bad["shards"][0]["datasets"][0]["annotation_sha256"] = "f" * 64
        cases.append(bad)
        bad = copy.deepcopy(plan)
        bad["shards"][1]["gcs_uri"] = bad["shards"][0]["gcs_uri"]
        cases.append(bad)
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    cosmos_sharding.validate_shard_plan(value)

    def test_incomplete_baseline_and_mutable_image_are_rejected(self):
        preflight = make_preflight()
        baseline = make_baseline(preflight)
        baseline["datasets"][0]["complete"] = False
        with self.assertRaisesRegex(ValueError, "incomplete"):
            cosmos_sharding.build_shard_plan(
                preflight=preflight,
                baseline_aggregate=baseline,
                gcs_run_uri=RUN_URI,
                shard_count=2,
                image_ref=IMAGE_REF,
                benchmark_git_sha=GIT_SHA,
                model_revision=MODEL_REVISION,
            )
        baseline = make_baseline(preflight)
        with self.assertRaisesRegex(ValueError, "immutable"):
            cosmos_sharding.build_shard_plan(
                preflight=preflight,
                baseline_aggregate=baseline,
                gcs_run_uri=RUN_URI,
                shard_count=2,
                image_ref="registry/image:latest",
                benchmark_git_sha=GIT_SHA,
                model_revision=MODEL_REVISION,
            )


class AggregateTests(unittest.TestCase):
    def test_merge_is_exact_and_recomputes_macro_metrics(self):
        plan, _, baseline = make_plan()
        aggregates = {
            shard["shard_id"]: shard_aggregate(plan, shard)
            for shard in plan["shards"]
        }
        merged = cosmos_sharding.merge_aggregates(plan, baseline, aggregates)
        self.assertEqual(merged["status"], "complete")
        self.assertEqual(merged["dataset_count"], 6)
        self.assertEqual(merged["completed_image_count"], 27)
        self.assertEqual([item["dataset"] for item in merged["datasets"]], sorted(
            item["dataset"] for item in plan["datasets"]
        ))
        self.assertAlmostEqual(merged["macro_AP"], (0.1 + 5 * 0.2) / 6)
        self.assertAlmostEqual(merged["macro_AP50"], (0.2 + 5 * 0.4) / 6)

    def test_missing_unexpected_incomplete_or_duplicate_summary_fails(self):
        plan, _, baseline = make_plan()
        aggregates = {
            shard["shard_id"]: shard_aggregate(plan, shard)
            for shard in plan["shards"]
        }
        with self.assertRaisesRegex(ValueError, "aggregate set"):
            cosmos_sharding.merge_aggregates(
                plan, baseline, {next(iter(aggregates)): next(iter(aggregates.values()))}
            )
        shard_id = plan["shards"][0]["shard_id"]
        cases = []
        missing = copy.deepcopy(aggregates[shard_id])
        missing["datasets"].pop()
        cases.append(missing)
        duplicate = copy.deepcopy(aggregates[shard_id])
        duplicate["datasets"].append(copy.deepcopy(duplicate["datasets"][0]))
        cases.append(duplicate)
        incomplete = copy.deepcopy(aggregates[shard_id])
        incomplete["datasets"][0]["complete"] = False
        cases.append(incomplete)
        for aggregate in cases:
            with self.subTest(aggregate=aggregate):
                with self.assertRaises(ValueError):
                    cosmos_sharding.verify_shard_aggregate(plan, shard_id, aggregate)

    def test_frozen_baseline_content_tampering_is_rejected(self):
        plan, _, baseline = make_plan()
        aggregates = {
            shard["shard_id"]: shard_aggregate(plan, shard)
            for shard in plan["shards"]
        }
        tampered = copy.deepcopy(baseline)
        tampered["datasets"][0]["metrics"]["AP"] = 0.999
        with self.assertRaisesRegex(ValueError, "content hash"):
            cosmos_sharding.merge_aggregates(plan, tampered, aggregates)


class GCSLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.plan, _, self.baseline = make_plan()
        self.storage = FakeStorage()
        self.client_patch = mock.patch(
            "cosmos_sharding.gcs_client", side_effect=lambda: self.storage
        )
        self.client_patch.start()

    def tearDown(self):
        self.client_patch.stop()

    def dataset_shard(self, name: str) -> dict:
        return next(
            shard
            for shard in self.plan["shards"]
            if name in {item["dataset"] for item in shard["datasets"]}
        )

    def populate_complete_inputs(self) -> None:
        self.storage.put_json(self.plan["baseline"]["aggregate_uri"], self.baseline)
        dataset_index = {item["dataset"]: item for item in self.plan["datasets"]}
        for name in self.plan["baseline"]["datasets"]:
            self.storage.put_json(
                f"{self.plan['canonical_full_uri']}/{name}/summary.json",
                next(item for item in self.baseline["datasets"] if item["dataset"] == name),
            )
            self.storage.put_json(
                f"{self.plan['canonical_full_uri']}/{name}/cosmos_detection_results.json", []
            )
            self.storage.put_json(
                f"{self.plan['canonical_full_uri']}/{name}/run_config_test.json", {}
            )
            for image in range(dataset_index[name]["image_count"]):
                self.storage.put_json(
                    f"{self.plan['canonical_full_uri']}/{name}/records/run/{image}.json",
                    {"image_id": image},
                )
        for shard in self.plan["shards"]:
            aggregate = shard_aggregate(self.plan, shard)
            self.storage.put_json(f"{shard['gcs_uri']}/aggregate_summary.json", aggregate)
            self.storage.put_json(f"{shard['gcs_uri']}/_SUCCESS.json", {"ok": True})
            for dataset in shard["datasets"]:
                name = dataset["dataset"]
                self.storage.put_json(
                    f"{shard['gcs_uri']}/{name}/summary.json",
                    next(item for item in aggregate["datasets"] if item["dataset"] == name),
                )
                self.storage.put_json(
                    f"{shard['gcs_uri']}/{name}/cosmos_detection_results.json", []
                )
                self.storage.put_json(
                    f"{shard['gcs_uri']}/{name}/run_config_test.json", {}
                )
                for image in range(dataset["image_count"]):
                    self.storage.put_json(
                        f"{shard['gcs_uri']}/{name}/records/run/{image}.json",
                        {"image_id": image},
                    )

    def test_partial_checkpoint_is_seeded_to_exactly_one_shard(self):
        partial_name = self.plan["shards"][0]["datasets"][0]["dataset"]
        baseline_name = self.plan["baseline"]["datasets"][0]
        self.storage.put_json(
            f"{self.plan['canonical_full_uri']}/{partial_name}/records/run/1.json",
            {"image_id": 1},
        )
        self.storage.put_json(
            f"{self.plan['canonical_full_uri']}/{partial_name}/run_config.json", {}
        )
        self.storage.put_json(
            f"{self.plan['canonical_full_uri']}/{baseline_name}/records/run/1.json",
            {"image_id": 1},
        )
        copied = cosmos_sharding.seed_existing_checkpoints(self.plan)
        self.assertEqual(copied, {partial_name: 2})
        assigned = self.dataset_shard(partial_name)
        self.assertTrue(
            self.storage.has(
                f"{assigned['gcs_uri']}/{partial_name}/records/run/1.json"
            )
        )
        other = next(shard for shard in self.plan["shards"] if shard != assigned)
        self.assertFalse(
            self.storage.has(f"{other['gcs_uri']}/{partial_name}/records/run/1.json")
        )

    def test_finalizer_waits_without_writing_any_canonical_success(self):
        first = self.plan["shards"][0]
        self.storage.put_json(f"{first['gcs_uri']}/_SUCCESS.json", {"ok": True})
        self.storage.put_json(
            f"{first['gcs_uri']}/aggregate_summary.json",
            shard_aggregate(self.plan, first),
        )
        result = cosmos_sharding.finalize_if_ready(self.plan)
        self.assertEqual(result["status"], "waiting")
        self.assertFalse(
            self.storage.has(f"{self.plan['canonical_full_uri']}/_SUCCESS.json")
        )

    def test_finalizer_copies_all_artifacts_aggregates_and_writes_success_last(self):
        self.populate_complete_inputs()
        result = cosmos_sharding.finalize_if_ready(self.plan)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["dataset_count"], 6)
        self.assertEqual(result["image_record_count"], 27)
        aggregate = self.storage.read_json(
            f"{self.plan['canonical_full_uri']}/aggregate_summary.json"
        )
        self.assertEqual(aggregate["status"], "complete")
        self.assertEqual(aggregate["scored_dataset_count"], 6)
        for shard in self.plan["shards"]:
            for dataset in shard["datasets"]:
                name = dataset["dataset"]
                self.assertTrue(
                    self.storage.has(
                        f"{self.plan['canonical_full_uri']}/{name}/summary.json"
                    )
                )
        success_uri = f"{self.plan['canonical_full_uri']}/_SUCCESS.json"
        self.assertTrue(self.storage.has(success_uri))
        self.assertEqual(self.storage.operations[-1][0], "upload")
        self.assertEqual(self.storage.operations[-1][2], cosmos_sharding._parse_uri(success_uri)[1])

    def test_missing_or_extra_record_rejects_before_canonical_success(self):
        for mutation in ("missing", "extra"):
            with self.subTest(mutation=mutation):
                self.storage = FakeStorage()
                self.populate_complete_inputs()
                dataset = self.plan["shards"][0]["datasets"][0]
                base = f"{self.plan['shards'][0]['gcs_uri']}/{dataset['dataset']}/records/run"
                if mutation == "missing":
                    bucket, name = cosmos_sharding._parse_uri(f"{base}/0.json")
                    del self.storage.objects[(bucket, name)]
                else:
                    self.storage.put_json(f"{base}/extra.json", {"image_id": "extra"})
                with self.assertRaisesRegex(ValueError, "Record count"):
                    cosmos_sharding.finalize_if_ready(self.plan)
                self.assertFalse(
                    self.storage.has(f"{self.plan['canonical_full_uri']}/_SUCCESS.json")
                )

    def test_missing_predictions_summary_or_run_config_rejects_success(self):
        for suffix in (
            "summary.json",
            "cosmos_detection_results.json",
            "run_config_test.json",
        ):
            with self.subTest(suffix=suffix):
                self.storage = FakeStorage()
                self.populate_complete_inputs()
                dataset = self.plan["shards"][0]["datasets"][0]
                uri = f"{self.plan['shards'][0]['gcs_uri']}/{dataset['dataset']}/{suffix}"
                bucket, name = cosmos_sharding._parse_uri(uri)
                del self.storage.objects[(bucket, name)]
                with self.assertRaisesRegex(ValueError, "missing"):
                    cosmos_sharding.finalize_if_ready(self.plan)
                self.assertFalse(
                    self.storage.has(f"{self.plan['canonical_full_uri']}/_SUCCESS.json")
                )

    def test_copy_failure_never_publishes_success(self):
        self.populate_complete_inputs()
        with (
            mock.patch(
                "cosmos_sharding._copy_prefix",
                side_effect=RuntimeError("simulated copy failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "copy failure"),
        ):
            cosmos_sharding.finalize_if_ready(self.plan)
        self.assertFalse(
            self.storage.has(f"{self.plan['canonical_full_uri']}/_SUCCESS.json")
        )

    def test_finalization_is_idempotent(self):
        self.populate_complete_inputs()
        first = cosmos_sharding.finalize_if_ready(self.plan)
        second = cosmos_sharding.finalize_if_ready(self.plan)
        self.assertEqual(first["dataset_count"], second["dataset_count"])
        self.assertEqual(first["image_record_count"], second["image_record_count"])
        self.assertTrue(
            self.storage.has(f"{self.plan['canonical_full_uri']}/_SUCCESS.json")
        )


if __name__ == "__main__":
    unittest.main()
