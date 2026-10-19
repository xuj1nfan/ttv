"""Offline integration tests; scripted responses test plumbing, not model efficacy."""

from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
from urllib.error import URLError

from experiments.nli_diagnostics import export_annotations, score_annotations
from experiments.run_first_round import make_runner, run_group, summarize
from experiments.smoke_nli import probe_premise
from framework.data import load_cases, sample_cail, scan_training, text_fingerprint, write_jsonl
from framework.evaluation import classification_metrics, paired_comparison
from framework.local_generator import LocalChatClient, LocalChatGenerator
from framework.labels import normalize_charge
from framework.nli_verifier import NLIVerifier, PairTooLong, label_mapping
from framework.schema import Claim


LABELS = ["盗窃", "抢劫"]
CONFIG = {"seed": 42, "intervention": {"conflict_threshold": 0.9, "insufficient_threshold": 0.95}}


class FixtureNLIBackend:
    mapping = label_mapping({0: "entailment", 1: "contradiction", 2: "neutral"})
    metadata = {"model": "offline-fixture", "fixture": True}

    def predict(self, premise, hypothesis):
        if hypothesis == "too long":
            raise PairTooLong(600, 512)
        logits = [0.0, 8.0, 0.0] if "未使用暴力" in premise else [8.0, 0.0, 0.0]
        return logits, {"input_tokens": 20, "truncated": False}


def fixture_transport(captured):
    def respond(request, timeout):
        if request.full_url.endswith("/models"):
            return io.BytesIO(json.dumps({"data": [{"id": "test-qwen"}]}).encode())
        payload = json.loads(request.data)
        captured.append(payload)
        messages = payload["messages"]
        instruction = messages[-1]["content"]
        if "阶段3" in instruction:
            reviewed = any("核查反馈：" in item["content"] for item in messages)
            # Scripted answers ONLY verify that feedback reaches a later request.
            content = {"charge": "盗窃" if reviewed else "抢劫"}
        else:
            content = {"analysis": "用于接口测试的分析摘要", "claim": "取得手机时使用了暴力。"}
        return io.BytesIO(json.dumps({
            "choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }).encode())
    return respond


class FirstRoundTests(unittest.TestCase):
    def client(self):
        return LocalChatClient("http://127.0.0.1:8000/v1", "test-qwen", extra_body={"chat_template_kwargs": {"enable_thinking": False}})

    def cases(self):
        texts = ["张某拿走手机，未使用暴力。", "李某使用暴力取得手机。"]
        return [{"case_id": f"c{i}", "case_text": text, "text_sha256": text_fingerprint(text), "gold_charge": LABELS[i], "split": "dev"} for i, text in enumerate(texts)]

    def test_real_adapters_and_four_groups_through_http_transport(self):
        captured = []
        with tempfile.TemporaryDirectory() as directory, patch("framework.local_generator.build_opener") as opener:
            opener.return_value.open.side_effect = fixture_transport(captured)
            client = self.client()
            client.check_model()
            groups = {}
            nli = NLIVerifier(backend=FixtureNLIBackend())
            for group in ("baseline", "shadow", "self_check", "nli_conflict"):
                groups[group] = run_group(group, self.cases(), LABELS, client, CONFIG, Path(directory) / group, nli=nli)
            report = summarize(groups, LABELS, resamples=20)
            self.assertEqual(report["paired_vs_baseline"]["shadow"]["delta_accuracy"], 0)
            self.assertEqual(report["paired_vs_baseline"]["nli_conflict"]["wrong_to_right"], 1)
            self.assertEqual(report["paired_vs_baseline"]["nli_conflict"]["right_to_wrong"], 0)
            self.assertEqual(report["groups"]["baseline"]["tokens"]["total_tokens"], 720)
            self.assertEqual(report["groups"]["baseline"]["nli_model_calls"], 0)
            self.assertEqual(report["groups"]["shadow"]["nli_model_calls"], 4)
            baseline_requests, shadow_requests = captured[:6], captured[6:12]
            self.assertEqual(baseline_requests, shadow_requests)
            self.assertTrue(all("gold_charge" not in json.dumps(item) for item in captured))
            self.assertTrue(all(not item["chat_template_kwargs"]["enable_thinking"] for item in captured))
            final_requests = [item for item in captured if "阶段3" in item["messages"][-1]["content"]]
            self.assertTrue(final_requests)
            self.assertTrue(all(item["response_format"]["json_schema"]["schema"]["properties"]["charge"]["enum"] == LABELS for item in final_requests))
            self.assertTrue(all(item["response_format"] == {"type": "json_object"} for item in captured if "阶段3" not in item["messages"][-1]["content"]))
            self.assertEqual(groups["nli_conflict"][0]["intervention_count"], 2)
            self.assertEqual(groups["nli_conflict"][1]["intervention_count"], 0)

    def test_blinded_annotations_and_threshold_sweep(self):
        with tempfile.TemporaryDirectory() as directory, patch("framework.local_generator.build_opener") as opener:
            opener.return_value.open.side_effect = fixture_transport([])
            root = Path(directory)
            run_group("shadow", self.cases(), LABELS, self.client(), CONFIG, root / "shadow", nli=NLIVerifier(backend=FixtureNLIBackend()))
            trace = root / "shadow/traces.jsonl"
            annotations = export_annotations(trace)
            self.assertEqual(len(annotations), 4)
            unique = export_annotations(trace, unique=True)
            self.assertEqual(len(unique), 2)
            self.assertTrue(all(len(item["duplicate_annotation_ids"]) == 1 for item in unique))
            self.assertEqual(len(export_annotations(trace, unique=True, limit=1, seed=42)), 1)
            for item in annotations:
                self.assertNotIn("probabilities", item)
                self.assertNotIn("gold_charge", item)
                self.assertNotIn("decision", item)
                item["gold_label"] = "conflict" if item["case_id"] == "c0" else "support"
            path = root / "annotations.jsonl"
            write_jsonl(path, annotations)
            report = score_annotations(trace, path, [0.9, 1.0])
            self.assertEqual(report["accuracy_on_evaluated"], 1)
            self.assertEqual(report["conflict_thresholds"][0]["conflict_precision"], 1)
            self.assertIsNone(report["conflict_thresholds"][1]["conflict_precision"])
            annotations[0]["claim"] = "changed claim"
            write_jsonl(path, annotations)
            with self.assertRaisesRegex(ValueError, "Changed"):
                score_annotations(trace, path, [0.9])

    def test_nli_mapping_and_overlength_abstention(self):
        self.assertEqual(probe_premise({"case_id": "c0", "hypothesis": "claim"}, {"c0": "case fact"}), "case fact")
        self.assertEqual(probe_premise({"premise": "direct fact"}, {}), "direct fact")
        with self.assertRaisesRegex(ValueError, "Unknown probe case_id"):
            probe_premise({"case_id": "missing"}, {})
        with self.assertRaises(ValueError):
            label_mapping({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
        with self.assertRaises(ValueError):
            label_mapping({0: "entailment", 1: "entailment", 2: "neutral"})
        verifier = NLIVerifier(backend=FixtureNLIBackend())
        result = verifier.verify("未使用暴力", Claim("使用了暴力", 1))
        self.assertEqual(result.label, "conflict")
        self.assertGreater(result.probabilities["conflict"], 0.99)
        self.assertAlmostEqual(sum(result.probabilities.values()), 1)
        skipped = verifier.verify("fact", Claim("too long", 1))
        self.assertEqual(skipped.label, "unverifiable")
        self.assertFalse(skipped.metadata["evaluated"])
        self.assertIsNone(skipped.metadata["model_probabilities"])

    def test_conflict_only_does_not_feedback_on_neutral_probability_one(self):
        runner = make_runner("nli_conflict", self.client(), LABELS, CONFIG, NLIVerifier(backend=FixtureNLIBackend()))
        from framework.schema import Decision
        self.assertFalse(runner.policy.decide(Decision("insufficient", 1.0, {"insufficient": 1.0})).intervene)

    def test_network_failure_is_retained_and_usage_not_reported_as_zero(self):
        with tempfile.TemporaryDirectory() as directory, patch("framework.local_generator.build_opener") as opener:
            opener.return_value.open.side_effect = URLError("offline")
            rows = run_group("baseline", self.cases()[:1], LABELS, self.client(), CONFIG, Path(directory) / "baseline")
            self.assertEqual(rows[0]["status"], "error")
            self.assertEqual(rows[0]["usage_missing_calls"], 1)
            metrics = classification_metrics(rows, LABELS)
            self.assertEqual(metrics["accuracy"], 0)
            self.assertEqual(metrics["errors"], 1)
            self.assertIsNone(metrics["tokens"])

    def test_malformed_continuation_preserves_pending_trace_and_cost(self):
        good = fixture_transport([])
        def respond(request, timeout):
            payload = json.loads(request.data)
            if "阶段2" in payload["messages"][-1]["content"]:
                return io.BytesIO(json.dumps({"choices": [{"message": {"content": "bad JSON"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}).encode())
            return good(request, timeout)
        with tempfile.TemporaryDirectory() as directory, patch("framework.local_generator.build_opener") as opener:
            opener.return_value.open.side_effect = respond
            root = Path(directory) / "baseline"
            rows = run_group("baseline", self.cases()[:1], LABELS, self.client(), CONFIG, root)
            self.assertEqual(rows[0]["status"], "error")
            self.assertEqual(rows[0]["usage"]["total_tokens"], 240)
            trace = [json.loads(line) for line in (root / "traces.jsonl").read_text().splitlines()]
            self.assertEqual(trace[0]["record_type"], "checkpoint")
            self.assertIsNone(trace[0]["reasoning_after"])
            self.assertEqual(trace[1]["record_type"], "run_error")

    def test_invalid_charge_counts_as_wrong_without_fuzzy_mapping(self):
        rows = [{"case_id": "c", "gold_charge": "盗窃", "prediction": "盗窃罪，也可能抢劫", "status": "ok"}]
        report = classification_metrics(rows, LABELS)
        self.assertEqual(report["invalid_outputs"], 1)
        self.assertEqual(report["accuracy"], 0)
        self.assertEqual(normalize_charge(" [走私、贩卖、运输、制造]毒品 "), "[走私、贩卖、运输、制造]毒品")

    def test_final_schema_uses_exact_candidate_labels(self):
        labels = ["骗取[贷款、票据承兑、金融票证]", "盗窃"]
        client = self.client()
        generator = LocalChatGenerator(client, labels)
        prompt = generator.start("案件事实").metadata["messages"][0]["content"]
        self.assertIn("骗取[贷款、票据承兑、金融票证]", prompt)
        schema = generator.final_response_format["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["charge"]["enum"], labels)
        self.assertEqual(schema["required"], ["charge"])
        self.assertFalse(schema["additionalProperties"])

    def test_paired_evaluation_rejects_missing_or_changed_cases(self):
        before = [{"case_id": "c", "gold_charge": "盗窃", "prediction": "盗窃", "status": "ok", "text_sha256": "a"}]
        after = deepcopy(before)
        after[0]["text_sha256"] = "b"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            paired_comparison(before, after)
        with self.assertRaises(ValueError):
            paired_comparison(before, [])
        after = deepcopy(before)
        after[0].update(status="error", prediction=None)
        self.assertEqual(paired_comparison(before, after, resamples=0)["right_to_wrong"], 1)

    def test_cail_split_deduplication_and_training_only_vocabulary(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return list(text)
        def raw(text, charges=None):
            return {"fact": text, "meta": {"criminals": ["张某"], "accusation": charges or ["盗窃"]}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, dev, test = [root / f"{name}.jsonl" for name in ("train", "dev", "test")]
            write_jsonl(train, [raw("训练事实")])
            write_jsonl(dev, [raw("训练事实"), raw("开发甲"), raw("开发乙"), raw("本院认为，构成盗窃罪。"), raw("犯罪事实为拿走手机。"), raw("多罪名", ["盗窃", "抢劫"])])
            write_jsonl(test, [raw("开发乙"), raw("测试事实"), raw("新标签", ["抢劫"])])
            labels, seen = scan_training(train)
            self.assertEqual(labels, ["盗窃"])
            dev_rows, counts, audit = sample_cail(dev, split="dev", labels=labels, seen=seen, tokenizer=Tokenizer(), sample_size=1, seed=42)
            self.assertEqual(counts["duplicate_or_earlier_split"], 1)
            self.assertEqual(counts["possible_answer_cue"], 1)
            self.assertEqual(len(audit), 1)
            test_rows, counts, _ = sample_cail(test, split="test", labels=labels, seen=seen, tokenizer=Tokenizer(), sample_size=5, seed=42)
            self.assertEqual(len(test_rows), 1)
            self.assertEqual(counts["duplicate_or_earlier_split"], 1)
            self.assertEqual(counts["charge_absent_from_training"], 1)
            prepared = root / "prepared.jsonl"
            write_jsonl(prepared, dev_rows + test_rows)
            self.assertEqual(len(load_cases(prepared, labels)), 2)

    def test_batch_and_evaluation_cli_write_a_complete_auditable_run(self):
        from experiments.run_first_round import main as run_main
        from experiments.evaluate_first_round import main as evaluate_main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, labels, config, output = [root / name for name in ("cases.jsonl", "labels.json", "config.json", "output")]
            write_jsonl(data, self.cases())
            labels.write_text(json.dumps(LABELS), encoding="utf-8")
            config.write_text(json.dumps({
                **CONFIG, "data": {"cases": str(data), "labels": str(labels)},
                "generator": {"base_url": "http://127.0.0.1:8000/v1", "model": "test-qwen"},
                "verifier": {},
            }), encoding="utf-8")
            with patch("sys.argv", ["run", "--config", str(config), "--groups", "baseline", "shadow", "self_check", "nli_conflict", "--output-dir", str(output)]), patch("framework.local_generator.build_opener") as opener, patch("experiments.run_first_round.NLIVerifier", return_value=NLIVerifier(backend=FixtureNLIBackend())), redirect_stdout(io.StringIO()):
                opener.return_value.open.side_effect = fixture_transport([])
                run_main()
            snapshot = json.loads((output / "run.json").read_text())
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(snapshot["status"], "complete")
            self.assertEqual(snapshot["final_output_protocol"], "charge_json_schema_enum_v1")
            self.assertTrue(report["complete"])
            self.assertEqual(snapshot["case_count"], 2)
            self.assertEqual(len(snapshot["cases_sha256"]), 64)
            with patch("sys.argv", ["evaluate", "--run-dir", str(output), "--resamples", "0"]), redirect_stdout(io.StringIO()):
                evaluate_main()
            rebuilt = json.loads((output / "summary.json").read_text())
            self.assertEqual(rebuilt["groups"], report["groups"])
            write_jsonl(output / "shadow/predictions.jsonl", [])
            with patch("sys.argv", ["evaluate", "--run-dir", str(output)]), self.assertRaisesRegex(ValueError, "Incomplete"):
                evaluate_main()

    def test_all_cli_help_commands_work_without_model_dependencies(self):
        import subprocess
        root = Path(__file__).resolve().parents[1]
        for name in ("prepare_cail.py", "run_first_round.py", "evaluate_first_round.py", "nli_diagnostics.py"):
            result = subprocess.run([sys.executable, str(root / "experiments" / name), "--help"], cwd=tempfile.gettempdir(), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
