"""CPU-only orchestration tests; synthetic evaluator artifacts, no model imports."""
import copy
import csv
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_sparse_longbench_batch as batch


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.output_patch = patch.object(batch, "OUTPUT", self.folder / "results/sparse_longbench")
        self.output_patch.start()
        self.addCleanup(self.output_patch.stop)
        self.manifest = batch.read_json(batch.MANIFEST)
        self.cases = []
        for task in self.manifest["settings"]["tasks"]:
            selected = []
            for index in self.manifest["sample_indices"][task]:
                case = dict(task=task, example_index=index, example_id=f"{task}-{index}",
                            prompt="fixture prompt", answer="answer", length_group=100)
                selected.append(case)
                self.cases.append(case)
            self.manifest["sample_id_digest"][task] = batch.sha_json([(c["example_id"], c["answer"]) for c in selected])
        self.manifest_path = self.folder / "manifest.json"
        batch.save_new(self.manifest_path, self.manifest)
        self.argv = ["--preset", "initial-4k", "--batch-name", "test",
                     "--manifest", str(self.manifest_path)]
        self.args = batch.parser().parse_args(self.argv)
        self.args.keep_fraction = [0.75, 0.9]
        self.jobs = batch.make_jobs(self.args, self.manifest)
        self.calls = []

    def invoke(self, extra=(), argv=None):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return batch.main((self.argv if argv is None else argv) + list(extra))

    def result(self, path, job=None):
        """Write a small, internally consistent 80-case evaluator result."""
        job = job or self.jobs[0]
        path.mkdir(parents=True)
        config = job["config"]
        provenance = {key: config[key] for key in ("model", "manifest_sha256", "evaluation_context",
                                                  "reference_policy", "case_count", "settings")}
        provenance.update({key: config["sources"][key] for key in batch.PROVENANCE_HASHES})
        provenance.update(diagnostic_subset=False, tokenizer_digest="a" * 64,
                          cases_sha256=batch.sha_json(self.cases), environment=dict(torch="fixture", gpu="fixture"))
        batch.save_new(path / "provenance.json", provenance)
        batch.save_new(path / "source_manifest.json", self.manifest)
        batch.save_new(path / "cases.json", self.cases)
        summaries = {}
        for arm in batch.ARMS:
            nll = 2.5 if arm == "reference_sparse" else 2.0
            backend = "original" if arm == "original" else "reference"
            rows, windows, calls = [], [], []
            for i, case in enumerate(self.cases):
                tokens = i % 3 + 1  # Nonuniform counts catch unweighted PPL aggregation.
                rows.append({**{k: case[k] for k in ("task", "example_index", "example_id")},
                             "reference_index": 0, "reference_policy": "first", "answer_tokens": tokens,
                             "tokens_scored": tokens, "nll_sum": nll * tokens,
                             "avg_nll": nll, "perplexity": math.exp(nll), "evaluation_context_tokens": 4096})
                windows.append(dict(shape=[1, 256], token_ids_sha256=batch.sha_json(i), forward_seconds=0.01))
                for layer in range(2):
                    call = dict(backend=backend, layer=layer)
                    if backend == "reference":
                        call.update(q_len=256, kv_len=256, dtype="torch.bfloat16", eligible_blocks=4,
                                    retained_blocks=3 if arm == "reference_sparse" else 4,
                                    eligible_causal_pairs=100, retained_causal_pairs=75 if arm == "reference_sparse" else 100)
                    calls.append(call)
            execution = dict(calls_by_backend={backend: len(calls)}, completed_attention_calls=len(calls),
                             failures={}, dense_fallback_calls=0,
                             calls_omitting_blocks=len(calls) if arm == "reference_sparse" else 0,
                             causal_pair_sparsity=0.25 if arm == "reference_sparse" else 0.0)
            for key in ("retained_blocks", "eligible_blocks", "retained_causal_pairs", "eligible_causal_pairs"):
                execution[key] = sum(c.get(key, 0) for c in calls)
            tokens = sum(r["tokens_scored"] for r in rows)
            summaries[arm] = dict(nll_sum=nll * tokens, scored_tokens=tokens, avg_nll=nll,
                                  perplexity=math.exp(nll), execution=execution, examples=80,
                                  measured_evaluation_wall_seconds=2.0, warmup_and_compilation_wall_seconds=1.0,
                                  model_forward_seconds=1.0, peak_allocated_bytes=1000, peak_reserved_bytes=2000)
            batch.save_new(path / f"{arm}_summary.json", summaries[arm])
            batch.save_new(path / f"{arm}_windows.json", windows)
            batch.save_new(path / f"{arm}_attention_calls.json", calls)
            with (path / f"{arm}_examples.jsonl").open("x") as handle:
                handle.writelines(json.dumps(r) + "\n" for r in rows)
        batch.save_new(path / "summary.json", dict(status="complete", paired_window_identity=True, arms=summaries))
        return path

    def fake_child(self, cmd, log_path, gpu, lock_fd):
        self.calls.append(cmd)
        self.assertEqual(gpu, "0")
        self.assertGreaterEqual(lock_fd, 0)
        log_path.write_text("mock evaluator\n")
        model = cmd[cmd.index("--model") + 1]
        keep = float(cmd[cmd.index("--keep-fraction") + 1])
        job = next(j for j in self.jobs if j["config"]["model"]["model"] == model
                   and j["config"]["settings"]["keep_fraction"] == keep)
        self.result(Path(cmd[-1]), job)
        return 0

    def test_selection_and_multiple_fractions(self):
        self.assertEqual(len(self.jobs), 6)
        self.assertEqual([j["config"]["model"]["model"] for j in self.jobs[::2]], batch.INITIAL_MODELS)
        self.assertEqual(len({j["evaluation_id"] for j in self.jobs}), 6)
        spec = next(m for m in self.manifest["models"] if m["model"] == "adpretko/celerity-906m-8k-ad0p2-ild")
        self.args.preset = None
        self.args.model = [spec["model"]]
        self.assertEqual(batch.select_models(self.args, self.manifest), [spec])
        self.assertEqual(batch.metadata(spec)["checkpoint_variant"], "ad0p2-ild")

    def test_models_file_uses_exact_manifest_ids(self):
        path = self.folder / "models.txt"
        path.write_text("# exact selection\n" + batch.INITIAL_MODELS[1] + "\n")
        self.args.preset, self.args.models_file = None, path
        selected = batch.select_models(self.args, self.manifest)
        self.assertEqual(selected[0]["model"], batch.INITIAL_MODELS[1])
        path.write_text("adpretko/celerity-guessed\n")
        with self.assertRaisesRegex(ValueError, "absent"):
            batch.select_models(self.args, self.manifest)

    def test_unsupported_and_catalog(self):
        self.args.preset = None
        unsupported = next(m for m in self.manifest["models"] if "1810m" in m["model"])
        self.args.model = [batch.INITIAL_MODELS[0], unsupported["model"]]
        with self.assertRaisesRegex(ValueError, unsupported["model"]):
            batch.select_models(self.args, self.manifest)
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(batch.main(["--list-models"]), 0)
        rows = list(csv.DictReader(io.StringIO(out.getvalue())))
        self.assertEqual(len(rows), 189)
        self.assertEqual(sum(r["supported"] == "False" for r in rows), 1)
        self.assertFalse(batch.OUTPUT.exists())

    def test_dry_run_no_writes_or_child(self):
        with patch.object(batch, "run_child", side_effect=AssertionError("must not launch")):
            self.assertEqual(self.invoke(["--dry-run"]), 0)
        self.assertFalse(batch.OUTPUT.exists())

    def test_dry_run_without_site_packages(self):
        cmd = [sys.executable, "-S", "-B", str(ROOT / "scripts/run_sparse_longbench_batch.py"),
               *self.argv, "--dry-run"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("RUN adpretko/"), 6)

    def test_forwarding_and_csv_and_resume_no_overwrite(self):
        with patch.object(batch, "run_child", side_effect=self.fake_child):
            self.assertEqual(self.invoke(), 0)
        self.assertEqual(len(self.calls), 6)
        for cmd in self.calls:
            for option, value in (("--backend", "reference"), ("--limit-cases", "0"),
                                  ("--local-blocks", "1"), ("--manifest", str(self.manifest_path))):
                self.assertEqual(cmd[cmd.index(option) + 1], value)
        evidence = {p: p.read_bytes() for p in batch.OUTPUT.rglob("*") if p.is_file()}
        with patch.object(batch, "run_child", side_effect=AssertionError("must reuse")):
            self.assertEqual(self.invoke(), 0)
        for path, data in evidence.items():
            self.assertEqual(path.read_bytes(), data)
        csv_paths = list(batch.OUTPUT.glob("batches/test/invocation_*/results.csv"))
        self.assertEqual(len(csv_paths), 2)
        with csv_paths[0].open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 6)
        row = rows[0]
        self.assertEqual(row["numerical_control_status"], "passed_exact")
        self.assertAlmostEqual(float(row["delta_nll_per_token"]), 0.5)
        self.assertAlmostEqual(float(row["ppl_change_percent"]), 100 * (math.exp(0.5) - 1))
        self.assertAlmostEqual(float(row["original_nll_sum"]), 2 * int(row["scored_answer_tokens"]))
        self.assertEqual(float(row["evaluation_seconds"]), 6.0)
        self.assertEqual(float(row["causal_pair_sparsity"]), 0.25)
        self.assertEqual(float(row["execution_coverage"]), 1.0)

    def test_resume_failure_preserves_attempt(self):
        def fail(cmd, log, gpu, fd):
            log.write_text("simulated failure")
            Path(cmd[-1]).mkdir()
            return 1
        with patch.object(batch, "run_child", side_effect=fail):
            self.assertEqual(self.invoke(), 1)
        old = {p: p.read_bytes() for p in batch.OUTPUT.rglob("*") if p.is_file()}
        with patch.object(batch, "run_child", side_effect=self.fake_child):
            self.assertEqual(self.invoke(), 0)
        self.assertEqual(len(self.calls), 6)
        for path, value in old.items():
            self.assertEqual(path.read_bytes(), value)
        self.assertEqual(len(list(batch.OUTPUT.glob("evaluations/*/attempt_*"))), 7)

    def test_interrupt_restarts_only_unfinished_evaluation(self):
        def interrupt_second(cmd, log, gpu, fd):
            if len(self.calls) == 1:
                log.write_text("interrupted")
                raise KeyboardInterrupt
            return self.fake_child(cmd, log, gpu, fd)
        with patch.object(batch, "run_child", side_effect=interrupt_second):
            self.assertEqual(self.invoke(), 130)
        self.calls.clear()
        with patch.object(batch, "run_child", side_effect=self.fake_child):
            self.assertEqual(self.invoke(), 0)
        self.assertEqual(len(self.calls), 5)

    def test_complete_after_parent_crash_is_recovered(self):
        job = self.jobs[0]
        attempt = batch.job_directory(job) / "attempt_crashed"
        attempt.mkdir(parents=True)
        batch.save_new(attempt / "request.json", job)
        self.result(attempt / "result", job)
        path, verified, rejected = batch.reusable(job, self.manifest)
        self.assertEqual(path, attempt / "result")
        self.assertEqual(verified[1], "passed_exact")
        self.assertFalse(rejected)

    def test_config_identity_and_batch_name_mismatch(self):
        with patch.object(batch, "run_child", return_value=1):
            self.assertEqual(self.invoke(), 1)
        with patch.object(batch, "run_child", side_effect=AssertionError("must not launch")):
            self.assertEqual(self.invoke(["--keep-fraction", "0.5"]), 2)
            self.assertEqual(self.invoke(["--local-blocks", "2"]), 2)
        changed_sources = dict(self.jobs[0]["config"]["sources"], attention_sha256="f" * 64)
        with patch.object(batch, "source_hashes", return_value=changed_sources):
            self.assertEqual(self.invoke(["--dry-run"]), 2)
        manifest = copy.deepcopy(self.manifest)
        next(m for m in manifest["models"] if m["model"] == batch.INITIAL_MODELS[0])["revision"] = "f" * 40
        self.manifest_path.write_text(json.dumps(manifest))
        self.assertEqual(self.invoke(["--dry-run"]), 2)

    def test_mismatched_provenance_rejected(self):
        path = self.result(self.folder / "result")
        original = batch.read_json(path / "provenance.json")
        for key in ("model", "manifest_sha256", "settings", "evaluation_context", "reference_policy", *batch.PROVENANCE_HASHES):
            with self.subTest(key=key):
                doc = copy.deepcopy(original)
                doc[key] = "changed"
                (path / "provenance.json").write_text(json.dumps(doc))
                with self.assertRaises(ValueError):
                    batch.verify_result(path, self.jobs[0]["config"], self.manifest)

    def test_missing_and_malformed_artifacts_rejected(self):
        path = self.result(self.folder / "result")
        for artifact in list(path.iterdir()):
            content = artifact.read_bytes()
            with self.subTest(artifact=artifact.name):
                artifact.rename(artifact.with_suffix(".hidden"))
                with self.assertRaises(ValueError):
                    batch.verify_result(path, self.jobs[0]["config"], self.manifest)
                artifact.with_suffix(".hidden").rename(artifact)
                artifact.write_text("null")
                with self.assertRaises(ValueError):
                    batch.verify_result(path, self.jobs[0]["config"], self.manifest)
                artifact.write_bytes(content)

    def test_incomplete_nonfinite_and_control_failures_rejected(self):
        path = self.result(self.folder / "result")
        original = batch.read_json(path / "summary.json")
        mutations = [lambda d: d.update(status="running"),
                     lambda d: d["arms"].pop("reference_full"),
                     lambda d: d["arms"]["reference_sparse"].update(examples=79),
                     lambda d: d["arms"]["original"].update(avg_nll=float("nan")),
                     lambda d: d["arms"]["reference_full"].update(avg_nll=2.1),
                     lambda d: d["arms"]["reference_sparse"]["execution"].update(dense_fallback_calls=1)]
        for mutation in mutations:
            doc = copy.deepcopy(original)
            mutation(doc)
            (path / "summary.json").write_text(json.dumps(doc))
            with self.assertRaises(ValueError):
                batch.verify_result(path, self.jobs[0]["config"], self.manifest)

    def test_bad_attempt_request_or_status_never_reused(self):
        job = self.jobs[0]
        attempt = batch.job_directory(job) / "attempt_wrong"
        attempt.mkdir(parents=True)
        self.result(attempt / "result", job)
        batch.save_new(attempt / "request.json", dict(evaluation_id="wrong"))
        self.assertIsNone(batch.reusable(job, self.manifest)[0])
        (attempt / "request.json").write_text(json.dumps(job))
        batch.save_new(attempt / "status.json", dict(status="failed"))
        self.assertIsNone(batch.reusable(job, self.manifest)[0])

    def test_existing_control_threshold_is_not_relaxed(self):
        path = self.result(self.folder / "result")
        summary = batch.read_json(path / "summary.json")
        rows_path = path / "reference_full_examples.jsonl"
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
        for nll, passes in ((2.01, True), (2.021, False)):
            for row in rows:
                row.update(nll_sum=nll * row["tokens_scored"], avg_nll=nll, perplexity=math.exp(nll))
            rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            stats = summary["arms"]["reference_full"]
            stats.update(nll_sum=sum(r["nll_sum"] for r in rows), avg_nll=nll, perplexity=math.exp(nll))
            (path / "reference_full_summary.json").write_text(json.dumps(stats))
            (path / "summary.json").write_text(json.dumps(summary))
            if passes:
                self.assertEqual(batch.verify_result(path, self.jobs[0]["config"], self.manifest)[1], "passed_within_0.02")
            else:
                with self.assertRaisesRegex(ValueError, "numerical control failed"):
                    batch.verify_result(path, self.jobs[0]["config"], self.manifest)

    def test_corrupt_success_is_rejected_and_preserved(self):
        job = self.jobs[0]
        attempt = batch.job_directory(job) / "attempt_corrupt"
        attempt.mkdir(parents=True)
        batch.save_new(attempt / "request.json", job)
        batch.save_new(attempt / "status.json", dict(status="complete"))
        path = self.result(attempt / "result", job)
        (path / "reference_sparse_examples.jsonl").write_text("{}\n")
        before = {p: p.read_bytes() for p in attempt.rglob("*") if p.is_file()}
        with patch.object(batch, "run_child", side_effect=self.fake_child):
            self.assertEqual(self.invoke(), 0)
        self.assertEqual(len(self.calls), 6)
        for p, content in before.items():
            self.assertEqual(p.read_bytes(), content)

    def test_window_pairing_and_case_identity_rejected(self):
        for filename, mutate in (
                ("reference_sparse_windows.json", lambda d: d[0].update(token_ids_sha256="f" * 64)),
                ("cases.json", lambda d: d[0].update(example_index=999)),
                ("reference_sparse_attention_calls.json", lambda d: d.pop())):
            with self.subTest(filename=filename):
                path = self.result(self.folder / filename)
                doc = batch.read_json(path / filename)
                mutate(doc)
                (path / filename).write_text(json.dumps(doc))
                with self.assertRaises(ValueError):
                    batch.verify_result(path, self.jobs[0]["config"], self.manifest)
    def test_invalid_cli_options_rejected(self):
        for options in (["--keep-fraction", "nan"], ["--keep-fraction", "0.75", "0.75"],
                        ["--keep-fraction", "1"], ["--local-blocks", "0"], ["--gpu", "0,1"],
                        ["--batch-name", "../experiments"], ["--model", batch.INITIAL_MODELS[0]]):
            with self.subTest(options=options):
                self.assertEqual(self.invoke(options), 2)
        for options in (["--backend", "flex"], ["--context", "8192"], ["--reference-policy", "best"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                batch.main(self.argv + options)

    def test_global_lock_rejects_concurrent_campaign(self):
        with batch.campaign_lock():
            with self.assertRaisesRegex(ValueError, "Another batch"):
                with batch.campaign_lock():
                    self.fail("concurrent lock acquired")

    def test_child_gpu_and_lock_forwarding(self):
        with patch.object(batch.subprocess, "Popen") as popen:
            popen.return_value.wait.return_value = 0
            popen.return_value.poll.return_value = 0
            self.assertEqual(batch.run_child(["fixture"], self.folder / "stdout.log", "2", 99), 0)
            kwargs = popen.call_args.kwargs
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2")
            self.assertEqual(kwargs["pass_fds"], (99,))
            self.assertTrue(kwargs["start_new_session"])

    def test_interruption_waits_for_child_to_stop(self):
        with patch.object(batch.subprocess, "Popen") as popen, patch.object(batch.os, "killpg") as kill:
            popen.return_value.wait.side_effect = [KeyboardInterrupt, 0]
            popen.return_value.poll.return_value = None
            popen.return_value.pid = 12345
            with self.assertRaises(KeyboardInterrupt):
                batch.run_child(["fixture"], self.folder / "stdout.log", "0", 99)
            kill.assert_called_once_with(12345, batch.signal.SIGTERM)
            self.assertEqual(popen.return_value.wait.call_count, 2)

    def test_symlink_to_archive_cannot_receive_outputs(self):
        batch.OUTPUT.parent.mkdir(parents=True)
        batch.OUTPUT.symlink_to(ROOT / "experiments", target_is_directory=True)
        self.assertEqual(self.invoke(["--dry-run"]), 2)

    def test_continue_on_error_records_failed_and_pending_status(self):
        with patch.object(batch, "run_child", return_value=1) as child:
            self.assertEqual(self.invoke(["--continue-on-error"]), 1)
        self.assertEqual(child.call_count, 6)
        csv_path = next(batch.OUTPUT.glob("batches/test/invocation_*/results.csv"))
        with csv_path.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(all(r["status"] == "failed" and not r["sparse_ppl"] for r in rows))


if __name__ == "__main__":
    unittest.main()
