import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import cpu_audit as audit
import gpu_probe


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_pinned_source_integrity_and_api_arguments(self):
        self.assertEqual(audit.source_findings(), dict(attn_mask_consumed=False,
                         ampere_causal_argument=False, selector_receives_scale=False))
        header = (HERE / "upstream/csrc/qattn/attn_cuda.h").read_text()
        signature = header.split("qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(")[1].split(");")[0]
        self.assertEqual(signature.split(",")[10].strip(), "int is_causal")

    def test_actual_upstream_python_selector_matches_simple_topk_case(self):
        q, _, k, _ = audit.query_witness()
        block_map, *_ = audit.select(q, k)
        # Hand-computed row scores [4, 0, -4, 0]/sqrt(64) for the second query block.
        self.assertEqual(block_map[0, 0, 1].tolist(), [True, False, False, False])
        self.assertEqual(block_map.shape, (1, 1, 2, 4))

    def test_future_query_changes_past_selection_even_with_final_causal_mask(self):
        q, changed, k, v = audit.query_witness()
        self.assertTrue(torch.equal(q[..., :129, :], changed[..., :129, :]))
        first, *_ = audit.select(q, k)
        second, *_ = audit.select(changed, k)
        self.assertEqual(first[0, 0, 1].nonzero().flatten().tolist(), [0])
        self.assertEqual(second[0, 0, 1].nonzero().flatten().tolist(), [1])
        original = audit.dense_oracle(q, k, v, block_map=first)
        altered = audit.dense_oracle(changed, k, v, block_map=second)
        self.assertEqual(float((original[..., 128, :] - altered[..., 128, :]).abs().max()), 1.0)

    def test_final_causal_mask_alone_blocks_future_values(self):
        q, _, k, v = audit.query_witness()
        changed = v.clone()
        changed[..., 129:, :] = 123
        first = audit.dense_oracle(q, k, v)
        second = audit.dense_oracle(q, k, changed)
        torch.testing.assert_close(first[..., :129, :], second[..., :129, :], atol=0, rtol=0)

    def test_coarse_causal_map_allows_future_tokens(self):
        q = torch.ones(1, 1, 128, 64)
        block_map, *_ = audit.select(q, q, topk=1, smooth=False)
        self.assertTrue(block_map.all())
        v = torch.zeros_like(q)
        v[..., 1:, :] = 1
        leaked = audit.dense_oracle(q, q, v, block_map=block_map, causal=False)
        causal = audit.dense_oracle(q, q, v, block_map=block_map, causal=True)
        self.assertGreater(float(leaked[0, 0, 0, 0]), 0.99)
        self.assertEqual(float(causal[0, 0, 0, 0]), 0)

    def test_sequence_length_quantization_and_smoothing_dependencies(self):
        result = audit.findings()
        self.assertEqual(result["length_dependency"], dict(selected_at_length_256=1, selected_at_length_512=2))
        self.assertGreater(result["q_quantization_future_dependency"]["earlier_row_dequantized_max_abs_change"], 0.1)
        self.assertEqual(result["key_centering_future_dependency"], dict(old_block_gate_before=False, old_block_gate_after=True))

    def test_alibi_key_bias_equivalence_and_one_over_d_scaling(self):
        torch.manual_seed(17)
        for dim in (64, 128):
            q, k, v = [torch.randn(1, 2, 33, dim, dtype=torch.float64) for _ in range(3)]
            slopes = torch.tensor([0.03, 0.4], dtype=torch.float64)
            bias = audit.alibi(slopes, 33)
            expected = audit.dense_oracle(q, k, v, bias)
            key_bias = slopes[:, None, None] * torch.arange(33, dtype=torch.float64)[None, None, :]
            equivalent = audit.dense_oracle(q, k, v, key_bias)
            torch.testing.assert_close(expected, equivalent, atol=1e-14, rtol=1e-14)
            # API scale=1/d is necessary: default 1/sqrt(d) gives different probabilities.
            wrong_scale = audit.dense_oracle(q * math.sqrt(dim), k, v, bias)
            self.assertGreater(float((expected - wrong_scale).abs().max()), 0.01)

    def test_alibi_changes_historical_block_ranking(self):
        q = torch.zeros(64, dtype=torch.float64)
        q[0] = 1
        keys = torch.zeros(2, 64, dtype=torch.float64)
        keys[:, 0] = torch.tensor([64.0, 32.0])
        content = keys @ q / 64
        # Query block [128,255], completed key blocks [0,63], [64,127].
        means = torch.tensor([31.5, 95.5], dtype=torch.float64)
        biased = content + 0.02 * (means - 191.5)
        self.assertEqual(int(content.argmax()), 0)
        self.assertEqual(int(biased.argmax()), 1)
        for block, mean in enumerate(means):
            explicit = audit.alibi(torch.tensor([0.02], dtype=torch.float64), 256)[0, 128:256, block*64:(block+1)*64].mean()
            torch.testing.assert_close(explicit, 0.02 * (mean - 191.5))

    def test_global_key_centering_is_invariant_before_quantization(self):
        torch.manual_seed(20)
        q, k, v = [torch.randn(1, 1, 33, 64, dtype=torch.float64) for _ in range(3)]
        centered = k - k.mean(-2, keepdim=True)
        torch.testing.assert_close(audit.dense_oracle(q, k, v), audit.dense_oracle(q, centered, v),
                                   atol=1e-14, rtol=1e-14)

    def test_default_similarity_gate_passes_nonzero_complete_blocks(self):
        torch.manual_seed(19)
        x = torch.randn(1, 2, 128, 64)
        _, gate, *_ = audit.pool_quant_cpu(x, None, 128, torch.tensor([-0.1, -0.1]))
        self.assertTrue(gate.all())

    def test_heterogeneous_query_gate_keeps_every_causally_eligible_block(self):
        q, _, k, _ = audit.query_witness()
        q[..., 128:192, 0], q[..., 192:, 0] = 1, -1
        block_map, *_ = audit.select(q, k, threshold=0.6)
        self.assertTrue(block_map[0, 0, 1].all())

    def test_partial_block_probe_is_defined_but_does_not_certify_triton(self):
        x = torch.ones(1, 1, 129, 64)
        x[..., -1, :] = 3
        pooled, gates, quant, scales = audit.pool_quant_cpu(x, None, 128, torch.tensor([-0.1]))
        self.assertEqual(pooled[0, 0, 1, 0].item(), 3)
        self.assertTrue(gates.all())
        self.assertEqual(quant.shape, x.shape)
        self.assertTrue(torch.isfinite(scales).all())
        # Upstream's normalized zero padding can produce NaN, even with other=0.
        zero = torch.zeros(64)
        self.assertTrue(torch.isnan(zero / zero.norm()).all())

    def test_full_selection_still_contains_quantization_error(self):
        torch.manual_seed(21)
        q = torch.randn(1, 1, 128, 64)
        blocks, quant, scale, *_ = audit.select(q, q, topk=1, smooth=False)
        self.assertTrue(blocks.all())
        self.assertGreater(float((quant.float() * scale[..., None] - q).abs().max()), 1e-3)

    def test_gpu_probe_rejects_missing_package_without_installing(self):
        with patch.object(gpu_probe.importlib.util, "find_spec", return_value=None):
            with self.assertRaisesRegex(ValueError, "already be installed"):
                gpu_probe.inspect_installation(audit.PIN)

    def test_gpu_probe_rejects_unpinned_installation_before_loading_model(self):
        spec = types.SimpleNamespace(origin=str(HERE / "upstream/spas_sage_attn/__init__.py"))
        with patch.object(gpu_probe.importlib.util, "find_spec", return_value=spec), \
             patch.object(gpu_probe.subprocess, "check_output", return_value="f" * 40):
            with self.assertRaisesRegex(ValueError, "not the audited revision"):
                gpu_probe.inspect_installation(audit.PIN)


if __name__ == "__main__":
    unittest.main()
