import hashlib
from pathlib import Path
import sys
import unittest

import torch

from celerity_sparse import CeleritySparseAdapter, SparseSettings
from model_fixture import model_for

LB = Path(__file__).resolve().parents[2] / "Perplexity_LongBench"
sys.path.insert(0, str(LB))
from longbench_perplexity import answer_conditional_ppl


class TinyTokenizer:
    bos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [3 + ord(ch) % 90 for ch in text]}


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @torch.no_grad()
    def test_disabled_exact_and_reversible(self):
        model = model_for()
        ids = torch.arange(269).remainder(97)[None]
        original = model(ids).logits
        params = [(name, id(p), p._version) for name, p in model.named_parameters()]
        config = model.config.to_dict()
        adapter = CeleritySparseAdapter(model)
        torch.testing.assert_close(model(ids).logits, original, atol=0, rtol=0)
        adapter.enabled = True
        model(ids)
        adapter.enabled = False
        torch.testing.assert_close(model(ids).logits, original, atol=0, rtol=0)
        adapter.remove()
        torch.testing.assert_close(model(ids).logits, original, atol=0, rtol=0)
        self.assertEqual(params, [(name, id(p), p._version) for name, p in model.named_parameters()])
        self.assertEqual(config, model.config.to_dict())

    @torch.no_grad()
    def test_full_reference_all_three_layouts_fp32_bf16(self):
        for heads, dim in ((10, 64), (14, 64), (9, 128)):
            for dtype in (torch.float32, torch.bfloat16):
                model = model_for(heads, dim, dtype=dtype)
                ids = torch.arange(259).remainder(97)[None]
                original = model(ids).logits
                adapter = CeleritySparseAdapter(model, SparseSettings(keep_fraction=1))
                adapter.enabled = True
                torch.testing.assert_close(model(ids).logits, original, atol=0, rtol=0)
                self.assertEqual(adapter.statistics()["calls_by_backend"], {"reference": 2})
                self.assertEqual(adapter.statistics()["causal_pair_sparsity"], 0)

    @torch.no_grad()
    def test_causal_logits_within_across_blocks(self):
        model = model_for()
        adapter = CeleritySparseAdapter(model)
        adapter.enabled = True
        ids = torch.arange(517).remainder(97)[None]
        original = model(ids).logits
        for cut in (127, 128, 129, 255, 256, 300, 512):
            altered = ids.clone()
            altered[:, cut:] = (altered[:, cut:] + 19) % 97
            torch.testing.assert_close(model(altered).logits[:, :cut], original[:, :cut],
                                       atol=1e-6, rtol=1e-6)
        self.assertGreater(adapter.statistics()["calls_omitting_blocks"], 0)

    @torch.no_grad()
    def test_unsupported_requests_raise_no_fallback(self):
        model = model_for()
        adapter = CeleritySparseAdapter(model)
        adapter.enabled = True
        ids = torch.arange(129).remainder(97)[None]
        for kw in ({"attention_mask": torch.ones_like(ids)},
                   {"output_attentions": True}, {"position_ids": torch.arange(129)[None]},
                   {"past_key_values": ()}):
            with self.assertRaises(ValueError):
                model(ids, **kw)
        model.train()
        with self.assertRaises(ValueError):
            model(ids)
        model.eval()
        adapter.settings = SparseSettings(backend="flex")
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            model(ids)
        self.assertEqual(adapter.statistics()["dense_fallback_calls"], 0)
        self.assertEqual(adapter.statistics()["completed_attention_calls"], 0)
        self.assertTrue(adapter.statistics()["failures"])

    @torch.no_grad()
    def test_canonical_scorer_identical_windows_targets_and_nll(self):
        scorer_path = LB / "longbench_perplexity.py"
        before = hashlib.sha256(scorer_path.read_bytes()).hexdigest()
        model = model_for(layers=1)
        tok = TinyTokenizer()
        windows = []
        hook = model.register_forward_pre_hook(
            lambda m, a, kw: windows.append(kw["input_ids"].clone()), with_kwargs=True)
        arguments = dict(model=model, tokenizer=tok, prompt="abcdef" * 100,
                         answer="gold answer" * 31, max_length=389, stride=128,
                         use_chat_template=False)
        dense = answer_conditional_ppl(**arguments)
        original_windows = windows[:]
        self.assertEqual(dense[1], len(tok(arguments["answer"])["input_ids"]))
        self.assertEqual(len(original_windows), 3)
        windows.clear()
        adapter = CeleritySparseAdapter(model, SparseSettings(keep_fraction=1))
        adapter.enabled = True
        full = answer_conditional_ppl(**arguments)
        self.assertEqual(dense, full)
        for a, b in zip(original_windows, windows):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        windows.clear()
        adapter.settings = SparseSettings(keep_fraction=0.25)
        sparse = answer_conditional_ppl(**arguments)
        self.assertEqual(sparse[1], dense[1])
        self.assertEqual(len(original_windows), len(windows))
        for a, b in zip(original_windows, windows):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        self.assertGreater(adapter.statistics()["calls_omitting_blocks"], 0)
        self.assertEqual(before, hashlib.sha256(scorer_path.read_bytes()).hexdigest())
        hook.remove()


if __name__ == "__main__":
    unittest.main()
