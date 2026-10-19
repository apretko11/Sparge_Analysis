import unittest
import torch

from celerity_sparse.attention import (SparseSettings, select_blocks, token_mask,
                                      reference_attention, make_block_mask, block_statistics)


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(47)

    def tensors(self, length=517, dim=64):
        q, k, v = [torch.randn(1, 3, length, dim) for _ in range(3)]
        slopes = torch.tensor([0.01, 0.03, 0.1])
        pos = torch.arange(length)
        bias = -slopes[:, None, None] * (pos[:, None] - pos[None, :]).abs()
        mask = torch.zeros(1, 1, length, length).masked_fill(
            pos[None, :] > pos[:, None], torch.finfo(q.dtype).min)
        return q, k, v, slopes, bias, mask

    def test_full_equation_both_head_dimensions(self):
        for dim in (64, 128):
            q, k, v, slopes, bias, mask = self.tensors(259, dim)
            blocks = select_blocks(q, k, slopes, SparseSettings(keep_fraction=1))
            expected = torch.softmax((q @ k.transpose(-1, -2) / dim + mask + bias).float(), -1) @ v
            actual = reference_attention(q, k, v, bias, mask, blocks)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
            wrong_scale = torch.softmax(q @ k.transpose(-1, -2) / dim**0.5 + mask + bias, -1) @ v
            no_bias = torch.softmax(q @ k.transpose(-1, -2) / dim + mask, -1) @ v
            self.assertGreater((actual - wrong_scale).abs().max().item(), 0.01)
            self.assertGreater((actual - no_bias).abs().max().item(), 0.01)

    def test_suffix_invariance_within_and_across_blocks(self):
        q, k, v, slopes, bias, mask = self.tensors()
        settings = SparseSettings(keep_fraction=0.5, local_blocks=2)
        blocks = select_blocks(q, k, slopes, settings)
        expected = reference_attention(q, k, v, bias, mask, blocks)
        for cut in (1, 127, 128, 129, 255, 256, 300, 512):
            changed = [x.clone() for x in (q, k, v)]
            for x in changed:
                x[:, :, cut:] = torch.randn_like(x[:, :, cut:]) * 9
            altered = select_blocks(changed[0], changed[1], slopes, settings)
            torch.testing.assert_close(blocks[:, :, :(cut + 127) // 128],
                                       altered[:, :, :(cut + 127) // 128], atol=0, rtol=0)
            actual = reference_attention(*changed, bias, mask, altered)
            torch.testing.assert_close(actual[:, :, :cut], expected[:, :, :cut], atol=0, rtol=0)

    def test_real_block_omission_local_retention_and_metadata(self):
        q, k, _, slopes, _, _ = self.tensors()
        blocks = select_blocks(q, k, slopes, SparseSettings(local_blocks=2))
        mask = token_mask(blocks, q.shape[-2])
        self.assertTrue(mask.any(-1).all())
        self.assertFalse(mask.triu(1).any())
        for row in range(blocks.shape[-2]):
            self.assertTrue(blocks[:, :, row, max(0, row-1):row+1].all())
        stats = block_statistics(blocks, q.shape[-2])
        self.assertLess(stats["retained_blocks"], stats["eligible_blocks"])
        self.assertEqual(stats["retained_causal_pairs"], int(mask.sum()))
        flex_mask = make_block_mask(blocks)
        torch.testing.assert_close(flex_mask.to_dense().bool(), blocks)
        self.assertEqual(tuple(flex_mask.BLOCK_SIZE), (128, 128))

    def test_direct_metadata_matches_create_block_mask(self):
        from torch.nn.attention.flex_attention import create_block_mask
        q, k, _, slopes, _, _ = self.tensors()
        blocks = select_blocks(q, k, slopes, SparseSettings())

        def selected_causal(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & blocks[b, h, q_idx // 128, kv_idx // 128]

        public = create_block_mask(selected_causal, B=1, H=3, Q_LEN=517, KV_LEN=517,
                                   device="cpu", BLOCK_SIZE=128)
        torch.testing.assert_close(public.to_dense(), make_block_mask(blocks).to_dense())

    def test_selector_is_content_dependent(self):
        q, k, _, slopes, _, _ = self.tensors(512)
        q.fill_(1)
        k.zero_()
        slopes.zero_()
        settings = SparseSettings(keep_fraction=0.25)
        k[:, :, :128] = 4
        a = select_blocks(q, k, slopes, settings)
        k[:, :, :128] = 0
        k[:, :, 128:256] = 4
        b = select_blocks(q, k, slopes, settings)
        self.assertTrue(a[:, :, 3, 0].all())
        self.assertTrue(b[:, :, 3, 1].all())
        self.assertFalse(torch.equal(a, b))

    def test_short_sequences_and_odd_tail(self):
        for length in (1, 17, 127, 128, 129, 257):
            q, k, v, slopes, bias, mask = self.tensors(length)
            blocks = select_blocks(q, k, slopes, SparseSettings())
            self.assertTrue(torch.isfinite(reference_attention(q, k, v, bias, mask, blocks)).all())


if __name__ == "__main__":
    unittest.main()
