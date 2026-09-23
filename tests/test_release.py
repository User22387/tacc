from __future__ import annotations

import unittest
from pathlib import Path

import torch

from tacc import (
    LagerNVSTACC,
    ProgressiveTreeCarrier,
    TACCConfig,
    allocate_quotas,
    build_progressive_token_tree,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints/tacc_lagernvs_v3.pt"


class TACCReleaseTest(unittest.TestCase):
    def test_checkpoint_loads_strictly(self):
        module = LagerNVSTACC.from_checkpoint(CHECKPOINT)
        self.assertEqual(module.config, TACCConfig())
        self.assertEqual(sum(p.numel() for p in module.parameters()), 302785)

    def test_leaf_is_exact_anchor(self):
        carrier = ProgressiveTreeCarrier(token_dim=8, hidden_dim=4)
        anchor = torch.randn(3, 8)
        output = carrier(
            anchor,
            torch.randn_like(anchor),
            torch.rand(3),
            torch.ones(3),
            torch.ones(3),
            torch.ones(3),
        )
        torch.testing.assert_close(output, anchor, rtol=0, atol=0)

    def test_allocation_is_exact(self):
        generator = torch.Generator().manual_seed(7)
        relevance = torch.randn(1, 4, 60, generator=generator)
        quotas, _ = allocate_quotas(relevance, 13690, 1369)
        self.assertTrue(bool((quotas.sum(dim=-1) == 13690).all()))
        self.assertTrue(bool((quotas.max(dim=-1).values == 1369).all()))
        self.assertTrue(bool((quotas >= 0).all()))
        self.assertTrue(bool((quotas <= 1369).all()))

    def test_hierarchy_readout_shape_and_zero_quota(self):
        generator = torch.Generator().manual_seed(11)
        source_tokens = torch.randn(1, 2, 9, 8, generator=generator)
        carrier = ProgressiveTreeCarrier(token_dim=8, hidden_dim=4)
        tree = build_progressive_token_tree(source_tokens, grid_size=3)
        tree.precompute_frontier_tables()
        tree.precompute_carriers(carrier)
        quotas = torch.tensor([[[9, 0], [4, 5]]])
        tokens, metadata = tree.gather(quotas)
        self.assertEqual(tokens.shape, (2, 9, 8))
        self.assertEqual(metadata["tokens_after"], 9)
        torch.testing.assert_close(tokens[0], source_tokens[0, 0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
