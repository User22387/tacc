# TACC for LagerNVS

This anonymous artifact contains the TACC inference module and the LagerNVS
Carrier checkpoint used for the DL3DV experiments. It is intentionally limited
to the proposed method: training code and comparison-method implementations are
not included.

## Contents

```text
tacc/
  allocation.py       target-aware exact-budget allocation
  carrier.py          learned regional Carrier module
  hierarchy.py        progressive spatial hierarchy and frontier readout
  lagernvs.py         LagerNVS inference adapter
checkpoints/
  tacc_lagernvs_v3.pt inference-only Carrier checkpoint
configs/
  lagernvs_tacc_v3.json
examples/
  lagernvs_integration.py
tests/
  test_release.py
```

## Requirements

Use the environment from the official LagerNVS repository. This package was
validated against the public LagerNVS API at commit
`35e81bed35672309e458fa5d1f33787ac7b187e2`
with PyTorch 2.8. The official frozen LagerNVS DL3DV 2--6 view, 256px model is
required separately.

## Insertion point

TACC operates on the per-view scene tokens produced by
`lagernvs_model.reconstructor`:

```python
source_tokens = lagernvs_model.reconstructor(source_images, source_camera_tokens)
source_cache = tacc.build_source_cache(source_tokens)
prediction = tacc.render_targets(
    lagernvs_model.renderer,
    source_cache,
    source_plucker_rays,
    target_plucker_rays,
    equivalent_views=10,
    target_batch_size=4,
)
```

For 256px LagerNVS, each source view has `37 x 37 = 1369` tokens. A budget of
`A=10` therefore exposes exactly `13,690` source tokens to the frozen renderer.
Target rays are not compressed or modified.

`build_source_cache` performs the source-only hierarchy and Carrier computation
once. The cache can be reused across all target views. At target time, TACC
computes per-view quotas, reads the corresponding hierarchy frontiers, and
passes the resulting tokens to the original LagerNVS renderer.

See `examples/lagernvs_integration.py` for the complete adapter call.

## Checkpoint

The included checkpoint is inference-only. It contains the Carrier state dict
and public configuration, but no optimizer state, training paths, scene lists,
or identifying metadata. It has 302,785 parameters and is compatible with source
view counts 60, 80, 100, and 120 at the A10 budget.

## Verification

Run the standalone tests from this directory:

```bash
python -m unittest tests.test_release -v
```

The tests check checkpoint loading, leaf identity, exact token budgets, Top-1
full-view preservation, zero-quota skipping, and the LagerNVS renderer-facing
tensor shape.

## License

This LagerNVS-integrated artifact is released under the included FAIR
Noncommercial Research License. See `THIRD_PARTY_NOTICE.md` for dependencies
that must be obtained separately.
