# Mouse Brain Cell Instance Segmentation

Public research companion for *Mouse Brain Cell Segmentation in Fluorescence
Microscopy Images* (CISS 2025).

The repository provides the six-class COCO data contract, exact configurations
for the paper's six CNN/transformer methods, COCO-style AP/AR evaluation
primitives, and a deterministic synthetic fluorescence fixture. It does not
redistribute the private 1,050-image dataset or trained model checkpoints.

`DATA.md` documents the optional public extension path: BBBC038v1 from the
Broad Bioimage Benchmark Collection, a CC0 fluorescence nuclei instance
segmentation dataset suitable for a future transfer benchmark.

```bash
uv sync
make test
make reproduce-smoke
make reproduce-results
```
