# Data Documentation

## Private Paper Dataset

The CISS 2025 paper used a private 1,050-image mouse-brain fluorescence
microscopy dataset split into 700 training, 200 validation, and 150 test
images. The images, annotations, and trained checkpoints are not redistributed
in this public companion.

## Included Public Fixture

The committed runnable experiment uses deterministic synthetic six-class
fluorescence-like images. It validates the COCO schema, AP/AR metric code,
qualitative overlays, and reconstruction artifacts, but it is not comparable to
the paper's private microscopy benchmark.

## Optional Public Extension

BBBC038v1, the 2018 Data Science Bowl nuclei segmentation image set from the
Broad Bioimage Benchmark Collection, is the selected legally reusable public
extension path:

- Dataset page: https://bbbc.broadinstitute.org/BBBC038
- Source: Broad Bioimage Benchmark Collection
- License status verified June 15, 2026: images and collection copyright are
  listed as CC0/public-domain waiver on the BBBC038 page.
- Intended use: future public fluorescence nuclei instance-segmentation
  transfer benchmark.

This dataset is not mouse-brain tissue and is not included in the current
repository. Any future transfer run should report it separately from the
private paper results and from the deterministic synthetic fixture.
