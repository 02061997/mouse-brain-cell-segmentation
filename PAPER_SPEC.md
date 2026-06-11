# Paper Specification

Source: *Mouse Brain Cell Segmentation in Fluorescence Microscopy Images*,
CISS 2025, DOI `10.1109/CISS64860.2025.10944731`.

## Implemented specification

- COCO-style instance annotations for six cerebellar structures.
- Paper split metadata: 700 train, 200 validation, and 150 test images.
- Exact reported configurations for Mask R-CNN, CenterMask2, YOLACT++,
  Mask2Former, MaskDINO, and FastInst.
- COCO-style IoU-threshold AP/AR evaluation primitives.
- Published AP/AP50/AP75/size and AR tables stored separately.

The private 1,050-image microscopy dataset and trained checkpoints are not
distributed. A deterministic six-class fluorescence fixture validates the data
contract and evaluator, but its metrics are not biological reproduction
results.
