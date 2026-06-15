# Results

## Published paper results

Mask2Former reports the highest overall AP at 30.2 and AR10 at 34.1.
CenterMask2 is the strongest non-transformer method with AP 25.2 and AR10 31.0.
MaskDINO has the highest AP50 at 51.7. Complete AP, size-stratified AP, and AR
tables are stored in `reports/latest/statistical_tests.json`.

## Local reference results

The June 15, 2026 run evaluates a simple connected-component reference method
on deterministic six-class synthetic fluorescence images. It obtains mean
synthetic AP 42.18 and AP50 50.64.

Those values are **not comparable** to the paper's microscopy results because
the fixture is generated and substantially simpler. Training and evaluation of
the six deep models on the private 700/200/150 split remain `NOT_RUN` and are
recorded explicitly in `reports/latest/statistical_tests.json`.

The latest qualitative figure now uses colored instance overlays for the
synthetic ground truth and reference predictions. A BBBC038v1 public transfer
benchmark is documented in `DATA.md` but remains `NOT_RUN` in the committed
results.
