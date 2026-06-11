# Verified Results

## Provenance

This is a reconstructed reference implementation evaluated on deterministic
synthetic microscopy images. It is not the original research implementation,
and no biological-performance claim is made.

## Local reproduction

`make reproduce-results` completed locally on June 11, 2026.

| Metric | Mean | Std. dev. |
|---|---:|---:|
| Dice | 0.907 | 0.013 |
| Iou | 0.831 | 0.022 |
| Precision | 0.976 | 0.005 |
| Predicted Objects | 12.650 | 1.494 |
| Recall | 0.848 | 0.022 |
| True Objects | 12.800 | 1.728 |

The benchmark primarily tests pipeline correctness, determinism, and metric
generation. Evaluation on a legally reusable public microscopy dataset remains
`NOT_RUN` pending dataset selection and paper review.
