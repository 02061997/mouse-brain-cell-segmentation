import argparse

import matplotlib.pyplot as plt
import pandas as pd

from .artifacts import environment, output_dir, save
from .core import metrics, segment, synthetic_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = output_dir(args.smoke)
    seeds = range(3) if args.smoke else range(40)
    rows = []
    example = None
    for seed in seeds:
        image, truth = synthetic_image(seed=seed)
        prediction = segment(image)
        rows.append({"seed": seed, **metrics(truth, prediction)})
        if seed == 0:
            example = image, truth, prediction
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "predictions.parquet", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, value, title in zip(
        axes, example, ["Synthetic fluorescence", "Ground truth", "Prediction"]
    ):
        ax.imshow(value, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "qualitative_example.png", dpi=180)
    plt.close(fig)
    summary = frame.drop(columns="seed").agg(["mean", "std"]).to_dict()
    save(out / "metrics.json", summary)
    save(out / "statistical_tests.json", {"images": len(frame)})
    save(out / "environment.json", environment())
    save(
        out / "data_manifest.json",
        {"source": "deterministic synthetic fluorescence generator", "images": len(frame)},
    )
    save(out / "config.yaml", {"image_size": 128, "nominal_cells": 18})
    (out / "run.log").write_text("completed\n")
    print(out)


if __name__ == "__main__":
    main()
