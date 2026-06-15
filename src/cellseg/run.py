import argparse

import matplotlib.pyplot as plt
import pandas as pd

from .artifacts import environment, output_dir, publish_latest, save
from .core import (
    MODEL_CONFIGS,
    PUBLISHED_COLUMNS,
    PUBLISHED_RESULTS,
    PUBLIC_DATASETS,
    coco_record,
    evaluate_instances,
    instance_overlay,
    reference_segment,
    synthetic_image,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = output_dir(args.smoke)
    seeds = range(3 if args.smoke else 30)
    rows = []
    example = None
    for seed in seeds:
        image, truth = synthetic_image(seed=seed)
        prediction = reference_segment(image)
        rows.append({"seed": seed, **evaluate_instances(truth, prediction)})
        if seed == 0:
            example = image, truth, prediction
            save(out / "synthetic_coco.json", coco_record(1, image, truth))
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "predictions.parquet", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    axes[0].imshow(example[0])
    axes[0].set_title("Synthetic merged channels")
    for ax, instances, title in zip(
        axes[1:], example[1:], ["Ground-truth instance overlay", "Reference prediction overlay"]
    ):
        ax.imshow(instance_overlay(instances, example[0].shape[:2]))
        ax.set_title(title)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "qualitative_example.png", dpi=180)
    plt.close(fig)
    save(out / "metrics.json", frame.drop(columns="seed").agg(["mean", "std"]).to_dict())
    save(
        out / "statistical_tests.json",
        {
            "published_results": {
                model: dict(zip(PUBLISHED_COLUMNS, values, strict=True))
                for model, values in PUBLISHED_RESULTS.items()
            },
            "published_results_reproduced": False,
            "local_reference_results": frame.drop(columns="seed").agg(["mean", "std"]).to_dict(),
            "not_run": [
                {
                    "experiment": "Six deep-model training/evaluation on private 700/200/150 split",
                    "reason": "Private microscopy images and trained checkpoints are not redistributed.",
                },
                {
                    "experiment": "BBBC038v1 public microscopy transfer evaluation",
                    "reason": "Legally reusable CC0 dataset path selected, but transfer evaluation was not run in the latest committed results.",
                },
            ],
        },
    )
    save(out / "environment.json", environment())
    save(
        out / "data_manifest.json",
        {
            "paper_dataset": {"images": 1050, "train": 700, "validation": 200, "test": 150},
            "paper_dataset_included": False,
            "local_source": "deterministic six-class synthetic fluorescence fixture",
            "local_images": len(frame),
            "optional_public_dataset_extension": PUBLIC_DATASETS["BBBC038v1"],
        },
    )
    save(out / "config.yaml", {"paper_model_configs": MODEL_CONFIGS})
    (out / "run.log").write_text("completed\n")
    if not args.smoke:
        publish_latest(out)
    print(out)


if __name__ == "__main__":
    main()
