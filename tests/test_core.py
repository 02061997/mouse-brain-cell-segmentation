from cellseg.core import (
    CATEGORIES,
    MODEL_CONFIGS,
    PUBLIC_DATASETS,
    coco_record,
    evaluate_instances,
    instance_overlay,
    reference_segment,
    synthetic_image,
)


def test_paper_dataset_schema_and_models():
    image, truth = synthetic_image(size=96)
    record = coco_record(1, image, truth)
    assert len(CATEGORIES) == 6
    assert len(MODEL_CONFIGS) == 6
    assert len(record["categories"]) == 6
    assert all(annotation["area"] > 0 for annotation in record["annotations"])


def test_reference_instance_metrics_are_bounded():
    image, truth = synthetic_image(size=96)
    metrics = evaluate_instances(truth, reference_segment(image))
    assert all(0 <= value <= 100 for value in metrics.values())
    overlay = instance_overlay(truth, image.shape[:2])
    assert overlay.shape == image.shape
    assert overlay.max() <= 1
    assert overlay.sum() > 0


def test_exact_paper_configuration_examples():
    assert MODEL_CONFIGS["mask_rcnn"]["iterations"] == 9150
    assert MODEL_CONFIGS["mask2former"]["iterations"] == 12200
    assert MODEL_CONFIGS["fastinst"]["evaluation_iteration"] == 4575


def test_public_dataset_extension_is_license_verified():
    bbbc = PUBLIC_DATASETS["BBBC038v1"]
    assert bbbc["license"] == "CC0"
    assert bbbc["url"].startswith("https://bbbc.broadinstitute.org/")
    assert not bbbc["included"]
