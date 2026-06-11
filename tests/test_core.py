from cellseg.core import (
    CATEGORIES,
    MODEL_CONFIGS,
    coco_record,
    evaluate_instances,
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


def test_exact_paper_configuration_examples():
    assert MODEL_CONFIGS["mask_rcnn"]["iterations"] == 9150
    assert MODEL_CONFIGS["mask2former"]["iterations"] == 12200
    assert MODEL_CONFIGS["fastinst"]["evaluation_iteration"] == 4575
