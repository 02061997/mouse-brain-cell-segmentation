from cellseg.core import metrics, segment, synthetic_image


def test_segmentation_shapes_and_bounds():
    image, truth = synthetic_image(size=64, cells=8)
    prediction = segment(image)
    result = metrics(truth, prediction)
    assert prediction.shape == image.shape
    assert 0 <= result["dice"] <= 1
    assert 0 <= result["iou"] <= 1
