import compare_qwen38_predictions as comparison


def fixture_ground_truth():
    return {
        "images": [
            {"id": 10, "file_name": "a.jpg", "width": 100, "height": 100},
            {"id": 20, "file_name": "b.jpg", "width": 100, "height": 100},
        ],
        "annotations": [
            {
                "id": 100,
                "image_id": 10,
                "category_id": 1,
                "bbox": [1, 2, 3, 4],
                "area": 12,
                "iscrowd": 0,
            },
            {
                "id": 200,
                "image_id": 20,
                "category_id": 1,
                "bbox": [5, 6, 7, 8],
                "area": 56,
                "iscrowd": 0,
            },
        ],
        "categories": [{"id": 1, "name": "object"}],
    }


def test_remap_sample_preserves_paired_duplicates():
    predictions = [
        {"image_id": 10, "category_id": 1, "bbox": [1, 2, 3, 4], "score": 1.0},
        {"image_id": 20, "category_id": 1, "bbox": [5, 6, 7, 8], "score": 1.0},
    ]
    ground_truth, remapped = comparison.remap_sample(
        fixture_ground_truth(), predictions, [20, 10, 20]
    )
    assert [image["id"] for image in ground_truth["images"]] == [1, 2, 3]
    assert [annotation["image_id"] for annotation in ground_truth["annotations"]] == [1, 2, 3]
    assert [annotation["id"] for annotation in ground_truth["annotations"]] == [1, 2, 3]
    assert [prediction["image_id"] for prediction in remapped] == [1, 2, 3]


def test_percentile_interpolates():
    assert comparison.percentile([0, 10], 0.25) == 2.5
    assert comparison.percentile([3, 1, 2], 0.5) == 2
