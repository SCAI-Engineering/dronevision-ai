import pytest

from bench.quant_split import evenly_spaced_indices, validate_split


def test_calibration_selection_covers_recording_without_duplicates():
    selected = evenly_spaced_indices(337, 64)
    assert len(selected) == len(set(selected)) == 64
    assert selected[0] == 0
    assert selected[-1] == 336
    assert set(b - a for a, b in zip(selected, selected[1:])) <= {5, 6}


def test_single_calibration_sample_is_first_frame():
    assert evenly_spaced_indices(337, 1) == [0]


@pytest.mark.parametrize("total,count", [(0, 1), (5, 0), (5, 6)])
def test_invalid_calibration_counts_fail(total, count):
    with pytest.raises(ValueError):
        evenly_spaced_indices(total, count)


def test_split_validator_rejects_overlap():
    split = {
        "corpus": {"frame_sets": 3, "cameras": ["a", "b"]},
        "calibration": {
            "frame_set_count": 2,
            "image_count": 4,
            "frame_set_ids": [0, 1],
        },
        "validation_holdout": {
            "frame_set_count": 2,
            "image_count": 4,
            "frame_set_ids": [1, 2],
        },
    }
    with pytest.raises(ValueError, match="overlap"):
        validate_split(split)
