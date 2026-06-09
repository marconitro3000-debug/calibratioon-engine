import pytest

from calibration_engine import CalibrationEngine, SearchSpace
from calibration_engine.optimizers import expand_grid, sample_random


def test_search_space_validates_and_serializes():
    space = SearchSpace.from_dict({"a": [1, 2], "b": (0.0, 1.0)})

    assert space.names == ["a", "b"]
    assert space.grid_size == 14
    assert space.to_dict()["parameters"][1]["bounds"] == [0.0, 1.0]


def test_search_space_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        SearchSpace.from_dict({})
    with pytest.raises(ValueError):
        SearchSpace.from_dict({"a": []})
    with pytest.raises(ValueError):
        SearchSpace.from_dict({"a": (2.0, 1.0)})


def test_optimizers_accept_search_space():
    space = SearchSpace.from_dict({"a": [1, 2], "b": [3]})

    assert list(expand_grid(space)) == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]
    assert len(list(sample_random(space, 3, seed=1))) == 3


def test_engine_records_search_space_manifest():
    result = CalibrationEngine().calibrate(
        lambda params, data: params["a"],
        SearchSpace.from_dict({"a": [1, 2]}),
        optimizer="grid",
    )

    assert result.metadata["search_space"]["grid_size"] == 2
