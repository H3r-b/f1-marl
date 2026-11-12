"""Tests for track geometry and Frenet coordinate transformations."""

import pytest

from f1_marl.simulation.track import Track


@pytest.fixture
def straight_track() -> Track:
    return Track(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
        ],
        closed=False,
    )


@pytest.fixture
def square_track() -> Track:
    return Track(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ],
        closed=True,
    )


def test_straight_track_length(straight_track: Track) -> None:
    assert straight_track.length_m == pytest.approx(20.0)


def test_track_sample_position(straight_track: Track) -> None:
    sample = straight_track.sample(5.0)

    assert sample.x == pytest.approx(5.0)
    assert sample.y == pytest.approx(0.0)
    assert sample.s == pytest.approx(5.0)


def test_track_sample_direction(straight_track: Track) -> None:
    sample = straight_track.sample(5.0)

    assert sample.tangent_x == pytest.approx(1.0)
    assert sample.tangent_y == pytest.approx(0.0)

    assert sample.normal_x == pytest.approx(0.0)
    assert sample.normal_y == pytest.approx(1.0)


def test_projection_positive_lateral_offset(
    straight_track: Track,
) -> None:
    result = straight_track.project(5.0, 2.0)

    assert result.s == pytest.approx(5.0)
    assert result.d == pytest.approx(2.0)


def test_projection_negative_lateral_offset(
    straight_track: Track,
) -> None:
    result = straight_track.project(5.0, -3.0)

    assert result.s == pytest.approx(5.0)
    assert result.d == pytest.approx(-3.0)


def test_open_track_clamps_sample_position(
    straight_track: Track,
) -> None:
    sample = straight_track.sample(100.0)

    assert sample.s == pytest.approx(20.0)
    assert sample.x == pytest.approx(20.0)


def test_closed_track_length(square_track: Track) -> None:
    assert square_track.length_m == pytest.approx(40.0)


def test_closed_track_wraps_position(
    square_track: Track,
) -> None:
    sample = square_track.sample(40.0)

    assert sample.s == pytest.approx(0.0)
    assert sample.x == pytest.approx(0.0)
    assert sample.y == pytest.approx(0.0)


def test_closed_track_projection(square_track: Track) -> None:
    result = square_track.project(-2.0, 5.0)

    assert result.s == pytest.approx(35.0)
    assert result.d == pytest.approx(-2.0)


def test_invalid_track_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        Track([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]])


def test_track_rejects_zero_length_segment() -> None:
    with pytest.raises(ValueError, match="zero-length"):
        Track(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [10.0, 0.0],
            ]
        )


def test_track_rejects_non_finite_coordinates() -> None:
    with pytest.raises(ValueError, match="finite"):
        Track(
            [
                [0.0, 0.0],
                [float("nan"), 1.0],
            ]
        )


def test_track_rejects_invalid_width() -> None:
    with pytest.raises(ValueError, match="half-width"):
        Track(
            [[0.0, 0.0], [10.0, 0.0]],
            half_width_m=0.0,
        )
