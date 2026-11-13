"""Tests for predefined racing track layouts."""

import numpy as np
import pytest

from f1_marl.simulation.tracks import (
    create_oval_track,
    create_technical_track,
)


@pytest.mark.parametrize(
    "factory",
    [create_oval_track, create_technical_track],
)
def test_track_is_closed(factory):
    track = factory()

    assert track.closed is True


@pytest.mark.parametrize(
    "factory",
    [create_oval_track, create_technical_track],
)
def test_track_has_positive_length(factory):
    track = factory()

    assert track.length_m > 0


@pytest.mark.parametrize(
    "factory",
    [create_oval_track, create_technical_track],
)
def test_track_has_expected_half_width(factory):
    track = factory(track_half_width_m=10.0)

    assert track.half_width_m == 10.0


@pytest.mark.parametrize(
    "factory",
    [create_oval_track, create_technical_track],
)
def test_track_centerline_has_expected_number_of_points(factory):
    track = factory(num_points=100)

    assert track.centerline.shape == (100, 2)


@pytest.mark.parametrize(
    "factory",
    [create_oval_track, create_technical_track],
)
def test_track_centerline_contains_only_finite_coordinates(factory):
    track = factory()

    assert np.isfinite(track.centerline).all()


@pytest.mark.parametrize(
    "factory",
    [create_oval_track, create_technical_track],
)
def test_track_sampling_wraps_around(factory):
    track = factory()

    start = track.sample(0.0)
    wrapped = track.sample(track.length_m)

    assert wrapped.x == pytest.approx(start.x)
    assert wrapped.y == pytest.approx(start.y)


@pytest.mark.parametrize(
    "factory",
    [create_oval_track, create_technical_track],
)
def test_track_projection_returns_valid_coordinates(factory):
    track = factory()

    sample = track.sample(track.length_m * 0.25)
    projected = track.project(sample.x, sample.y)

    assert projected.s >= 0.0
    assert projected.s < track.length_m
    assert projected.d == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize(
    "factory",
    [create_oval_track, create_technical_track],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {"semi_major_axis_m": 0.0},
        {"semi_minor_axis_m": -10.0},
        {"track_half_width_m": 0.0},
        {"num_points": 7},
        {"num_points": 10.5},
    ],
)
def test_invalid_track_parameters_are_rejected(factory, parameters):
    with pytest.raises(ValueError):
        factory(**parameters)


@pytest.mark.parametrize(
    "factory",
    [create_oval_track, create_technical_track],
)
def test_custom_track_dimensions(factory):
    track = factory(
        semi_major_axis_m=100.0,
        semi_minor_axis_m=50.0,
        track_half_width_m=7.0,
        num_points=120,
    )

    assert track.centerline.shape == (120, 2)
    assert track.half_width_m == 7.0
    assert track.closed is True
