"""Tests for vehicle-track interaction calculations."""

import math

import pytest

from f1_marl.simulation.track import Track
from f1_marl.simulation.track_interaction import (
    calculate_track_interaction,
)
from f1_marl.simulation.vehicle import VehicleState


@pytest.fixture
def straight_track() -> Track:
    return Track(
        [(0.0, 0.0), (100.0, 0.0)],
        half_width_m=5.0,
    )


def test_vehicle_on_track_centerline(straight_track: Track) -> None:
    vehicle = VehicleState(x=25.0, y=0.0, heading_rad=0.0)

    result = calculate_track_interaction(vehicle, straight_track)

    assert result.s == pytest.approx(25.0)
    assert result.d == pytest.approx(0.0)
    assert result.heading_error_rad == pytest.approx(0.0)
    assert result.is_on_track
    assert result.progress_fraction == pytest.approx(0.25)
    assert result.segment_index == 0


def test_vehicle_left_of_centerline(straight_track: Track) -> None:
    vehicle = VehicleState(x=25.0, y=2.0)

    result = calculate_track_interaction(vehicle, straight_track)

    assert result.d == pytest.approx(2.0)
    assert result.is_on_track


def test_vehicle_right_of_centerline(straight_track: Track) -> None:
    vehicle = VehicleState(x=25.0, y=-2.0)

    result = calculate_track_interaction(vehicle, straight_track)

    assert result.d == pytest.approx(-2.0)
    assert result.is_on_track


def test_vehicle_outside_track_boundaries(straight_track: Track) -> None:
    vehicle = VehicleState(x=25.0, y=6.0)

    result = calculate_track_interaction(vehicle, straight_track)

    assert result.d == pytest.approx(6.0)
    assert not result.is_on_track


def test_vehicle_exactly_on_track_boundary(straight_track: Track) -> None:
    vehicle = VehicleState(x=25.0, y=5.0)

    result = calculate_track_interaction(vehicle, straight_track)

    assert result.is_on_track


def test_heading_error_for_left_direction(
    straight_track: Track,
) -> None:
    vehicle = VehicleState(
        x=25.0,
        y=0.0,
        heading_rad=math.radians(30.0),
    )

    result = calculate_track_interaction(vehicle, straight_track)

    assert result.heading_error_rad == pytest.approx(math.radians(30.0))


def test_heading_error_for_right_direction(
    straight_track: Track,
) -> None:
    vehicle = VehicleState(
        x=25.0,
        y=0.0,
        heading_rad=math.radians(-30.0),
    )

    result = calculate_track_interaction(vehicle, straight_track)

    assert result.heading_error_rad == pytest.approx(math.radians(-30.0))


def test_heading_error_is_normalized(
    straight_track: Track,
) -> None:
    vehicle = VehicleState(
        x=25.0,
        y=0.0,
        heading_rad=3.0 * math.pi,
    )

    result = calculate_track_interaction(vehicle, straight_track)

    assert -math.pi <= result.heading_error_rad < math.pi
    assert result.heading_error_rad == pytest.approx(-math.pi)


def test_open_track_progress_at_endpoint(
    straight_track: Track,
) -> None:
    vehicle = VehicleState(x=100.0, y=0.0)

    result = calculate_track_interaction(vehicle, straight_track)

    assert result.s == pytest.approx(100.0)
    assert result.progress_fraction == pytest.approx(1.0)


def test_closed_track_progress_wraps() -> None:
    track = Track(
        [
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
        ],
        closed=True,
        half_width_m=2.0,
    )

    vehicle = VehicleState(x=0.0, y=5.0)

    result = calculate_track_interaction(vehicle, track)

    assert 0.0 <= result.progress_fraction < 1.0
    assert result.s == pytest.approx(35.0)
    assert result.progress_fraction == pytest.approx(0.875)


def test_heading_error_on_vertical_track() -> None:
    track = Track(
        [(0.0, 0.0), (0.0, 100.0)],
    )

    vehicle = VehicleState(
        x=0.0,
        y=25.0,
        heading_rad=math.pi / 2.0,
    )

    result = calculate_track_interaction(vehicle, track)

    assert result.heading_error_rad == pytest.approx(0.0)
