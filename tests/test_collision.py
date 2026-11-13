"""Tests for vehicle collision geometry and event detection."""

import math

import pytest

from f1_marl.simulation.collision import (
    TrackBoundaryEvent,
    VehicleCollisionEvent,
    VehicleFootprint,
    boxes_intersect,
    detect_simulation_events,
    detect_vehicle_collisions,
    oriented_box_corners,
)
from f1_marl.simulation.manager import SimulationManager
from f1_marl.simulation.track import Track
from f1_marl.simulation.vehicle import ControlInput, VehicleState


@pytest.fixture
def track() -> Track:
    """Create a straight test track."""
    return Track(
        [(0.0, 0.0), (100.0, 0.0)],
        half_width_m=5.0,
    )


@pytest.fixture
def footprint() -> VehicleFootprint:
    """Create a small vehicle footprint for predictable geometry."""
    return VehicleFootprint(length_m=4.0, width_m=2.0)


@pytest.fixture
def manager(track: Track) -> SimulationManager:
    """Create a simulation manager for integration tests."""
    return SimulationManager(track)


# ---------------------------------------------------------------------------
# Vehicle footprint
# ---------------------------------------------------------------------------


def test_default_vehicle_footprint() -> None:
    footprint = VehicleFootprint()

    assert footprint.length_m == 5.0
    assert footprint.width_m == 2.0


@pytest.mark.parametrize(
    ("length_m", "width_m"),
    [
        (0.0, 2.0),
        (-1.0, 2.0),
        (5.0, 0.0),
        (5.0, -1.0),
        (math.inf, 2.0),
        (5.0, math.inf),
        (math.nan, 2.0),
        (5.0, math.nan),
    ],
)
def test_footprint_rejects_invalid_dimensions(
    length_m: float,
    width_m: float,
) -> None:
    with pytest.raises(ValueError):
        VehicleFootprint(length_m=length_m, width_m=width_m)


def test_custom_vehicle_footprint() -> None:
    footprint = VehicleFootprint(length_m=4.5, width_m=1.8)

    assert footprint.length_m == 4.5
    assert footprint.width_m == 1.8


# ---------------------------------------------------------------------------
# Oriented bounding box geometry
# ---------------------------------------------------------------------------


def test_oriented_box_corners_at_zero_heading(
    footprint: VehicleFootprint,
) -> None:
    state = VehicleState(x=0.0, y=0.0, heading_rad=0.0)

    corners = oriented_box_corners(state, footprint)

    assert corners == (
        (2.0, 1.0),
        (2.0, -1.0),
        (-2.0, -1.0),
        (-2.0, 1.0),
    )


def test_oriented_box_corners_at_ninety_degree_heading(
    footprint: VehicleFootprint,
) -> None:
    state = VehicleState(
        x=0.0,
        y=0.0,
        heading_rad=math.pi / 2,
    )

    corners = oriented_box_corners(state, footprint)

    expected = (
        (-1.0, 2.0),
        (1.0, 2.0),
        (1.0, -2.0),
        (-1.0, -2.0),
    )

    for actual_corner, expected_corner in zip(
        corners,
        expected,
        strict=True,
    ):
        assert actual_corner == pytest.approx(expected_corner)


def test_oriented_box_corners_respect_vehicle_position(
    footprint: VehicleFootprint,
) -> None:
    state = VehicleState(x=10.0, y=20.0, heading_rad=0.0)

    corners = oriented_box_corners(state, footprint)

    assert corners[0] == (12.0, 21.0)
    assert corners[2] == (8.0, 19.0)


# ---------------------------------------------------------------------------
# Separating Axis Theorem
# ---------------------------------------------------------------------------


def test_overlapping_vehicles_intersect(
    footprint: VehicleFootprint,
) -> None:
    state_a = VehicleState(x=0.0, y=0.0)
    state_b = VehicleState(x=2.0, y=0.0)

    assert boxes_intersect(state_a, state_b, footprint, footprint)


def test_separated_vehicles_do_not_intersect(
    footprint: VehicleFootprint,
) -> None:
    state_a = VehicleState(x=0.0, y=0.0)
    state_b = VehicleState(x=5.0, y=0.0)

    assert not boxes_intersect(state_a, state_b, footprint, footprint)


def test_touching_vehicle_edges_count_as_collision(
    footprint: VehicleFootprint,
) -> None:
    state_a = VehicleState(x=0.0, y=0.0)
    state_b = VehicleState(x=4.0, y=0.0)

    assert boxes_intersect(state_a, state_b, footprint, footprint)


def test_perpendicular_vehicles_intersect(
    footprint: VehicleFootprint,
) -> None:
    state_a = VehicleState(x=0.0, y=0.0, heading_rad=0.0)
    state_b = VehicleState(x=0.0, y=0.0, heading_rad=math.pi / 2)

    assert boxes_intersect(state_a, state_b, footprint, footprint)


def test_rotated_vehicles_do_not_intersect(
    footprint: VehicleFootprint,
) -> None:
    state_a = VehicleState(x=0.0, y=0.0, heading_rad=0.0)
    state_b = VehicleState(
        x=0.0,
        y=5.0,
        heading_rad=math.pi / 4,
    )

    assert not boxes_intersect(state_a, state_b, footprint, footprint)


def test_collision_detection_accepts_different_footprints() -> None:
    state_a = VehicleState(x=0.0, y=0.0)
    state_b = VehicleState(x=3.0, y=0.0)

    footprint_a = VehicleFootprint(length_m=2.0, width_m=1.0)
    footprint_b = VehicleFootprint(length_m=4.0, width_m=2.0)

    assert boxes_intersect(
        state_a,
        state_b,
        footprint_a,
        footprint_b,
    )


@pytest.mark.parametrize(
    "tolerance_m",
    [-1.0, math.inf, math.nan],
)
def test_collision_detection_rejects_invalid_tolerance(
    tolerance_m: float,
    footprint: VehicleFootprint,
) -> None:
    with pytest.raises(ValueError):
        boxes_intersect(
            VehicleState(),
            VehicleState(x=1.0),
            footprint,
            footprint,
            tolerance_m=tolerance_m,
        )


# ---------------------------------------------------------------------------
# Multiple vehicle collision detection
# ---------------------------------------------------------------------------


def test_detect_vehicle_collisions_returns_intersecting_pairs(
    footprint: VehicleFootprint,
) -> None:
    states = {
        "car_b": VehicleState(x=2.0, y=0.0),
        "car_a": VehicleState(x=0.0, y=0.0),
        "car_c": VehicleState(x=20.0, y=0.0),
    }

    collisions = detect_vehicle_collisions(states, footprint)

    assert collisions == (("car_a", "car_b"),)


def test_detect_vehicle_collisions_returns_all_pairs(
    footprint: VehicleFootprint,
) -> None:
    states = {
        "car_a": VehicleState(x=0.0, y=0.0),
        "car_b": VehicleState(x=1.0, y=0.0),
        "car_c": VehicleState(x=2.0, y=0.0),
    }

    collisions = detect_vehicle_collisions(states, footprint)

    assert collisions == (
        ("car_a", "car_b"),
        ("car_a", "car_c"),
        ("car_b", "car_c"),
    )


def test_detect_vehicle_collisions_returns_empty_tuple(
    footprint: VehicleFootprint,
) -> None:
    states = {
        "car_a": VehicleState(x=0.0, y=0.0),
        "car_b": VehicleState(x=20.0, y=0.0),
    }

    assert detect_vehicle_collisions(states, footprint) == ()


def test_detect_vehicle_collisions_with_one_vehicle(
    footprint: VehicleFootprint,
) -> None:
    states = {"car_a": VehicleState()}

    assert detect_vehicle_collisions(states, footprint) == ()


# ---------------------------------------------------------------------------
# Simulation event detection
# ---------------------------------------------------------------------------


def test_detect_simulation_events_reports_collision(
    track: Track,
    footprint: VehicleFootprint,
) -> None:
    states = {
        "car_a": VehicleState(x=10.0, y=0.0),
        "car_b": VehicleState(x=12.0, y=0.0),
    }

    events = detect_simulation_events(
        states,
        track,
        footprint,
        simulation_time_s=1.0,
    )

    assert len(events) == 1
    assert events[0] == VehicleCollisionEvent(
        vehicle_id_a="car_a",
        vehicle_id_b="car_b",
        simulation_time_s=1.0,
    )


def test_detect_simulation_events_reports_boundary_violation(
    track: Track,
    footprint: VehicleFootprint,
) -> None:
    states = {
        "car_a": VehicleState(x=10.0, y=6.0),
    }

    events = detect_simulation_events(
        states,
        track,
        footprint,
        simulation_time_s=2.0,
    )

    assert len(events) == 1
    assert isinstance(events[0], TrackBoundaryEvent)
    assert events[0].vehicle_id == "car_a"
    assert events[0].simulation_time_s == 2.0
    assert events[0].lateral_offset_m == pytest.approx(6.0)
    assert events[0].track_half_width_m == 5.0


def test_detect_simulation_events_reports_collision_and_boundary(
    track: Track,
    footprint: VehicleFootprint,
) -> None:
    states = {
        "car_a": VehicleState(x=10.0, y=5.0),
        "car_b": VehicleState(x=12.0, y=6.0),
    }

    events = detect_simulation_events(
        states,
        track,
        footprint,
        simulation_time_s=3.0,
    )

    assert any(isinstance(event, VehicleCollisionEvent) for event in events)
    assert any(isinstance(event, TrackBoundaryEvent) for event in events)


@pytest.mark.parametrize(
    "simulation_time_s",
    [-1.0, math.inf, math.nan],
)
def test_detect_simulation_events_rejects_invalid_time(
    track: Track,
    footprint: VehicleFootprint,
    simulation_time_s: float,
) -> None:
    with pytest.raises(ValueError):
        detect_simulation_events(
            {},
            track,
            footprint,
            simulation_time_s,
        )


# ---------------------------------------------------------------------------
# Simulation manager integration
# ---------------------------------------------------------------------------


def test_manager_detects_collision_after_step(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle(
        "car_a",
        VehicleState(x=10.0, y=0.0),
    )
    manager.add_vehicle(
        "car_b",
        VehicleState(x=12.0, y=0.0),
    )

    manager.step(
        {
            "car_a": ControlInput(),
            "car_b": ControlInput(),
        },
        dt=0.1,
    )

    assert len(manager.collision_events) == 1
    assert manager.collision_events[0].vehicle_id_a == "car_a"
    assert manager.collision_events[0].vehicle_id_b == "car_b"
    assert manager.collision_events[0].simulation_time_s == pytest.approx(0.1)


def test_manager_detects_boundary_violation_after_step(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle(
        "car_a",
        VehicleState(x=10.0, y=6.0),
    )

    manager.step(
        {"car_a": ControlInput()},
        dt=0.1,
    )

    assert len(manager.boundary_events) == 1
    assert manager.boundary_events[0].vehicle_id == "car_a"


def test_manager_exposes_combined_events(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle(
        "car_a",
        VehicleState(x=10.0, y=5.0),
    )
    manager.add_vehicle(
        "car_b",
        VehicleState(x=12.0, y=6.0),
    )

    manager.step(
        {
            "car_a": ControlInput(),
            "car_b": ControlInput(),
        },
        dt=0.1,
    )

    assert len(manager.events) == 2
    assert len(manager.collision_events) == 1
    assert len(manager.boundary_events) == 1


def test_manager_clears_events_on_reset(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle(
        "car_a",
        VehicleState(x=10.0, y=6.0),
    )

    manager.step(
        {"car_a": ControlInput()},
        dt=0.1,
    )

    assert manager.events

    manager.reset()

    assert manager.events == ()
    assert manager.simulation_time_s == 0.0


def test_manager_clears_events_when_vehicle_is_added(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle(
        "car_a",
        VehicleState(x=10.0, y=6.0),
    )

    manager.step(
        {"car_a": ControlInput()},
        dt=0.1,
    )

    assert manager.events

    manager.add_vehicle("car_b")

    assert manager.events == ()


def test_manager_clears_events_when_vehicle_is_removed(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle(
        "car_a",
        VehicleState(x=10.0, y=6.0),
    )

    manager.step(
        {"car_a": ControlInput()},
        dt=0.1,
    )

    assert manager.events

    manager.remove_vehicle("car_a")

    assert manager.events == ()
