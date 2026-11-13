"""Tests for racing scenarios and starting-grid generation."""

import math

import pytest

from f1_marl.simulation.collision import VehicleFootprint, boxes_intersect
from f1_marl.simulation.scenarios import (
    RaceScenario,
    StartingGrid,
    generate_starting_grid,
)
from f1_marl.simulation.tracks import (
    create_oval_track,
    create_technical_track,
)
from f1_marl.simulation.vehicle import VehicleParameters


@pytest.fixture
def track():
    return create_oval_track()


@pytest.fixture
def vehicle_ids():
    return ("car_1", "car_2", "car_3", "car_4")


def test_starting_grid_generates_all_vehicles(track, vehicle_ids):
    grid = generate_starting_grid(track, vehicle_ids)

    assert isinstance(grid, StartingGrid)
    assert grid.vehicle_ids == vehicle_ids
    assert len(grid.vehicle_states) == 4


def test_starting_grid_is_deterministic(track, vehicle_ids):
    first = generate_starting_grid(track, vehicle_ids)
    second = generate_starting_grid(track, vehicle_ids)

    assert first == second


def test_starting_grid_uses_configured_initial_speed(track, vehicle_ids):
    grid = generate_starting_grid(
        track,
        vehicle_ids,
        initial_speed_mps=12.0,
    )

    assert all(state.speed_mps == 12.0 for state in grid.states.values())


def test_starting_grid_uses_configured_start_position(track, vehicle_ids):
    grid = generate_starting_grid(
        track,
        vehicle_ids[:2],
        start_s_m=50.0,
    )

    for state in grid.states.values():
        projected = track.project(state.x, state.y)
        assert projected.s == pytest.approx(50.0, abs=1e-6)


def test_starting_grid_uses_configured_lateral_offsets(track):
    grid = generate_starting_grid(
        track,
        ("left", "right"),
        lateral_offsets_m=(-3.0, 3.0),
    )

    assert track.project(
        grid.states["left"].x,
        grid.states["left"].y,
    ).d == pytest.approx(-3.0)

    assert track.project(
        grid.states["right"].x,
        grid.states["right"].y,
    ).d == pytest.approx(3.0, abs=0.02)


def test_starting_grid_aligns_vehicle_heading_with_track(track):
    grid = generate_starting_grid(track, ("car_1",))

    state = grid.states["car_1"]
    projected = track.project(state.x, state.y)
    sample = track.sample(projected.s)

    expected_heading = math.atan2(
        sample.tangent_y,
        sample.tangent_x,
    )

    assert state.heading_rad == pytest.approx(expected_heading)


def test_starting_grid_uses_longitudinal_spacing(track):
    grid = generate_starting_grid(
        track,
        ("car_1", "car_2", "car_3", "car_4"),
        longitudinal_spacing_m=12.0,
    )

    first = track.project(
        grid.states["car_1"].x,
        grid.states["car_1"].y,
    )
    third = track.project(
        grid.states["car_3"].x,
        grid.states["car_3"].y,
    )

    expected = track.sample(-12.0).s

    assert third.s == pytest.approx(expected)
    assert first.s == pytest.approx(0.0)


def test_starting_grid_contains_no_overlapping_vehicles(track, vehicle_ids):
    grid = generate_starting_grid(track, vehicle_ids)
    states = grid.states
    footprint = VehicleFootprint()

    for index, vehicle_id_a in enumerate(grid.vehicle_ids):
        for vehicle_id_b in grid.vehicle_ids[index + 1 :]:
            assert not boxes_intersect(
                states[vehicle_id_a],
                states[vehicle_id_b],
                footprint,
                footprint,
            )


def test_starting_grid_rejects_overlapping_lanes(track):
    with pytest.raises(ValueError, match="overlap"):
        generate_starting_grid(
            track,
            ("car_1", "car_2"),
            lateral_offsets_m=(0.0,),
            longitudinal_spacing_m=2.0,
        )


def test_starting_grid_rejects_vehicle_outside_track(track):
    with pytest.raises(ValueError, match="track boundaries"):
        generate_starting_grid(
            track,
            ("car_1",),
            lateral_offsets_m=(8.0,),
        )


def test_starting_grid_rejects_open_track():
    open_track = create_oval_track()
    open_track = type(open_track)(
        open_track.centerline,
        closed=False,
        half_width_m=open_track.half_width_m,
    )

    with pytest.raises(ValueError, match="closed"):
        generate_starting_grid(open_track, ("car_1",))


@pytest.mark.parametrize(
    "vehicle_ids",
    [
        (),
        ("car_1", ""),
        ("car_1", " "),
        ("car_1", "car_1"),
    ],
)
def test_starting_grid_rejects_invalid_vehicle_ids(track, vehicle_ids):
    with pytest.raises(ValueError):
        generate_starting_grid(track, vehicle_ids)


@pytest.mark.parametrize(
    "spacing",
    [0.0, -1.0, float("inf"), float("nan")],
)
def test_starting_grid_rejects_invalid_longitudinal_spacing(track, spacing):
    with pytest.raises(ValueError, match="spacing"):
        generate_starting_grid(
            track,
            ("car_1",),
            longitudinal_spacing_m=spacing,
        )


def test_starting_grid_rejects_empty_lateral_offsets(track):
    with pytest.raises(ValueError, match="lateral offset"):
        generate_starting_grid(
            track,
            ("car_1",),
            lateral_offsets_m=(),
        )


def test_starting_grid_rejects_non_finite_lateral_offsets(track):
    with pytest.raises(ValueError, match="finite"):
        generate_starting_grid(
            track,
            ("car_1",),
            lateral_offsets_m=(float("nan"),),
        )


@pytest.mark.parametrize(
    "speed",
    [-1.0, float("inf"), float("nan")],
)
def test_starting_grid_rejects_invalid_initial_speed(track, speed):
    with pytest.raises(ValueError, match="speed"):
        generate_starting_grid(
            track,
            ("car_1",),
            initial_speed_mps=speed,
        )


def test_starting_grid_rejects_non_finite_start_position(track):
    with pytest.raises(ValueError, match="Starting position"):
        generate_starting_grid(
            track,
            ("car_1",),
            start_s_m=float("nan"),
        )


def test_starting_grid_states_returns_independent_dictionary(track):
    grid = generate_starting_grid(track, ("car_1", "car_2"))

    states = grid.states
    states.clear()

    assert len(grid.states) == 2


def test_race_scenario_generates_starting_grid(track, vehicle_ids):
    scenario = RaceScenario(
        track=track,
        vehicle_ids=vehicle_ids,
    )

    grid = scenario.generate_grid()

    assert grid.vehicle_ids == vehicle_ids


def test_race_scenario_creates_simulation(track, vehicle_ids):
    scenario = RaceScenario(
        track=track,
        vehicle_ids=vehicle_ids,
    )

    manager = scenario.create_simulation()

    assert manager.track is track
    assert manager.vehicle_ids == vehicle_ids
    assert len(manager.states) == 4


def test_race_scenario_passes_vehicle_parameters(track):
    parameters = VehicleParameters(wheelbase_m=3.0)

    scenario = RaceScenario(
        track=track,
        vehicle_ids=("car_1",),
        parameters=parameters,
    )

    manager = scenario.create_simulation()

    assert manager.parameters is parameters


def test_race_scenario_passes_vehicle_footprint(track):
    footprint = VehicleFootprint(length_m=4.0, width_m=1.8)

    scenario = RaceScenario(
        track=track,
        vehicle_ids=("car_1", "car_2"),
        footprint=footprint,
    )

    manager = scenario.create_simulation()

    assert manager.footprint is footprint


def test_race_scenario_supports_technical_track():
    track = create_technical_track()

    scenario = RaceScenario(
        track=track,
        vehicle_ids=("car_1", "car_2"),
    )

    manager = scenario.create_simulation()

    assert manager.track is track
    assert len(manager.vehicle_ids) == 2


def test_race_scenario_reset_restores_starting_grid(track):
    scenario = RaceScenario(
        track=track,
        vehicle_ids=("car_1", "car_2"),
    )

    manager = scenario.create_simulation()
    initial_states = manager.states.copy()

    manager.reset()

    assert manager.states == initial_states
    assert manager.simulation_time_s == 0.0
