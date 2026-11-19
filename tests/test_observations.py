"""Tests for structured racing observations."""

import math

import numpy as np
import pytest

from f1_marl.environment.observations import (
    OBSERVATION_DIM,
    ObservationBuilder,
    ObservationConfig,
)
from f1_marl.simulation.manager import SimulationManager
from f1_marl.simulation.tracks import create_oval_track
from f1_marl.simulation.vehicle import ControlInput, VehicleState


@pytest.fixture
def track():
    return create_oval_track()


def make_vehicle_state(track, s, speed=20.0):
    """Create a vehicle state aligned with the track at distance s."""
    sample = track.sample(s)

    heading = math.atan2(
        sample.tangent_y,
        sample.tangent_x,
    )

    return VehicleState(
        x=sample.x,
        y=sample.y,
        heading_rad=heading,
        speed_mps=speed,
    )


@pytest.fixture
def simulation(track):
    """Create a simulation containing one vehicle."""
    sim = SimulationManager(track)
    sim.add_vehicle("ego", make_vehicle_state(track, 0.0))
    return sim


@pytest.fixture
def builder():
    return ObservationBuilder()


def test_observation_has_29_float32_values(builder, simulation):
    observation = builder.build("ego", simulation)

    assert observation.shape == (OBSERVATION_DIM,)
    assert observation.dtype == np.float32
    assert observation.shape == (29,)


def test_observation_contains_only_finite_values(builder, simulation):
    observation = builder.build("ego", simulation)

    assert np.all(np.isfinite(observation))


def test_observation_values_respect_normalized_bounds(builder, simulation):
    observation = builder.build("ego", simulation)

    assert np.all(observation >= -1.0)
    assert np.all(observation <= 1.0)


def test_speed_is_normalized(builder, track):
    sim = SimulationManager(track)
    sim.add_vehicle("ego", make_vehicle_state(track, 0.0, speed=50.0))

    observation = builder.build("ego", sim)

    assert observation[0] == pytest.approx(0.5)


def test_previous_actions_are_included(builder, simulation):
    observation = builder.build(
        "ego",
        simulation,
        previous_actions={"ego": (0.5, -0.25)},
    )

    assert observation[3] == pytest.approx(0.5)
    assert observation[4] == pytest.approx(-0.25)


def test_opponent_slots_are_populated(builder, track):
    sim = SimulationManager(track)

    sim.add_vehicle("ego", make_vehicle_state(track, 0.0))
    sim.add_vehicle("opponent_1", make_vehicle_state(track, 15.0))
    sim.add_vehicle("opponent_2", make_vehicle_state(track, 30.0))
    sim.add_vehicle("opponent_3", make_vehicle_state(track, 45.0))

    observation = builder.build("ego", sim)

    assert observation[16] == 1.0
    assert observation[21] == 1.0
    assert observation[26] == 1.0


def test_unused_opponent_slots_are_zero(builder, simulation):
    observation = builder.build("ego", simulation)

    assert np.all(observation[12:27] == 0.0)


def test_opponent_slots_are_ordered_by_distance(builder, track):
    sim = SimulationManager(track)

    sim.add_vehicle("ego", make_vehicle_state(track, 0.0))
    sim.add_vehicle("farther", make_vehicle_state(track, 40.0))
    sim.add_vehicle("nearer", make_vehicle_state(track, 10.0))

    observation = builder.build("ego", sim)

    first_opponent_distance = abs(float(observation[12]))
    second_opponent_distance = abs(float(observation[17]))

    assert first_opponent_distance < second_opponent_distance


def test_curvature_features_are_finite(builder, simulation):
    observation = builder.build("ego", simulation)

    assert np.all(np.isfinite(observation[7:12]))


def test_on_track_indicator(builder, simulation):
    observation = builder.build("ego", simulation)

    assert observation[27] == 1.0


def test_collision_indicator(builder, track):
    sim = SimulationManager(track)

    overlapping_state = make_vehicle_state(track, 0.0)

    sim.add_vehicle("ego", overlapping_state)
    sim.add_vehicle("opponent", overlapping_state)

    stationary_control = ControlInput(
        steering_angle_rad=0.0,
        throttle=0.0,
        brake=0.0,
    )

    sim.step(
        {
            "ego": stationary_control,
            "opponent": stationary_control,
        },
        0.05,
    )

    observation = builder.build("ego", sim)

    assert observation[28] == 1.0


def test_observation_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        ObservationConfig(speed_scale_mps=0.0)

    with pytest.raises(ValueError):
        ObservationConfig(perception_radius_m=-1.0)
