"""Tests for kinematic bicycle vehicle dynamics."""

import pytest

from f1_marl.simulation.physics import step_vehicle
from f1_marl.simulation.vehicle import (
    ControlInput,
    VehicleParameters,
    VehicleState,
)


@pytest.fixture
def parameters() -> VehicleParameters:
    return VehicleParameters()


def test_vehicle_accelerates(parameters: VehicleParameters) -> None:
    state = VehicleState()
    control = ControlInput(throttle=1.0)

    result = step_vehicle(state, control, parameters, 0.1)

    assert result.speed_mps == pytest.approx(0.9)
    assert result.x > 0.0
    assert result.y == pytest.approx(0.0)


def test_vehicle_brakes(parameters: VehicleParameters) -> None:
    state = VehicleState(speed_mps=20.0)
    control = ControlInput(brake=1.0)

    result = step_vehicle(state, control, parameters, 0.1)

    assert result.speed_mps == pytest.approx(18.8)


def test_vehicle_cannot_reverse(parameters: VehicleParameters) -> None:
    state = VehicleState(speed_mps=1.0)
    control = ControlInput(brake=1.0)

    result = step_vehicle(state, control, parameters, 1.0)

    assert result.speed_mps == 0.0


def test_vehicle_turns_left(parameters: VehicleParameters) -> None:
    state = VehicleState(speed_mps=20.0)
    control = ControlInput(steering_angle_rad=0.2)

    result = step_vehicle(state, control, parameters, 0.1)

    assert result.heading_rad > 0.0
    assert result.y > 0.0


def test_vehicle_turns_right(parameters: VehicleParameters) -> None:
    state = VehicleState(speed_mps=20.0)
    control = ControlInput(steering_angle_rad=-0.2)

    result = step_vehicle(state, control, parameters, 0.1)

    assert result.heading_rad < 0.0
    assert result.y < 0.0


def test_vehicle_limits_steering(parameters: VehicleParameters) -> None:
    state = VehicleState(speed_mps=10.0)

    normal = step_vehicle(
        state,
        ControlInput(steering_angle_rad=0.55),
        parameters,
        0.1,
    )

    excessive = step_vehicle(
        state,
        ControlInput(steering_angle_rad=100.0),
        parameters,
        0.1,
    )

    assert excessive.heading_rad == pytest.approx(normal.heading_rad)


def test_vehicle_rejects_invalid_timestep(
    parameters: VehicleParameters,
) -> None:
    with pytest.raises(ValueError, match="[Tt]imestep"):
        step_vehicle(
            VehicleState(),
            ControlInput(),
            parameters,
            0.0,
        )


def test_straight_line_motion(parameters: VehicleParameters) -> None:
    state = VehicleState(speed_mps=10.0)

    result = step_vehicle(
        state,
        ControlInput(),
        parameters,
        1.0,
    )

    assert result.x == pytest.approx(10.0)
    assert result.y == pytest.approx(0.0)
    assert result.heading_rad == pytest.approx(0.0)
