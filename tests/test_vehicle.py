"""Tests for vehicle state, parameters, and controls."""

import pytest

from f1_marl.simulation.vehicle import (
    ControlInput,
    VehicleParameters,
    VehicleState,
)


def test_default_vehicle_state() -> None:
    state = VehicleState()

    assert state.x == 0.0
    assert state.y == 0.0
    assert state.heading_rad == 0.0
    assert state.speed_mps == 0.0


def test_vehicle_rejects_negative_speed() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        VehicleState(speed_mps=-1.0)


def test_vehicle_rejects_non_finite_state() -> None:
    with pytest.raises(ValueError, match="finite"):
        VehicleState(x=float("nan"))


def test_vehicle_parameters_require_positive_wheelbase() -> None:
    with pytest.raises(ValueError, match="Wheelbase"):
        VehicleParameters(wheelbase_m=0.0)


def test_vehicle_parameters_reject_invalid_steering() -> None:
    with pytest.raises(ValueError, match="steering angle"):
        VehicleParameters(max_steering_angle_rad=2.0)


def test_control_input_accepts_valid_values() -> None:
    control = ControlInput(
        steering_angle_rad=0.2,
        throttle=0.8,
        brake=0.0,
    )

    assert control.throttle == 0.8


@pytest.mark.parametrize("throttle", [-0.1, 1.1])
def test_control_rejects_invalid_throttle(throttle: float) -> None:
    with pytest.raises(ValueError, match="Throttle"):
        ControlInput(throttle=throttle)


@pytest.mark.parametrize("brake", [-0.1, 1.1])
def test_control_rejects_invalid_brake(brake: float) -> None:
    with pytest.raises(ValueError, match="Brake"):
        ControlInput(brake=brake)
