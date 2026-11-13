"""Tests for the multi-vehicle simulation manager."""

import pytest

from f1_marl.simulation.manager import SimulationManager
from f1_marl.simulation.track import Track
from f1_marl.simulation.vehicle import (
    ControlInput,
    VehicleState,
)


@pytest.fixture
def manager() -> SimulationManager:
    track = Track(
        [(0.0, 0.0), (100.0, 0.0)],
        half_width_m=5.0,
    )
    return SimulationManager(track)


def test_manager_starts_without_vehicles(
    manager: SimulationManager,
) -> None:
    assert manager.vehicle_ids == ()
    assert manager.simulation_time_s == 0.0


def test_add_vehicle(manager: SimulationManager) -> None:
    manager.add_vehicle("car_1")

    assert manager.vehicle_ids == ("car_1",)
    assert manager.get_state("car_1") == VehicleState()


def test_add_vehicle_with_initial_state(
    manager: SimulationManager,
) -> None:
    initial_state = VehicleState(x=10.0, speed_mps=20.0)

    manager.add_vehicle("car_1", initial_state)

    assert manager.get_state("car_1") == initial_state


def test_duplicate_vehicle_id_is_rejected(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle("car_1")

    with pytest.raises(ValueError, match="already exists"):
        manager.add_vehicle("car_1")


@pytest.mark.parametrize("vehicle_id", ["", " ", "\t"])
def test_empty_vehicle_id_is_rejected(
    manager: SimulationManager,
    vehicle_id: str,
) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        manager.add_vehicle(vehicle_id)


def test_remove_vehicle(manager: SimulationManager) -> None:
    manager.add_vehicle("car_1")
    manager.remove_vehicle("car_1")

    assert manager.vehicle_ids == ()


def test_remove_unknown_vehicle_is_rejected(
    manager: SimulationManager,
) -> None:
    with pytest.raises(KeyError, match="does not exist"):
        manager.remove_vehicle("unknown")


def test_multiple_vehicles_have_independent_states(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle("car_1", VehicleState(x=10.0))
    manager.add_vehicle("car_2", VehicleState(x=20.0))

    manager.step(
        {
            "car_1": ControlInput(throttle=1.0),
            "car_2": ControlInput(),
        },
        dt=0.1,
    )

    assert manager.get_state("car_1").speed_mps == pytest.approx(0.9)
    assert manager.get_state("car_2").speed_mps == pytest.approx(0.0)
    assert manager.get_state("car_1").x > 10.0
    assert manager.get_state("car_2").x == pytest.approx(20.0)


def test_all_vehicles_advance_synchronously(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle("car_1", VehicleState(speed_mps=10.0))
    manager.add_vehicle("car_2", VehicleState(speed_mps=20.0))

    manager.step(
        {
            "car_1": ControlInput(),
            "car_2": ControlInput(),
        },
        dt=0.1,
    )

    assert manager.get_state("car_1").x == pytest.approx(1.0)
    assert manager.get_state("car_2").x == pytest.approx(2.0)
    assert manager.simulation_time_s == pytest.approx(0.1)


def test_step_requires_control_for_every_vehicle(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle("car_1")
    manager.add_vehicle("car_2")

    with pytest.raises(ValueError, match="Control IDs"):
        manager.step(
            {"car_1": ControlInput()},
            dt=0.1,
        )

    assert manager.simulation_time_s == 0.0
    assert manager.get_state("car_1") == VehicleState()


def test_step_rejects_unknown_vehicle_control(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle("car_1")

    with pytest.raises(ValueError, match="Control IDs"):
        manager.step(
            {"unknown": ControlInput()},
            dt=0.1,
        )


@pytest.mark.parametrize("dt", [0.0, -0.1, float("inf"), float("nan")])
def test_step_rejects_invalid_timestep(
    manager: SimulationManager,
    dt: float,
) -> None:
    manager.add_vehicle("car_1")

    with pytest.raises(ValueError, match="Timestep"):
        manager.step(
            {"car_1": ControlInput()},
            dt=dt,
        )


def test_get_unknown_vehicle_state_is_rejected(
    manager: SimulationManager,
) -> None:
    with pytest.raises(KeyError, match="does not exist"):
        manager.get_state("unknown")


def test_get_vehicle_track_interaction(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle(
        "car_1",
        VehicleState(x=25.0, y=2.0),
    )

    interaction = manager.get_track_interaction("car_1")

    assert interaction.s == pytest.approx(25.0)
    assert interaction.d == pytest.approx(2.0)
    assert interaction.is_on_track


def test_get_vehicle_snapshot(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle(
        "car_1",
        VehicleState(x=25.0, y=1.0),
    )

    snapshot = manager.get_snapshot("car_1")

    assert snapshot.vehicle_id == "car_1"
    assert snapshot.state.x == pytest.approx(25.0)
    assert snapshot.track_interaction.d == pytest.approx(1.0)


def test_reset_restores_initial_states(
    manager: SimulationManager,
) -> None:
    initial_state = VehicleState(x=10.0, speed_mps=5.0)

    manager.add_vehicle("car_1", initial_state)

    manager.step(
        {"car_1": ControlInput(throttle=1.0)},
        dt=1.0,
    )

    assert manager.get_state("car_1") != initial_state

    manager.reset()

    assert manager.get_state("car_1") == initial_state
    assert manager.simulation_time_s == 0.0


def test_states_property_returns_copy(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle("car_1")

    external_states = manager.states
    external_states.clear()

    assert manager.vehicle_ids == ("car_1",)
