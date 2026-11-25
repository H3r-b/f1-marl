"""Tests for configurable multi-agent racing rewards."""

import math

import numpy as np
import pytest

from f1_marl.environment.rewards import (
    RewardBreakdown,
    RewardCalculator,
    RewardConfig,
)
from f1_marl.simulation.manager import SimulationManager
from f1_marl.simulation.track import Track
from f1_marl.simulation.vehicle import ControlInput, VehicleState


@pytest.fixture
def manager() -> SimulationManager:
    track = Track(
        [(0.0, 0.0), (1000.0, 0.0)],
        half_width_m=5.0,
        closed=False,
    )
    return SimulationManager(track)


def test_default_reward_configuration() -> None:
    config = RewardConfig()

    assert config.progress_weight == 1.0
    assert config.speed_weight == 0.2
    assert config.overtaking_weight == 2.0
    assert config.collision_penalty == 5.0
    assert config.track_limit_penalty == 1.0
    assert config.oscillation_penalty == 0.1


@pytest.mark.parametrize(
    "field_name",
    [
        "progress_weight",
        "speed_weight",
        "overtaking_weight",
        "collision_penalty",
        "track_limit_penalty",
        "oscillation_penalty",
        "oscillation_threshold",
    ],
)
@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_invalid_reward_configuration_is_rejected(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        RewardConfig(**{field_name: value})


@pytest.mark.parametrize(
    "field_name",
    ["target_speed_mps", "overtake_detection_distance_m"],
)
@pytest.mark.parametrize(
    "value",
    [0.0, -1.0, float("inf"), float("nan")],
)
def test_invalid_positive_parameters_are_rejected(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        RewardConfig(**{field_name: value})


def test_default_reward_breakdown() -> None:
    assert RewardBreakdown().total == 0.0


def test_reward_breakdown_total() -> None:
    breakdown = RewardBreakdown(
        progress=10.0,
        speed=0.2,
        overtaking=2.0,
        collision=-5.0,
        track_limits=-1.0,
        oscillation=-0.1,
    )

    assert breakdown.total == pytest.approx(6.1)


def test_forward_progress_reward(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle(
        "car_1",
        VehicleState(x=10.0, speed_mps=20.0),
    )

    calculator = RewardCalculator()
    calculator.reset(manager)

    manager.step(
        {"car_1": ControlInput()},
        dt=0.1,
    )

    rewards = calculator.calculate(
        manager,
        {"car_1": np.zeros(2)},
    )

    assert rewards["car_1"].progress == pytest.approx(2.0)
    assert rewards["car_1"].total > 0.0
    assert math.isfinite(rewards["car_1"].total)


def test_target_speed_reward(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle(
        "car_1",
        VehicleState(x=10.0, speed_mps=50.0),
    )

    calculator = RewardCalculator()
    calculator.reset(manager)

    manager.step(
        {"car_1": ControlInput()},
        dt=0.1,
    )

    rewards = calculator.calculate(
        manager,
        {"car_1": np.zeros(2)},
    )

    assert rewards["car_1"].speed == pytest.approx(0.2)


def test_successful_overtake_reward(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle(
        "car_1",
        VehicleState(x=10.0, y=-2.0, speed_mps=25.0),
    )
    manager.add_vehicle(
        "car_2",
        VehicleState(x=18.0, y=2.0, speed_mps=5.0),
    )

    calculator = RewardCalculator()
    calculator.reset(manager)

    controls = {
        "car_1": ControlInput(),
        "car_2": ControlInput(),
    }

    actions = {
        "car_1": np.zeros(2),
        "car_2": np.zeros(2),
    }

    manager.step(controls, dt=1.0)
    rewards = calculator.calculate(manager, actions)

    assert rewards["car_1"].overtaking == pytest.approx(2.0)
    assert rewards["car_2"].overtaking == 0.0

    manager.step(controls, dt=0.1)
    rewards = calculator.calculate(manager, actions)

    assert rewards["car_1"].overtaking == 0.0
    assert rewards["car_2"].overtaking == 0.0


def test_collision_penalty(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle("car_1", VehicleState(x=10.0))
    manager.add_vehicle("car_2", VehicleState(x=11.0))

    calculator = RewardCalculator()
    calculator.reset(manager)

    manager.step(
        {
            "car_1": ControlInput(),
            "car_2": ControlInput(),
        },
        dt=0.1,
    )

    rewards = calculator.calculate(
        manager,
        {
            "car_1": np.zeros(2),
            "car_2": np.zeros(2),
        },
    )

    assert rewards["car_1"].collision == pytest.approx(-5.0)
    assert rewards["car_2"].collision == pytest.approx(-5.0)


def test_persistent_collision_is_not_penalized_twice(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle("car_1", VehicleState(x=10.0))
    manager.add_vehicle("car_2", VehicleState(x=11.0))

    calculator = RewardCalculator()
    calculator.reset(manager)

    controls = {
        "car_1": ControlInput(),
        "car_2": ControlInput(),
    }

    actions = {
        "car_1": np.zeros(2),
        "car_2": np.zeros(2),
    }

    manager.step(controls, dt=0.1)
    first_rewards = calculator.calculate(manager, actions)

    manager.step(controls, dt=0.1)
    second_rewards = calculator.calculate(manager, actions)

    assert first_rewards["car_1"].collision == pytest.approx(-5.0)
    assert first_rewards["car_2"].collision == pytest.approx(-5.0)

    assert second_rewards["car_1"].collision == 0.0
    assert second_rewards["car_2"].collision == 0.0


def test_track_boundary_penalty(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle(
        "car_1",
        VehicleState(x=10.0, y=6.0),
    )

    calculator = RewardCalculator()
    calculator.reset(manager)

    manager.step(
        {"car_1": ControlInput()},
        dt=0.1,
    )

    rewards = calculator.calculate(
        manager,
        {"car_1": np.zeros(2)},
    )

    assert rewards["car_1"].track_limits == pytest.approx(-1.0)
    assert rewards["car_1"].total < 0.0


def test_control_oscillation_penalty(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle("car_1", VehicleState(x=10.0))

    calculator = RewardCalculator()
    calculator.reset(manager)

    manager.step({"car_1": ControlInput()}, dt=0.1)

    first_rewards = calculator.calculate(
        manager,
        {"car_1": np.array([1.0, 0.0])},
    )

    manager.step({"car_1": ControlInput()}, dt=0.1)

    second_rewards = calculator.calculate(
        manager,
        {"car_1": np.array([-1.0, 0.0])},
    )

    assert first_rewards["car_1"].oscillation == 0.0
    assert second_rewards["car_1"].oscillation == pytest.approx(-0.1)


def test_custom_reward_weights(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle(
        "car_1",
        VehicleState(x=10.0, speed_mps=20.0),
    )

    config = RewardConfig(
        progress_weight=3.0,
        speed_weight=0.5,
        target_speed_mps=20.0,
    )

    calculator = RewardCalculator(config)
    calculator.reset(manager)

    manager.step({"car_1": ControlInput()}, dt=0.1)

    rewards = calculator.calculate(
        manager,
        {"car_1": np.zeros(2)},
    )

    assert rewards["car_1"].progress == pytest.approx(6.0)
    assert rewards["car_1"].speed == pytest.approx(0.5)
    assert rewards["car_1"].total == pytest.approx(6.5)


def test_zero_reward_weights_disable_components(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle(
        "car_1",
        VehicleState(x=10.0, y=6.0, speed_mps=20.0),
    )

    config = RewardConfig(
        progress_weight=0.0,
        speed_weight=0.0,
        overtaking_weight=0.0,
        collision_penalty=0.0,
        track_limit_penalty=0.0,
        oscillation_penalty=0.0,
    )

    calculator = RewardCalculator(config)
    calculator.reset(manager)

    manager.step({"car_1": ControlInput()}, dt=0.1)

    rewards = calculator.calculate(
        manager,
        {"car_1": np.zeros(2)},
    )

    assert rewards["car_1"] == RewardBreakdown()


def test_reset_clears_reward_history(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle("car_1", VehicleState(x=10.0))

    calculator = RewardCalculator()
    calculator.reset(manager)

    manager.step({"car_1": ControlInput()}, dt=0.1)

    calculator.calculate(
        manager,
        {"car_1": np.array([1.0, 0.0])},
    )

    manager.reset()
    calculator.reset(manager)

    manager.step({"car_1": ControlInput()}, dt=0.1)

    rewards = calculator.calculate(
        manager,
        {"car_1": np.array([-1.0, 0.0])},
    )

    assert rewards["car_1"].oscillation == 0.0
    assert rewards["car_1"].progress == 0.0


def test_missing_agent_action_is_rejected(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle("car_1")
    manager.add_vehicle("car_2")

    calculator = RewardCalculator()
    calculator.reset(manager)

    with pytest.raises(ValueError, match="every vehicle"):
        calculator.calculate(
            manager,
            {"car_1": np.zeros(2)},
        )


@pytest.mark.parametrize(
    "action",
    [
        np.array([0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([float("nan"), 0.0]),
        np.array([0.0, float("inf")]),
    ],
)
def test_invalid_reward_action_is_rejected(
    manager: SimulationManager,
    action: np.ndarray,
) -> None:
    manager.add_vehicle("car_1")

    calculator = RewardCalculator()
    calculator.reset(manager)

    with pytest.raises(ValueError, match="two finite values"):
        calculator.calculate(manager, {"car_1": action})


def test_calculation_requires_initialized_history(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle("car_1")

    calculator = RewardCalculator()

    with pytest.raises(RuntimeError, match="reset"):
        calculator.calculate(
            manager,
            {"car_1": np.zeros(2)},
        )


def test_closed_track_progress_wraparound() -> None:
    track = Track(
        [
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        ],
        half_width_m=5.0,
        closed=True,
    )

    manager = SimulationManager(track)

    manager.add_vehicle(
        "car_1",
        VehicleState(
            x=0.0,
            y=2.0,
            heading_rad=-math.pi / 2,
            speed_mps=10.0,
        ),
    )

    calculator = RewardCalculator()
    calculator.reset(manager)

    manager.step({"car_1": ControlInput()}, dt=0.2)

    rewards = calculator.calculate(
        manager,
        {"car_1": np.zeros(2)},
    )

    assert rewards["car_1"].progress == pytest.approx(2.0)


@pytest.mark.parametrize(
    "value",
    [0.0, -1.0, float("inf"), float("nan"), 20.0, 25.0],
)
def test_invalid_overtake_confirmation_distance(
    value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="overtake_confirmation_distance_m",
    ):
        RewardConfig(overtake_confirmation_distance_m=value)


def test_overtake_requires_confirmed_separation(
    manager: SimulationManager,
) -> None:
    manager.add_vehicle(
        "car_1",
        VehicleState(x=10.0, y=-2.0, speed_mps=10.0),
    )
    manager.add_vehicle(
        "car_2",
        VehicleState(x=14.0, y=2.0),
    )

    calculator = RewardCalculator()
    calculator.reset(manager)

    controls = {
        "car_1": ControlInput(),
        "car_2": ControlInput(),
    }

    actions = {
        "car_1": np.zeros(2),
        "car_2": np.zeros(2),
    }

    # Vehicles reach equal longitudinal progress.
    manager.step(controls, dt=0.4)
    rewards = calculator.calculate(manager, actions)

    assert rewards["car_1"].overtaking == 0.0

    # Car 1 moves ahead, but separation is insufficient.
    manager.step(controls, dt=0.1)
    rewards = calculator.calculate(manager, actions)

    assert rewards["car_1"].overtaking == 0.0

    # Car 1 exceeds the confirmation distance.
    manager.step(controls, dt=0.2)
    rewards = calculator.calculate(manager, actions)

    assert rewards["car_1"].overtaking == pytest.approx(2.0)
    assert rewards["car_2"].overtaking == 0.0

    # Remaining ahead must not generate another reward.
    manager.step(controls, dt=0.1)
    rewards = calculator.calculate(manager, actions)

    assert rewards["car_1"].overtaking == 0.0


def test_reward_calculation_after_agent_removal(
    manager: SimulationManager,
) -> None:
    """Remaining agents continue receiving rewards."""
    manager.add_vehicle(
        "car_1",
        VehicleState(x=10.0, speed_mps=20.0),
    )
    manager.add_vehicle(
        "car_2",
        VehicleState(x=30.0, speed_mps=10.0),
    )

    calculator = RewardCalculator()
    calculator.reset(manager)

    manager.remove_vehicle("car_1")
    calculator.remove_agents(("car_1",))

    manager.step({"car_2": ControlInput()}, dt=0.1)

    rewards = calculator.calculate(
        manager,
        {"car_2": np.zeros(2)},
    )

    assert set(rewards) == {"car_2"}
    assert rewards["car_2"].progress == pytest.approx(1.0)
    assert math.isfinite(rewards["car_2"].total)


def test_agent_removal_preserves_oscillation_history(
    manager: SimulationManager,
) -> None:
    """Removing an agent preserves surviving control history."""
    manager.add_vehicle("car_1", VehicleState(x=10.0))
    manager.add_vehicle("car_2", VehicleState(x=30.0))

    calculator = RewardCalculator()
    calculator.reset(manager)

    manager.step(
        {
            "car_1": ControlInput(),
            "car_2": ControlInput(),
        },
        dt=0.1,
    )

    calculator.calculate(
        manager,
        {
            "car_1": np.zeros(2),
            "car_2": np.array([1.0, 0.0]),
        },
    )

    manager.remove_vehicle("car_1")
    calculator.remove_agents(("car_1",))

    manager.step({"car_2": ControlInput()}, dt=0.1)

    rewards = calculator.calculate(
        manager,
        {"car_2": np.array([-1.0, 0.0])},
    )

    assert rewards["car_2"].oscillation == pytest.approx(-0.1)


def test_agent_removal_preserves_collision_history(
    manager: SimulationManager,
) -> None:
    """Persistent collisions between surviving agents remain tracked."""
    manager.add_vehicle("car_1", VehicleState(x=50.0))
    manager.add_vehicle("car_2", VehicleState(x=10.0))
    manager.add_vehicle("car_3", VehicleState(x=11.0))

    calculator = RewardCalculator()
    calculator.reset(manager)

    manager.step(
        {agent: ControlInput() for agent in manager.vehicle_ids},
        dt=0.1,
    )

    first_rewards = calculator.calculate(
        manager,
        {agent: np.zeros(2) for agent in manager.vehicle_ids},
    )

    assert first_rewards["car_2"].collision == pytest.approx(-5.0)
    assert first_rewards["car_3"].collision == pytest.approx(-5.0)

    manager.remove_vehicle("car_1")
    calculator.remove_agents(("car_1",))

    manager.step(
        {agent: ControlInput() for agent in manager.vehicle_ids},
        dt=0.1,
    )

    rewards = calculator.calculate(
        manager,
        {agent: np.zeros(2) for agent in manager.vehicle_ids},
    )

    assert rewards["car_2"].collision == 0.0
    assert rewards["car_3"].collision == 0.0


def test_agent_removal_preserves_overtaking_history(
    manager: SimulationManager,
) -> None:
    """Surviving agents retain their confirmed relative ordering."""
    manager.add_vehicle(
        "car_1",
        VehicleState(x=80.0),
    )
    manager.add_vehicle(
        "car_2",
        VehicleState(x=10.0, y=-2.0, speed_mps=10.0),
    )
    manager.add_vehicle(
        "car_3",
        VehicleState(x=14.0, y=2.0),
    )

    calculator = RewardCalculator()
    calculator.reset(manager)

    manager.step(
        {agent: ControlInput() for agent in manager.vehicle_ids},
        dt=0.4,
    )

    rewards = calculator.calculate(
        manager,
        {agent: np.zeros(2) for agent in manager.vehicle_ids},
    )

    assert rewards["car_2"].overtaking == 0.0

    manager.remove_vehicle("car_1")
    calculator.remove_agents(("car_1",))

    manager.step(
        {agent: ControlInput() for agent in manager.vehicle_ids},
        dt=0.3,
    )

    rewards = calculator.calculate(
        manager,
        {agent: np.zeros(2) for agent in manager.vehicle_ids},
    )

    assert rewards["car_2"].overtaking == pytest.approx(2.0)
    assert rewards["car_3"].overtaking == 0.0


def test_remove_unknown_reward_agent(
    manager: SimulationManager,
) -> None:
    """Invalid removal requests do not modify existing history."""
    manager.add_vehicle("car_1")
    manager.add_vehicle("car_2")

    calculator = RewardCalculator()
    calculator.reset(manager)

    with pytest.raises(ValueError, match="Unknown reward agents"):
        calculator.remove_agents(("car_1", "unknown"))

    manager.step(
        {agent: ControlInput() for agent in manager.vehicle_ids},
        dt=0.1,
    )

    rewards = calculator.calculate(
        manager,
        {agent: np.zeros(2) for agent in manager.vehicle_ids},
    )

    assert set(rewards) == {"car_1", "car_2"}


def test_remove_multiple_reward_agents(
    manager: SimulationManager,
) -> None:
    """Multiple finished agents can be removed simultaneously."""
    manager.add_vehicle("car_1", VehicleState(x=10.0))
    manager.add_vehicle("car_2", VehicleState(x=30.0))
    manager.add_vehicle("car_3", VehicleState(x=50.0))

    calculator = RewardCalculator()
    calculator.reset(manager)

    manager.remove_vehicle("car_1")
    manager.remove_vehicle("car_2")

    calculator.remove_agents(("car_1", "car_2"))

    manager.step({"car_3": ControlInput()}, dt=0.1)

    rewards = calculator.calculate(
        manager,
        {"car_3": np.zeros(2)},
    )

    assert set(rewards) == {"car_3"}
    assert math.isfinite(rewards["car_3"].total)
