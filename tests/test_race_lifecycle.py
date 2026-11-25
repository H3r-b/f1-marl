"""Tests for race progress and lifecycle management."""

import math
from types import SimpleNamespace

import pytest

from f1_marl.environment.race_lifecycle import (
    RaceConfig,
    RaceLifecycleManager,
    RaceProgress,
)
from f1_marl.simulation.manager import SimulationManager
from f1_marl.simulation.scenarios import RaceScenario
from f1_marl.simulation.track import Track
from f1_marl.simulation.tracks import create_oval_track
from f1_marl.simulation.vehicle import ControlInput, VehicleState


def test_default_race_config():
    """The default race consists of three laps."""
    assert RaceConfig().total_laps == 3


def test_custom_race_config():
    """Race length can be configured."""
    assert RaceConfig(total_laps=10).total_laps == 10


@pytest.mark.parametrize(
    "total_laps",
    [0, -1, 1.5, True, False, "3", None],
)
def test_invalid_race_config(total_laps):
    """Invalid lap counts must be rejected."""
    with pytest.raises(ValueError):
        RaceConfig(total_laps=total_laps)


def test_default_race_progress():
    """New progress snapshots contain neutral values."""
    progress = RaceProgress(agent_id="car_1")

    assert progress.agent_id == "car_1"
    assert progress.cumulative_progress_m == 0.0
    assert progress.completed_laps == 0
    assert progress.race_position == 0
    assert progress.finished is False
    assert progress.finish_position is None
    assert progress.finish_time_s is None


def test_invalid_lifecycle_config():
    """The lifecycle manager rejects invalid configuration types."""
    with pytest.raises(TypeError):
        RaceLifecycleManager(config={})


def test_lifecycle_initialization():
    """Reset initializes every racing agent."""
    scenario = RaceScenario(track=create_oval_track())
    simulation = scenario.create_simulation()

    lifecycle = RaceLifecycleManager()
    lifecycle.reset(
        simulation,
        start_s_m=scenario.start_s_m,
    )

    assert set(lifecycle.progress) == set(scenario.vehicle_ids)
    assert lifecycle.finishing_order == ()

    for progress in lifecycle.progress.values():
        assert progress.completed_laps == 0
        assert progress.finished is False
        assert progress.finish_position is None
        assert progress.finish_time_s is None


def test_configured_race_distance():
    """Race distance equals circuit length multiplied by lap count."""
    scenario = RaceScenario(track=create_oval_track())
    simulation = scenario.create_simulation()

    lifecycle = RaceLifecycleManager(RaceConfig(total_laps=5))
    lifecycle.reset(simulation)

    assert lifecycle.race_distance_m == pytest.approx(scenario.track.length_m * 5)


def test_initial_race_classification():
    """Initial classification respects starting-grid rows."""
    scenario = RaceScenario(track=create_oval_track())
    simulation = scenario.create_simulation()

    lifecycle = RaceLifecycleManager()
    lifecycle.reset(simulation)

    progress = lifecycle.progress

    assert {
        progress["car_1"].race_position,
        progress["car_2"].race_position,
    } == {1, 2}

    assert {
        progress["car_3"].race_position,
        progress["car_4"].race_position,
    } == {3, 4}


def test_rear_grid_has_negative_progress():
    """Vehicles behind the starting line have negative progress."""
    scenario = RaceScenario(track=create_oval_track())
    simulation = scenario.create_simulation()

    lifecycle = RaceLifecycleManager()
    lifecycle.reset(simulation)

    assert lifecycle.progress["car_3"].cumulative_progress_m < 0
    assert lifecycle.progress["car_4"].cumulative_progress_m < 0


def test_progress_mapping_is_defensive_copy():
    """External dictionary changes cannot modify lifecycle state."""
    scenario = RaceScenario(track=create_oval_track())
    simulation = scenario.create_simulation()

    lifecycle = RaceLifecycleManager()
    lifecycle.reset(simulation)

    progress = lifecycle.progress
    progress.clear()

    assert len(lifecycle.progress) == len(scenario.vehicle_ids)


def test_update_before_reset():
    """Lifecycle updates require initialization."""
    scenario = RaceScenario(track=create_oval_track())
    simulation = scenario.create_simulation()

    lifecycle = RaceLifecycleManager()

    with pytest.raises(RuntimeError, match="reset"):
        lifecycle.update(simulation)


def test_update_without_time_advancement():
    """Duplicate updates without simulation advancement are rejected."""
    scenario = RaceScenario(track=create_oval_track())
    simulation = scenario.create_simulation()

    lifecycle = RaceLifecycleManager()
    lifecycle.reset(simulation)

    with pytest.raises(ValueError, match="Simulation time"):
        lifecycle.update(simulation)


@pytest.fixture
def closed_track_manager() -> SimulationManager:
    """Create a deterministic closed-track simulation."""
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

    return SimulationManager(track)


def test_forward_race_progress(
    closed_track_manager: SimulationManager,
) -> None:
    """Forward movement increases cumulative race progress."""
    manager = closed_track_manager

    manager.add_vehicle(
        "car_1",
        VehicleState(x=10.0, speed_mps=20.0),
    )

    lifecycle = RaceLifecycleManager()
    lifecycle.reset(manager)

    manager.step({"car_1": ControlInput()}, dt=0.1)

    assert lifecycle.update(manager) == ()

    progress = lifecycle.progress["car_1"]

    assert progress.cumulative_progress_m == pytest.approx(12.0)
    assert progress.completed_laps == 0
    assert not progress.finished


def test_backward_race_progress(
    closed_track_manager: SimulationManager,
) -> None:
    """Backward movement decreases cumulative race progress."""
    manager = closed_track_manager

    manager.add_vehicle(
        "car_1",
        VehicleState(
            x=10.0,
            heading_rad=math.pi,
            speed_mps=20.0,
        ),
    )

    lifecycle = RaceLifecycleManager()
    lifecycle.reset(manager)

    manager.step({"car_1": ControlInput()}, dt=0.1)
    lifecycle.update(manager)

    progress = lifecycle.progress["car_1"]

    assert progress.cumulative_progress_m == pytest.approx(8.0)
    assert progress.completed_laps == 0


def test_initial_start_line_crossing(
    closed_track_manager: SimulationManager,
) -> None:
    """Crossing the starting line from behind does not complete a lap."""
    manager = closed_track_manager

    manager.add_vehicle(
        "car_1",
        VehicleState(
            x=0.0,
            y=2.0,
            heading_rad=-math.pi / 2,
            speed_mps=10.0,
        ),
    )

    lifecycle = RaceLifecycleManager()
    lifecycle.reset(manager)

    assert lifecycle.progress["car_1"].cumulative_progress_m == pytest.approx(-2.0)

    manager.step({"car_1": ControlInput()}, dt=0.2)
    lifecycle.update(manager)

    progress = lifecycle.progress["car_1"]

    assert progress.cumulative_progress_m == pytest.approx(0.0)
    assert progress.completed_laps == 0
    assert not progress.finished


def test_backward_start_line_crossing(
    closed_track_manager: SimulationManager,
) -> None:
    """Moving backward across the line produces negative progress."""
    manager = closed_track_manager

    manager.add_vehicle(
        "car_1",
        VehicleState(
            x=0.0,
            y=0.0,
            heading_rad=math.pi / 2,
            speed_mps=10.0,
        ),
    )

    lifecycle = RaceLifecycleManager()
    lifecycle.reset(manager)

    manager.step({"car_1": ControlInput()}, dt=0.2)
    lifecycle.update(manager)

    progress = lifecycle.progress["car_1"]

    assert progress.cumulative_progress_m == pytest.approx(-2.0)
    assert progress.completed_laps == 0
    assert not progress.finished


def test_dynamic_race_classification(
    closed_track_manager: SimulationManager,
) -> None:
    """Race positions change when a trailing vehicle moves ahead."""
    manager = closed_track_manager

    manager.add_vehicle(
        "car_1",
        VehicleState(x=18.0, y=2.0, speed_mps=5.0),
    )

    manager.add_vehicle(
        "car_2",
        VehicleState(x=10.0, y=-2.0, speed_mps=25.0),
    )

    lifecycle = RaceLifecycleManager()
    lifecycle.reset(manager)

    assert lifecycle.progress["car_1"].race_position == 1
    assert lifecycle.progress["car_2"].race_position == 2

    manager.step(
        {
            "car_1": ControlInput(),
            "car_2": ControlInput(),
        },
        dt=1.0,
    )

    lifecycle.update(manager)

    assert lifecycle.progress["car_2"].race_position == 1
    assert lifecycle.progress["car_1"].race_position == 2


@pytest.fixture
def controlled_race(
    closed_track_manager: SimulationManager,
    monkeypatch: pytest.MonkeyPatch,
):
    """Provide deterministic longitudinal movement for lifecycle tests."""
    manager = closed_track_manager

    positions: dict[str, float] = {}

    def get_track_interaction(agent: str):
        return SimpleNamespace(s=positions[agent] % manager.track.length_m)

    monkeypatch.setattr(
        manager,
        "get_track_interaction",
        get_track_interaction,
    )

    def setup(
        initial_positions: dict[str, float],
        total_laps: int = 3,
    ):
        positions.update(initial_positions)

        for agent in initial_positions:
            manager.add_vehicle(agent)

        lifecycle = RaceLifecycleManager(RaceConfig(total_laps=total_laps))

        lifecycle.reset(manager)

        def advance(
            new_positions: dict[str, float],
            dt: float = 1.0,
        ) -> tuple[str, ...]:
            positions.update(new_positions)

            manager.step(
                {agent: ControlInput() for agent in manager.vehicle_ids},
                dt=dt,
            )

            return lifecycle.update(manager)

        return manager, lifecycle, advance

    return setup


def test_multiple_completed_laps(controlled_race) -> None:
    """Completed laps increase after each full circuit."""
    _, lifecycle, advance = controlled_race(
        {"car_1": 0.0},
        total_laps=3,
    )

    for distance in range(100, 801, 100):
        assert advance({"car_1": float(distance)}) == ()

        progress = lifecycle.progress["car_1"]

        assert progress.cumulative_progress_m == pytest.approx(float(distance))
        assert progress.completed_laps == distance // 400
        assert not progress.finished


def test_repeated_start_line_crossings(
    controlled_race,
) -> None:
    """Oscillating across the starting line cannot create false laps."""
    _, lifecycle, advance = controlled_race(
        {"car_1": -5.0},
    )

    for _ in range(5):
        advance({"car_1": 5.0})
        advance({"car_1": -5.0})

        progress = lifecycle.progress["car_1"]

        assert progress.completed_laps == 0
        assert progress.cumulative_progress_m == pytest.approx(-5.0)
        assert not progress.finished


def test_backward_crossing_after_completed_lap(
    controlled_race,
) -> None:
    """Moving backward across the line reduces net completed laps."""
    _, lifecycle, advance = controlled_race(
        {"car_1": 0.0},
    )

    for distance in (100.0, 200.0, 300.0, 405.0):
        advance({"car_1": distance})

    assert lifecycle.progress["car_1"].completed_laps == 1

    advance({"car_1": 395.0})

    progress = lifecycle.progress["car_1"]

    assert progress.completed_laps == 0
    assert progress.cumulative_progress_m == pytest.approx(395.0)


def test_individual_race_completion(
    controlled_race,
) -> None:
    """An agent finishes after completing the configured race distance."""
    _, lifecycle, advance = controlled_race(
        {"car_1": 0.0},
        total_laps=1,
    )

    for distance in (100.0, 200.0, 300.0):
        assert advance({"car_1": distance}) == ()

    assert advance({"car_1": 400.0}) == ("car_1",)

    progress = lifecycle.progress["car_1"]

    assert progress.finished
    assert progress.completed_laps == 1
    assert progress.finish_position == 1
    assert progress.race_position == 1
    assert progress.finish_time_s == pytest.approx(4.0)
    assert lifecycle.finishing_order == ("car_1",)


def test_interpolated_finish_time(
    controlled_race,
) -> None:
    """Finish time is interpolated within the final simulation step."""
    _, lifecycle, advance = controlled_race(
        {"car_1": 0.0},
        total_laps=1,
    )

    for distance in (100.0, 200.0, 390.0):
        advance({"car_1": distance})

    assert advance(
        {"car_1": 410.0},
        dt=1.0,
    ) == ("car_1",)

    progress = lifecycle.progress["car_1"]

    assert progress.finish_time_s == pytest.approx(3.5)
    assert progress.finished


def test_simultaneous_finish_order(
    controlled_race,
) -> None:
    """Agents finishing in the same step are ranked by crossing time."""
    _, lifecycle, advance = controlled_race(
        {
            "car_1": 0.0,
            "car_2": 0.0,
        },
        total_laps=1,
    )

    for distance in (100.0, 200.0, 300.0):
        advance(
            {
                "car_1": distance,
                "car_2": distance,
            }
        )

    advance(
        {
            "car_1": 390.0,
            "car_2": 395.0,
        }
    )

    newly_finished = advance(
        {
            "car_1": 410.0,
            "car_2": 415.0,
        }
    )

    assert newly_finished == ("car_2", "car_1")
    assert lifecycle.finishing_order == ("car_2", "car_1")

    first = lifecycle.progress["car_2"]
    second = lifecycle.progress["car_1"]

    assert first.finish_position == 1
    assert second.finish_position == 2

    assert first.finish_time_s == pytest.approx(4.25)
    assert second.finish_time_s == pytest.approx(4.5)


def test_equal_finish_times_use_grid_order(
    controlled_race,
) -> None:
    """Exact finishing-time ties are resolved using grid order."""
    _, lifecycle, advance = controlled_race(
        {
            "car_2": 0.0,
            "car_1": 0.0,
        },
        total_laps=1,
    )

    for distance in (100.0, 200.0, 300.0):
        advance(
            {
                "car_2": distance,
                "car_1": distance,
            }
        )

    assert advance(
        {
            "car_2": 400.0,
            "car_1": 400.0,
        }
    ) == ("car_2", "car_1")

    assert lifecycle.finishing_order == ("car_2", "car_1")

    assert lifecycle.progress["car_2"].finish_time_s == (lifecycle.progress["car_1"].finish_time_s)


def test_remaining_agents_continue_after_finish(
    controlled_race,
) -> None:
    """Remaining agents continue after a finished vehicle is removed."""
    manager, lifecycle, advance = controlled_race(
        {
            "car_1": 0.0,
            "car_2": -10.0,
        },
        total_laps=1,
    )

    for distance in (100.0, 200.0, 300.0):
        advance(
            {
                "car_1": distance,
                "car_2": distance - 10.0,
            }
        )

    assert advance(
        {
            "car_1": 400.0,
            "car_2": 390.0,
        }
    ) == ("car_1",)

    first_finisher = lifecycle.progress["car_1"]

    manager.remove_vehicle("car_1")

    assert manager.vehicle_ids == ("car_2",)

    assert advance({"car_2": 395.0}) == ()

    assert lifecycle.progress["car_1"] == first_finisher
    assert lifecycle.progress["car_2"].race_position == 2
    assert not lifecycle.progress["car_2"].finished

    assert advance({"car_2": 405.0}) == ("car_2",)

    assert lifecycle.finishing_order == ("car_1", "car_2")
    assert lifecycle.progress["car_2"].finish_position == 2
    assert lifecycle.progress["car_1"] == first_finisher


def test_finished_vehicle_must_be_removed(
    controlled_race,
) -> None:
    """Finished vehicles cannot participate in subsequent updates."""
    _, _, advance = controlled_race(
        {"car_1": 0.0},
        total_laps=1,
    )

    for distance in (100.0, 200.0, 300.0, 400.0):
        advance({"car_1": distance})

    with pytest.raises(ValueError, match="unfinished racing agents"):
        advance({"car_1": 410.0})


def test_unexpected_vehicle_removal_is_rejected(
    controlled_race,
) -> None:
    """Removing an unfinished agent violates lifecycle consistency."""
    manager, _, advance = controlled_race(
        {
            "car_1": 0.0,
            "car_2": -10.0,
        }
    )

    manager.remove_vehicle("car_2")

    with pytest.raises(ValueError, match="unfinished racing agents"):
        advance({"car_1": 10.0})


def test_lifecycle_reset_clears_race_history(
    controlled_race,
) -> None:
    """Reset restores initial progress and clears finishing history."""
    manager, lifecycle, advance = controlled_race(
        {"car_1": 0.0},
        total_laps=1,
    )

    initial_progress = lifecycle.progress

    for distance in (100.0, 200.0, 300.0, 400.0):
        advance({"car_1": distance})

    assert lifecycle.progress["car_1"].finished
    assert lifecycle.finishing_order == ("car_1",)

    manager.reset()
    lifecycle.reset(manager)

    assert lifecycle.progress == initial_progress
    assert lifecycle.finishing_order == ()

    assert advance({"car_1": 10.0}) == ()

    progress = lifecycle.progress["car_1"]

    assert progress.cumulative_progress_m == pytest.approx(10.0)
    assert progress.completed_laps == 0
    assert not progress.finished
    assert progress.finish_position is None
    assert progress.finish_time_s is None


def test_open_track_is_rejected() -> None:
    """Race lifecycle management requires a closed circuit."""
    track = Track(
        [(0.0, 0.0), (100.0, 0.0)],
        half_width_m=5.0,
        closed=False,
    )

    manager = SimulationManager(track)
    manager.add_vehicle("car_1")

    lifecycle = RaceLifecycleManager()

    with pytest.raises(ValueError, match="closed track"):
        lifecycle.reset(manager)
