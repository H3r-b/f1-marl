"""Tests for the PettingZoo Parallel racing environment."""

from dataclasses import replace

import numpy as np
import pytest
from gymnasium.spaces import Box
from pettingzoo.test import parallel_api_test

from f1_marl.environment import ParallelRacingEnv, RaceConfig
from f1_marl.environment.rewards import RewardConfig
from f1_marl.simulation import RaceScenario, create_oval_track
from f1_marl.simulation.collision import VehicleCollisionEvent


@pytest.fixture
def scenario():
    return RaceScenario(track=create_oval_track())


@pytest.fixture
def env(scenario):
    return ParallelRacingEnv(
        scenario,
        dt=0.05,
        max_episode_steps=100,
    )


def test_environment_initialization(env):
    assert env.possible_agents == ["car_1", "car_2", "car_3", "car_4"]
    assert env.agents == []


def test_observation_and_action_spaces(env):
    for agent in env.possible_agents:
        observation_space = env.observation_space(agent)
        action_space = env.action_space(agent)

        assert isinstance(observation_space, Box)
        assert observation_space.shape == (29,)
        assert observation_space.dtype == np.float32

        assert isinstance(action_space, Box)
        assert action_space.shape == (2,)
        assert action_space.dtype == np.float32


def test_reset_returns_valid_observations_and_infos(env):
    observations, infos = env.reset(seed=42)

    assert set(observations) == set(env.possible_agents)
    assert set(infos) == set(env.possible_agents)

    for observation in observations.values():
        assert observation.shape == (29,)
        assert observation.dtype == np.float32
        assert np.all(np.isfinite(observation))
        assert np.all(observation >= -1.0)
        assert np.all(observation <= 1.0)


def test_reset_reinitializes_episode(env):
    env.reset(seed=42)

    actions = {agent: np.zeros(2, dtype=np.float32) for agent in env.agents}

    env.step(actions)

    observations, _ = env.reset(seed=42)

    assert env._step_count == 0
    assert env._simulation.simulation_time_s == 0.0
    assert env.agents == env.possible_agents
    assert set(observations) == set(env.possible_agents)


def test_step_advances_simulation(env):
    env.reset(seed=42)

    actions = {agent: np.zeros(2, dtype=np.float32) for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)

    assert env._simulation.simulation_time_s == pytest.approx(0.05)
    assert env._step_count == 1

    assert set(observations) == set(env.possible_agents)
    assert all(reward == 0.0 for reward in rewards.values())
    assert not any(terminations.values())
    assert not any(truncations.values())

    assert all(info["step_count"] == 1 for info in infos.values())


def test_actions_outside_bounds_are_clipped(env):
    env.reset(seed=42)

    action = env._convert_action(np.array([2.0, -2.0]))

    assert action.steering_angle_rad == pytest.approx(
        env._simulation.parameters.max_steering_angle_rad
    )
    assert action.throttle == 0.0
    assert action.brake == 1.0


@pytest.mark.parametrize(
    "action",
    [
        np.array([np.nan, 0.0]),
        np.array([0.0, np.inf]),
        np.array([0.0]),
        np.array([0.0, 0.0, 0.0]),
    ],
)
def test_invalid_actions_are_rejected(env, action):
    env.reset(seed=42)

    with pytest.raises(ValueError):
        env._convert_action(action)


def test_step_requires_actions_for_all_active_agents(env):
    env.reset(seed=42)

    actions = {agent: np.zeros(2, dtype=np.float32) for agent in env.agents[:-1]}

    with pytest.raises(ValueError, match="Action IDs"):
        env.step(actions)


def test_episode_truncates_at_maximum_steps(scenario):
    env = ParallelRacingEnv(
        scenario,
        max_episode_steps=1,
    )

    env.reset(seed=42)

    actions = {agent: np.zeros(2, dtype=np.float32) for agent in env.agents}

    _, _, terminations, truncations, _ = env.step(actions)

    assert not any(terminations.values())
    assert all(truncations.values())
    assert env.agents == []


def test_parallel_api_compliance(scenario):
    env = ParallelRacingEnv(
        scenario,
        max_episode_steps=100,
    )

    parallel_api_test(env, num_cycles=20)


def test_step_observation_contains_previous_actions(env):
    env.reset(seed=42)

    actions = {agent: np.array([0.5, -0.25], dtype=np.float32) for agent in env.agents}

    observations, _, _, _, _ = env.step(actions)

    for observation in observations.values():
        assert observation[3] == pytest.approx(0.5)
        assert observation[4] == pytest.approx(-0.25)


def test_reset_clears_previous_actions(env):
    env.reset(seed=42)

    actions = {agent: np.array([0.5, -0.25], dtype=np.float32) for agent in env.agents}

    env.step(actions)

    observations, _ = env.reset(seed=42)

    for observation in observations.values():
        assert observation[3] == pytest.approx(0.0)
        assert observation[4] == pytest.approx(0.0)


def test_step_returns_finite_rewards(
    env: ParallelRacingEnv,
) -> None:
    env.reset(seed=42)

    actions = {agent: np.array([0.0, 1.0], dtype=np.float32) for agent in env.agents}

    _, rewards, _, _, _ = env.step(actions)

    assert set(rewards) == set(actions)
    assert all(np.isfinite(value) for value in rewards.values())
    assert all(isinstance(value, float) for value in rewards.values())
    assert any(value > 0.0 for value in rewards.values())


def test_reward_components_are_exposed_in_infos(
    env: ParallelRacingEnv,
) -> None:
    env.reset(seed=42)

    actions = {agent: np.array([0.0, 1.0], dtype=np.float32) for agent in env.agents}

    _, rewards, _, _, infos = env.step(actions)

    expected_components = {
        "progress",
        "speed",
        "overtaking",
        "collision",
        "track_limits",
        "oscillation",
    }

    for agent in actions:
        components = infos[agent]["reward_components"]

        assert set(components) == expected_components
        assert all(np.isfinite(value) for value in components.values())
        assert rewards[agent] == pytest.approx(sum(components.values()))


def test_environment_accepts_custom_reward_configuration(
    scenario: RaceScenario,
) -> None:
    config = RewardConfig(
        progress_weight=0.0,
        speed_weight=0.0,
        overtaking_weight=0.0,
        collision_penalty=0.0,
        track_limit_penalty=0.0,
        oscillation_penalty=0.0,
    )

    env = ParallelRacingEnv(
        scenario,
        reward_config=config,
    )

    env.reset(seed=42)

    actions = {agent: np.array([0.0, 1.0], dtype=np.float32) for agent in env.agents}

    _, rewards, _, _, infos = env.step(actions)

    for agent in actions:
        assert rewards[agent] == 0.0
        assert all(value == 0.0 for value in infos[agent]["reward_components"].values())


def test_environment_rejects_invalid_reward_configuration(
    scenario: RaceScenario,
) -> None:
    with pytest.raises(TypeError, match="reward_config"):
        ParallelRacingEnv(
            scenario,
            reward_config={},
        )


def test_reset_clears_environment_reward_history(
    env: ParallelRacingEnv,
) -> None:
    env.reset(seed=42)

    actions = {agent: np.array([0.0, 1.0], dtype=np.float32) for agent in env.agents}

    _, first_rewards, _, _, first_infos = env.step(actions)

    reverse_actions = {agent: np.array([0.0, -1.0], dtype=np.float32) for agent in env.agents}

    env.step(reverse_actions)

    env.reset(seed=42)

    _, reset_rewards, _, _, reset_infos = env.step(actions)

    for agent in actions:
        assert reset_rewards[agent] == pytest.approx(first_rewards[agent])

        assert reset_infos[agent]["reward_components"] == pytest.approx(
            first_infos[agent]["reward_components"]
        )


def test_environment_default_race_configuration(scenario):
    """The environment defaults to a three-lap race."""
    env = ParallelRacingEnv(scenario)

    _, infos = env.reset(seed=42)

    for info in infos.values():
        assert info["race"]["total_laps"] == 3
        assert info["race"]["race_distance_m"] == pytest.approx(3.0 * scenario.track.length_m)


@pytest.mark.parametrize("total_laps", [1, 5, 10])
def test_environment_accepts_custom_race_configuration(
    scenario,
    total_laps,
):
    """The configured lap count determines the race distance."""
    env = ParallelRacingEnv(
        scenario,
        race_config=RaceConfig(total_laps=total_laps),
    )

    _, infos = env.reset(seed=42)

    for info in infos.values():
        assert info["race"]["total_laps"] == total_laps
        assert info["race"]["race_distance_m"] == pytest.approx(
            total_laps * scenario.track.length_m
        )


def test_environment_rejects_invalid_race_configuration(scenario):
    """Invalid race configuration objects are rejected."""
    with pytest.raises(TypeError, match="race_config"):
        ParallelRacingEnv(
            scenario,
            race_config={},
        )


def test_reset_initializes_race_classification(env):
    """Every agent receives a unique initial race position."""
    _, infos = env.reset(seed=42)

    positions = [infos[agent]["race"]["race_position"] for agent in env.agents]

    assert sorted(positions) == list(range(1, len(env.agents) + 1))

    assert env._race_lifecycle.finishing_order == ()

    for info in infos.values():
        race = info["race"]

        assert race["completed_laps"] == 0
        assert race["finished"] is False
        assert race["finish_position"] is None
        assert race["finish_time_s"] is None


def test_reset_returns_race_aware_observations(env):
    """Initial observations reflect the beginning of the race."""
    observations, infos = env.reset(seed=42)

    for agent, observation in observations.items():
        assert observation[5] == pytest.approx(0.0, abs=1e-6)
        assert observation[6] == pytest.approx(1.0)

        assert infos[agent]["race"]["completed_laps"] == 0
        assert env.observation_space(agent).contains(observation)


def test_environment_respects_configured_starting_line():
    """Race progress is relative to the configured starting line."""
    track = create_oval_track()

    scenario = RaceScenario(
        track=track,
        start_s_m=track.length_m / 4.0,
    )

    env = ParallelRacingEnv(scenario)

    observations, infos = env.reset(seed=42)

    for agent in env.agents:
        assert observations[agent][5] == pytest.approx(
            0.0,
            abs=1e-6,
        )
        assert observations[agent][6] == pytest.approx(1.0)
        assert infos[agent]["race"]["completed_laps"] == 0


@pytest.fixture
def controlled_race(scenario, monkeypatch):
    """Create an environment with controllable longitudinal positions."""

    def setup(total_laps=1, max_episode_steps=100):
        env = ParallelRacingEnv(
            scenario,
            race_config=RaceConfig(total_laps=total_laps),
            max_episode_steps=max_episode_steps,
        )

        env.reset(seed=42)

        simulation = env._simulation
        track_length = simulation.track.length_m

        original_interaction = simulation.get_track_interaction

        positions = {agent: original_interaction(agent).s for agent in env.agents}

        def get_track_interaction(agent):
            return replace(
                original_interaction(agent),
                s=positions[agent] % track_length,
            )

        monkeypatch.setattr(
            simulation,
            "get_track_interaction",
            get_track_interaction,
        )

        def advance(distances):
            for agent, distance in distances.items():
                positions[agent] += distance

            actions = {agent: np.zeros(2, dtype=np.float32) for agent in env.agents}

            return env.step(actions)

        return env, track_length, advance

    return setup


def test_environment_updates_race_progress(controlled_race):
    """Forward movement updates race progress and observations."""
    env, track_length, advance = controlled_race(
        total_laps=3,
    )

    agent = env.agents[0]
    distance = track_length / 4.0

    observations, _, terminations, truncations, infos = advance({agent: distance})

    race = infos[agent]["race"]

    assert race["cumulative_progress_m"] == pytest.approx(distance)
    assert race["completed_laps"] == 0
    assert observations[agent][5] == pytest.approx(0.25)
    assert observations[agent][6] == pytest.approx(11.0 / 12.0)
    assert not any(terminations.values())
    assert not any(truncations.values())


def test_environment_completes_intermediate_lap(controlled_race):
    """Completing an intermediate lap does not terminate an agent."""
    env, track_length, advance = controlled_race(
        total_laps=3,
    )

    agent = env.agents[0]

    for _ in range(5):
        observations, _, terminations, truncations, infos = advance({agent: track_length / 4.0})

    race = infos[agent]["race"]

    assert race["completed_laps"] == 1
    assert race["cumulative_progress_m"] == pytest.approx(1.25 * track_length)

    assert observations[agent][5] == pytest.approx(0.25)
    assert observations[agent][6] == pytest.approx(7.0 / 12.0)
    assert terminations[agent] is False
    assert truncations[agent] is False
    assert agent in env.agents


def test_environment_backward_progress(controlled_race):
    """Backward movement reduces cumulative race progress."""
    env, track_length, advance = controlled_race()

    agent = env.agents[0]

    advance({agent: track_length / 4.0})

    observations, _, terminations, _, infos = advance({agent: -track_length / 8.0})

    race = infos[agent]["race"]

    assert race["cumulative_progress_m"] == pytest.approx(track_length / 8.0)

    assert observations[agent][5] == pytest.approx(0.125)
    assert observations[agent][6] == pytest.approx(0.875)
    assert race["completed_laps"] == 0
    assert terminations[agent] is False


def test_environment_individual_race_completion(controlled_race):
    """An agent terminates naturally after completing its race."""
    env, track_length, advance = controlled_race()

    agent = env.agents[0]
    initial_agents = set(env.agents)

    for _ in range(5):
        observations, rewards, terminations, truncations, infos = advance(
            {agent: track_length * 0.21}
        )

    race = infos[agent]["race"]

    assert terminations[agent] is True
    assert truncations[agent] is False

    assert race["finished"] is True
    assert race["completed_laps"] == 1
    assert race["finish_position"] == 1
    assert race["finish_time_s"] is not None

    assert observations[agent][5] == 1.0
    assert observations[agent][6] == 0.0
    assert np.isfinite(rewards[agent])

    assert set(env.agents) == initial_agents - {agent}
    assert env._race_lifecycle.finishing_order == (agent,)

    for other_agent in env.agents:
        assert terminations[other_agent] is False
        assert truncations[other_agent] is False


def test_environment_preserves_finished_agent_transition(
    controlled_race,
):
    """Every participating agent receives its final transition."""
    env, track_length, advance = controlled_race()

    agent = env.agents[0]
    initial_agents = set(env.agents)

    for _ in range(5):
        transition = advance({agent: track_length * 0.21})

    observations, rewards, terminations, truncations, infos = transition

    assert set(observations) == initial_agents
    assert set(rewards) == initial_agents
    assert set(terminations) == initial_agents
    assert set(truncations) == initial_agents
    assert set(infos) == initial_agents

    assert terminations[agent] is True
    assert agent not in env.agents
    assert env.observation_space(agent).contains(observations[agent])

    assert rewards[agent] == pytest.approx(sum(infos[agent]["reward_components"].values()))


def test_finished_vehicle_removed_from_simulation(controlled_race):
    """Finished vehicles no longer participate in the simulation."""
    env, track_length, advance = controlled_race()

    agent = env.agents[0]

    for _ in range(5):
        advance({agent: track_length * 0.21})

    assert agent not in env.agents
    assert agent not in env._simulation.vehicle_ids
    assert agent not in env._simulation.states
    assert agent not in env._previous_actions

    assert set(env._simulation.vehicle_ids) == set(env.agents)

    # The original agent population remains unchanged.
    assert agent in env.possible_agents


def test_remaining_agents_continue_after_finish(controlled_race):
    """Remaining competitors continue racing after another agent finishes."""
    env, track_length, advance = controlled_race()

    winner = env.agents[0]
    remaining_agent = env.agents[1]

    for _ in range(5):
        advance({winner: track_length * 0.21})

    previous_progress = env._race_lifecycle.progress[remaining_agent].cumulative_progress_m

    previous_time = env._simulation.simulation_time_s

    observations, rewards, terminations, truncations, infos = advance(
        {remaining_agent: track_length * 0.1}
    )

    for result in (
        observations,
        rewards,
        terminations,
        truncations,
        infos,
    ):
        assert set(result) == set(env.agents)
        assert winner not in result

    assert infos[remaining_agent]["race"]["cumulative_progress_m"] == pytest.approx(
        previous_progress + track_length * 0.1
    )

    assert env._simulation.simulation_time_s == pytest.approx(previous_time + env.dt)

    assert all(np.isfinite(value) for value in rewards.values())
    assert not any(terminations.values())
    assert not any(truncations.values())


def test_finished_agent_actions_are_rejected(controlled_race):
    """Actions from finished agents are rejected."""
    env, track_length, advance = controlled_race()

    winner = env.agents[0]

    for _ in range(5):
        advance({winner: track_length * 0.21})

    actions = {agent: np.zeros(2, dtype=np.float32) for agent in env.possible_agents}

    previous_time = env._simulation.simulation_time_s

    with pytest.raises(ValueError, match="Action IDs"):
        env.step(actions)

    assert env._simulation.simulation_time_s == previous_time


def test_environment_sequential_race_completion(controlled_race):
    """Agents finish independently and retain their finishing positions."""
    env, track_length, advance = controlled_race()

    initial_agents = env.agents.copy()

    for position, agent in enumerate(initial_agents, start=1):
        for _ in range(5):
            _, _, terminations, truncations, infos = advance({agent: track_length * 0.22})

        assert terminations[agent] is True
        assert truncations[agent] is False

        assert infos[agent]["race"]["finish_position"] == position
        assert agent not in env.agents

        assert env._race_lifecycle.finishing_order == tuple(initial_agents[:position])

    assert env.agents == []
    assert env._simulation.vehicle_ids == ()

    for position, agent in enumerate(initial_agents, start=1):
        progress = env._race_lifecycle.progress[agent]

        assert progress.finished is True
        assert progress.finish_position == position
        assert progress.race_position == position


def test_environment_simultaneous_race_completion(controlled_race):
    """Multiple agents can finish during the same simulation step."""
    env, track_length, advance = controlled_race()

    initial_agents = env.agents.copy()

    for _ in range(5):
        _, _, terminations, truncations, infos = advance(
            {agent: track_length * 0.22 for agent in env.agents}
        )

    assert all(terminations.values())
    assert not any(truncations.values())

    assert env.agents == []
    assert env._simulation.vehicle_ids == ()

    finishing_order = env._race_lifecycle.finishing_order

    assert len(finishing_order) == len(initial_agents)
    assert set(finishing_order) == set(initial_agents)

    for position, agent in enumerate(finishing_order, start=1):
        assert infos[agent]["race"]["finish_position"] == position
        assert infos[agent]["race"]["finished"] is True


def test_step_rejected_after_natural_episode_completion(
    controlled_race,
):
    """A completed race must be reset before further steps."""
    env, track_length, advance = controlled_race()

    for _ in range(5):
        advance({agent: track_length * 0.22 for agent in env.agents})

    assert env.agents == []

    with pytest.raises(RuntimeError, match="episode has ended"):
        env.step({})


def test_race_completion_takes_precedence_over_truncation(
    controlled_race,
):
    """Natural completion takes precedence over the episode time limit."""
    env, track_length, advance = controlled_race(
        max_episode_steps=5,
    )

    winner = env.agents[0]
    initial_agents = env.agents.copy()

    for _ in range(5):
        observations, rewards, terminations, truncations, infos = advance(
            {winner: track_length * 0.21}
        )

    assert terminations[winner] is True
    assert truncations[winner] is False
    assert infos[winner]["race"]["finished"] is True

    for agent in initial_agents:
        if agent == winner:
            continue

        assert terminations[agent] is False
        assert truncations[agent] is True
        assert infos[agent]["race"]["finished"] is False
        assert infos[agent]["race"]["finish_position"] is None

    assert set(observations) == set(initial_agents)
    assert set(rewards) == set(initial_agents)
    assert env.agents == []


def test_truncation_after_individual_race_completion(
    controlled_race,
):
    """Remaining competitors truncate without changing earlier results."""
    env, track_length, advance = controlled_race(
        max_episode_steps=6,
    )

    winner = env.agents[0]

    for _ in range(5):
        advance({winner: track_length * 0.21})

    winner_progress = env._race_lifecycle.progress[winner]
    remaining_agents = set(env.agents)

    observations, rewards, terminations, truncations, infos = advance({})

    assert set(observations) == remaining_agents
    assert set(rewards) == remaining_agents
    assert set(infos) == remaining_agents

    assert not any(terminations.values())
    assert all(truncations.values())
    assert env.agents == []

    assert env._race_lifecycle.finishing_order == (winner,)
    assert env._race_lifecycle.progress[winner] == winner_progress


def test_reset_restores_finished_agents(controlled_race):
    """Reset restores the complete initial racing population."""
    env, track_length, advance = controlled_race()

    initial_agents = env.agents.copy()
    winner = initial_agents[0]

    for _ in range(5):
        advance({winner: track_length * 0.21})

    assert winner not in env.agents

    observations, infos = env.reset(seed=42)

    assert env.agents == initial_agents
    assert set(env._simulation.vehicle_ids) == set(initial_agents)
    assert set(observations) == set(initial_agents)
    assert set(infos) == set(initial_agents)

    assert env._race_lifecycle.finishing_order == ()
    assert env._step_count == 0
    assert env._simulation.simulation_time_s == 0.0

    for agent in initial_agents:
        race = infos[agent]["race"]

        assert race["finished"] is False
        assert race["completed_laps"] == 0
        assert race["finish_position"] is None
        assert race["finish_time_s"] is None
        assert env._previous_actions[agent] == (0.0, 0.0)


def test_environment_lifecycle_reset_is_deterministic(
    controlled_race,
):
    """Race initialization is reproducible after a completed episode."""
    env, track_length, advance = controlled_race()

    initial_observations, initial_infos = env.reset(seed=42)

    # Reinitialize the controlled fixture after capturing the baseline.
    env, track_length, advance = controlled_race()

    for _ in range(5):
        advance({agent: track_length * 0.22 for agent in env.agents})

    assert env.agents == []

    reset_observations, reset_infos = env.reset(seed=42)

    assert initial_infos == reset_infos

    for agent in env.possible_agents:
        np.testing.assert_array_equal(
            initial_observations[agent],
            reset_observations[agent],
        )


def test_parallel_api_compliance_with_race_completion(
    scenario,
    monkeypatch,
):
    """The Parallel API remains valid when agents finish independently."""
    env = ParallelRacingEnv(
        scenario,
        race_config=RaceConfig(total_laps=1),
        max_episode_steps=100,
    )

    original_reset = env.reset

    def reset(seed=None, options=None):
        observations, infos = original_reset(
            seed=seed,
            options=options,
        )

        simulation = env._simulation
        track_length = simulation.track.length_m
        original_interaction = simulation.get_track_interaction
        original_step = simulation.step

        positions = {agent: original_interaction(agent).s for agent in env.agents}

        increments = {
            agent: track_length * (0.22 - index * 0.03) for index, agent in enumerate(env.agents)
        }

        def get_track_interaction(agent):
            return replace(
                original_interaction(agent),
                s=positions[agent] % track_length,
            )

        def step(controls, dt):
            original_step(controls, dt)

            for agent in simulation.vehicle_ids:
                positions[agent] += increments[agent]

        monkeypatch.setattr(
            simulation,
            "get_track_interaction",
            get_track_interaction,
        )

        monkeypatch.setattr(simulation, "step", step)

        return observations, infos

    monkeypatch.setattr(env, "reset", reset)

    parallel_api_test(env, num_cycles=20)

    assert env.agents == []
    assert len(env._race_lifecycle.finishing_order) == len(env.possible_agents)


def test_final_step_collision_is_preserved(
    controlled_race,
    monkeypatch,
):
    """Preserve collision information when an involved agent finishes."""
    env, track_length, advance = controlled_race()

    winner = env.agents[0]
    opponent = env.agents[1]

    simulation = env._simulation

    for _ in range(4):
        advance({winner: track_length * 0.21})

    original_step = simulation.step

    def step_with_collision(controls, dt):
        original_step(controls, dt)

        simulation._last_events = (
            VehicleCollisionEvent(
                vehicle_id_a=winner,
                vehicle_id_b=opponent,
                simulation_time_s=simulation.simulation_time_s,
            ),
        )

    monkeypatch.setattr(
        simulation,
        "step",
        step_with_collision,
    )

    observations, rewards, terminations, truncations, infos = advance({winner: track_length * 0.21})

    assert terminations[winner] is True
    assert truncations[winner] is False
    assert terminations[opponent] is False

    # Both involved agents receive the final-step collision penalty.
    expected_penalty = -RewardConfig().collision_penalty

    for agent in (winner, opponent):
        assert infos[agent]["reward_components"]["collision"] == pytest.approx(expected_penalty)

        assert observations[agent][28] == pytest.approx(1.0)
        assert np.isfinite(rewards[agent])

    # Uninvolved agents must not receive a collision penalty.
    for agent in observations:
        if agent not in (winner, opponent):
            assert infos[agent]["reward_components"]["collision"] == 0.0
            assert observations[agent][28] == 0.0

    assert winner not in env.agents
    assert winner not in simulation.vehicle_ids
    assert opponent in env.agents


def test_finished_agent_collision_does_not_persist(
    controlled_race,
    monkeypatch,
):
    """Removed vehicles cannot generate subsequent collision penalties."""
    env, track_length, advance = controlled_race()

    winner = env.agents[0]
    opponent = env.agents[1]

    simulation = env._simulation

    for _ in range(4):
        advance({winner: track_length * 0.21})

    original_step = simulation.step

    def step_with_collision(controls, dt):
        original_step(controls, dt)

        simulation._last_events = (
            VehicleCollisionEvent(
                vehicle_id_a=winner,
                vehicle_id_b=opponent,
                simulation_time_s=simulation.simulation_time_s,
            ),
        )

    with monkeypatch.context() as patch:
        patch.setattr(
            simulation,
            "step",
            step_with_collision,
        )

        _, _, terminations, _, infos = advance({winner: track_length * 0.21})

        assert terminations[winner] is True
        assert infos[opponent]["reward_components"]["collision"] < 0.0

    # Resume normal simulation after the finished vehicle is removed.
    observations, rewards, terminations, truncations, infos = advance({})

    assert winner not in simulation.vehicle_ids
    assert winner not in observations
    assert winner not in rewards
    assert winner not in infos

    assert infos[opponent]["reward_components"]["collision"] == 0.0
    assert observations[opponent][28] == 0.0

    assert simulation.collision_events == ()
    assert not any(terminations.values())
    assert not any(truncations.values())
