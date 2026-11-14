"""Tests for the PettingZoo Parallel racing environment."""

import numpy as np
import pytest
from gymnasium.spaces import Box
from pettingzoo.test import parallel_api_test

from f1_marl.environment import ParallelRacingEnv
from f1_marl.simulation import RaceScenario, create_oval_track


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
        assert np.all(observation == 0.0)


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
