"""Race progress, classification, and episode lifecycle management."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from f1_marl.simulation.manager import SimulationManager


@dataclass(frozen=True, slots=True)
class RaceConfig:
    """Configuration for a multi-agent racing episode."""

    total_laps: int = 3

    def __post_init__(self) -> None:
        """Validate race configuration."""
        if (
            isinstance(self.total_laps, bool)
            or not isinstance(self.total_laps, int)
            or self.total_laps <= 0
        ):
            raise ValueError("total_laps must be a positive integer.")


@dataclass(frozen=True, slots=True)
class RaceProgress:
    """Read-only snapshot of an agent's race progress."""

    agent_id: str
    cumulative_progress_m: float = 0.0
    completed_laps: int = 0
    race_position: int = 0
    finished: bool = False
    finish_position: int | None = None
    finish_time_s: float | None = None


class RaceLifecycleManager:
    """Manage race progress and classification across an episode."""

    def __init__(self, config: RaceConfig | None = None) -> None:
        if config is not None and not isinstance(config, RaceConfig):
            raise TypeError("config must be a RaceConfig.")

        self.config = config or RaceConfig()

        self._track_length_m: float = 0.0
        self._start_s_m: float = 0.0
        self._previous_time_s: float = 0.0

        self._previous_positions: dict[str, float] = {}
        self._progress: dict[str, RaceProgress] = {}
        self._grid_order: dict[str, int] = {}
        self._finishing_order: list[str] = []

    @property
    def race_distance_m(self) -> float:
        """Return the total configured race distance."""
        return self._track_length_m * self.config.total_laps

    @property
    def progress(self) -> dict[str, RaceProgress]:
        """Return a copy of all agent progress snapshots."""
        return self._progress.copy()

    @property
    def finishing_order(self) -> tuple[str, ...]:
        """Return agents in their final finishing order."""
        return tuple(self._finishing_order)

    @staticmethod
    def _signed_distance(
        current_s: float,
        reference_s: float,
        track_length_m: float,
    ) -> float:
        """Return the shortest signed distance on a closed track."""
        return float(
            (current_s - reference_s + track_length_m / 2) % track_length_m - track_length_m / 2
        )

    def reset(
        self,
        simulation: SimulationManager,
        *,
        start_s_m: float = 0.0,
    ) -> None:
        """Initialize race progress from the current starting grid."""
        if not simulation.track.closed:
            raise ValueError("Race lifecycle requires a closed track.")

        if not math.isfinite(start_s_m):
            raise ValueError("Starting position must be finite.")

        if not simulation.vehicle_ids:
            raise ValueError("A race requires at least one vehicle.")

        track_length_m = simulation.track.length_m

        race_distance_m = track_length_m * self.config.total_laps

        if not math.isfinite(race_distance_m):
            raise ValueError("Race distance must be finite.")

        self._track_length_m = track_length_m
        self._start_s_m = start_s_m % track_length_m
        self._previous_time_s = simulation.simulation_time_s

        self._finishing_order = []

        self._grid_order = {agent: index for index, agent in enumerate(simulation.vehicle_ids)}

        self._previous_positions = {
            agent: simulation.get_track_interaction(agent).s for agent in simulation.vehicle_ids
        }

        initial_progress = {
            agent: self._signed_distance(
                position,
                self._start_s_m,
                self._track_length_m,
            )
            for agent, position in self._previous_positions.items()
        }

        ranked_agents = sorted(
            simulation.vehicle_ids,
            key=lambda agent: (
                -initial_progress[agent],
                self._grid_order[agent],
            ),
        )

        self._progress = {
            agent: RaceProgress(
                agent_id=agent,
                cumulative_progress_m=initial_progress[agent],
                race_position=position,
            )
            for position, agent in enumerate(ranked_agents, start=1)
        }

    def _update_progress(
        self,
        simulation: SimulationManager,
    ) -> None:
        """Update cumulative progress for every active racing agent."""
        for agent in simulation.vehicle_ids:
            previous_s = self._previous_positions[agent]
            current_s = simulation.get_track_interaction(agent).s

            progress_delta = self._signed_distance(
                current_s,
                previous_s,
                self._track_length_m,
            )

            previous_progress = self._progress[agent]

            cumulative_progress = previous_progress.cumulative_progress_m + progress_delta

            completed_laps = min(
                self.config.total_laps,
                max(
                    0,
                    math.floor(cumulative_progress / self._track_length_m),
                ),
            )

            self._progress[agent] = replace(
                previous_progress,
                cumulative_progress_m=cumulative_progress,
                completed_laps=completed_laps,
            )

            self._previous_positions[agent] = current_s

    def _detect_finishers(
        self,
        previous_progress: dict[str, RaceProgress],
        current_time_s: float,
    ) -> tuple[str, ...]:
        """Detect newly finished agents and assign finishing positions."""
        candidates: list[tuple[float, int, str]] = []

        race_distance = self.race_distance_m
        elapsed_time = current_time_s - self._previous_time_s

        for agent, progress in self._progress.items():
            if progress.finished:
                continue

            if progress.cumulative_progress_m < race_distance:
                continue

            previous_distance = previous_progress[agent].cumulative_progress_m
            current_distance = progress.cumulative_progress_m

            distance_delta = current_distance - previous_distance

            if distance_delta <= 0:
                continue

            crossing_fraction = (race_distance - previous_distance) / distance_delta

            crossing_fraction = max(0.0, min(1.0, crossing_fraction))

            finish_time = self._previous_time_s + crossing_fraction * elapsed_time

            candidates.append(
                (
                    finish_time,
                    self._grid_order[agent],
                    agent,
                )
            )

        candidates.sort()

        newly_finished: list[str] = []

        for finish_time, _, agent in candidates:
            self._finishing_order.append(agent)

            finish_position = len(self._finishing_order)

            self._progress[agent] = replace(
                self._progress[agent],
                completed_laps=self.config.total_laps,
                race_position=finish_position,
                finished=True,
                finish_position=finish_position,
                finish_time_s=finish_time,
            )

            newly_finished.append(agent)

        return tuple(newly_finished)

    def _update_classification(self) -> None:
        """Update race positions while preserving final finishing positions."""
        active_agents = [
            agent for agent, progress in self._progress.items() if not progress.finished
        ]

        ranked_agents = sorted(
            active_agents,
            key=lambda agent: (
                -self._progress[agent].cumulative_progress_m,
                self._grid_order[agent],
            ),
        )

        first_active_position = len(self._finishing_order) + 1

        for position, agent in enumerate(
            ranked_agents,
            start=first_active_position,
        ):
            self._progress[agent] = replace(
                self._progress[agent],
                race_position=position,
            )

    def update(
        self,
        simulation: SimulationManager,
    ) -> tuple[str, ...]:
        """Update race progress and return newly finished agents."""
        if not self._progress:
            raise RuntimeError("Call reset() before update().")

        if not simulation.track.closed:
            raise ValueError("Race lifecycle requires a closed track.")

        if simulation.track.length_m != self._track_length_m:
            raise ValueError("Simulation track length has changed.")

        expected_agents = {
            agent for agent, progress in self._progress.items() if not progress.finished
        }

        if set(simulation.vehicle_ids) != expected_agents:
            raise ValueError("Simulation vehicles must match unfinished racing agents.")

        current_time_s = simulation.simulation_time_s

        if not math.isfinite(current_time_s) or current_time_s <= self._previous_time_s:
            raise ValueError("Simulation time must advance before updating race progress.")

        previous_progress = self._progress.copy()

        self._update_progress(simulation)

        newly_finished = self._detect_finishers(
            previous_progress,
            current_time_s,
        )

        self._update_classification()

        self._previous_time_s = current_time_s

        return newly_finished
