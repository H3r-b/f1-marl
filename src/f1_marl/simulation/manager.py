"""Central manager for multi-vehicle racing simulations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from f1_marl.simulation.physics import step_vehicle
from f1_marl.simulation.track import Track
from f1_marl.simulation.track_interaction import (
    TrackInteraction,
    calculate_track_interaction,
)
from f1_marl.simulation.vehicle import (
    ControlInput,
    VehicleParameters,
    VehicleState,
)


@dataclass(frozen=True, slots=True)
class VehicleSnapshot:
    """Read-only snapshot of a vehicle and its track-relative state."""

    vehicle_id: str
    state: VehicleState
    track_interaction: TrackInteraction


class SimulationManager:
    """Manage multiple independently controlled vehicles on one track."""

    def __init__(
        self,
        track: Track,
        parameters: VehicleParameters | None = None,
    ) -> None:
        self.track = track
        self.parameters = parameters or VehicleParameters()

        self._states: dict[str, VehicleState] = {}
        self._initial_states: dict[str, VehicleState] = {}
        self._simulation_time_s = 0.0

    @property
    def simulation_time_s(self) -> float:
        """Elapsed simulation time in seconds."""
        return self._simulation_time_s

    @property
    def vehicle_ids(self) -> tuple[str, ...]:
        """Return the IDs of all currently registered vehicles."""
        return tuple(self._states)

    @property
    def states(self) -> dict[str, VehicleState]:
        """Return a copy of the current vehicle states."""
        return self._states.copy()

    def add_vehicle(
        self,
        vehicle_id: str,
        state: VehicleState | None = None,
    ) -> None:
        """Register a vehicle with an optional initial state."""
        if not isinstance(vehicle_id, str) or not vehicle_id.strip():
            raise ValueError("Vehicle ID must be a non-empty string.")

        if vehicle_id in self._states:
            raise ValueError(f"Vehicle '{vehicle_id}' already exists.")

        initial_state = state if state is not None else VehicleState()

        self._states[vehicle_id] = initial_state
        self._initial_states[vehicle_id] = initial_state

    def remove_vehicle(self, vehicle_id: str) -> None:
        """Remove a vehicle from the simulation."""
        if vehicle_id not in self._states:
            raise KeyError(f"Vehicle '{vehicle_id}' does not exist.")

        del self._states[vehicle_id]
        del self._initial_states[vehicle_id]

    def get_state(self, vehicle_id: str) -> VehicleState:
        """Return the current state of a registered vehicle."""
        try:
            return self._states[vehicle_id]
        except KeyError:
            raise KeyError(f"Vehicle '{vehicle_id}' does not exist.") from None

    def get_track_interaction(
        self,
        vehicle_id: str,
    ) -> TrackInteraction:
        """Calculate the current track-relative state of a vehicle."""
        state = self.get_state(vehicle_id)

        return calculate_track_interaction(state, self.track)

    def get_snapshot(self, vehicle_id: str) -> VehicleSnapshot:
        """Return a vehicle state and its track interaction together."""
        state = self.get_state(vehicle_id)

        return VehicleSnapshot(
            vehicle_id=vehicle_id,
            state=state,
            track_interaction=calculate_track_interaction(
                state,
                self.track,
            ),
        )

    def step(
        self,
        controls: Mapping[str, ControlInput],
        dt: float,
    ) -> None:
        """Advance all registered vehicles by one synchronized timestep.

        Exactly one control input must be provided for every registered
        vehicle. All next states are calculated before any are committed,
        so a failed update cannot partially advance the simulation.
        """
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Timestep must be finite and positive.")

        expected_ids = set(self._states)
        supplied_ids = set(controls)

        if supplied_ids != expected_ids:
            missing = expected_ids - supplied_ids
            unknown = supplied_ids - expected_ids

            raise ValueError(
                f"Control IDs must match registered vehicles. "
                f"Missing: {sorted(missing)}; "
                f"unknown: {sorted(unknown)}."
            )

        next_states = {
            vehicle_id: step_vehicle(
                state,
                controls[vehicle_id],
                self.parameters,
                dt,
            )
            for vehicle_id, state in self._states.items()
        }

        self._states = next_states
        self._simulation_time_s += dt

    def reset(self) -> None:
        """Restore all registered vehicles to their initial states."""
        self._states = self._initial_states.copy()
        self._simulation_time_s = 0.0
