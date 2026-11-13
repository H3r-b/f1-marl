"""Reusable racing scenarios and deterministic vehicle starting grids."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from f1_marl.simulation.collision import (
    VehicleFootprint,
    boxes_intersect,
    oriented_box_corners,
)
from f1_marl.simulation.manager import SimulationManager
from f1_marl.simulation.track import Track
from f1_marl.simulation.vehicle import VehicleParameters, VehicleState


@dataclass(frozen=True, slots=True)
class StartingGrid:
    """Immutable collection of vehicle starting states."""

    vehicle_states: tuple[tuple[str, VehicleState], ...]

    @property
    def vehicle_ids(self) -> tuple[str, ...]:
        """Return vehicle IDs in starting-grid order."""
        return tuple(vehicle_id for vehicle_id, _ in self.vehicle_states)

    @property
    def states(self) -> dict[str, VehicleState]:
        """Return a copy of the starting states as a dictionary."""
        return dict(self.vehicle_states)


def _validate_vehicle_ids(vehicle_ids: Sequence[str]) -> None:
    """Validate starting-grid vehicle identifiers."""
    if not vehicle_ids:
        raise ValueError("A starting grid requires at least one vehicle.")

    if any(not isinstance(vehicle_id, str) or not vehicle_id.strip() for vehicle_id in vehicle_ids):
        raise ValueError("Vehicle IDs must be non-empty strings.")

    if len(set(vehicle_ids)) != len(vehicle_ids):
        raise ValueError("Vehicle IDs must be unique.")


def _validate_vehicle_inside_track(
    state: VehicleState,
    track: Track,
    footprint: VehicleFootprint,
) -> bool:
    """Check that every vehicle corner lies within track boundaries."""
    for x, y in oriented_box_corners(state, footprint):
        coordinate = track.project(x, y)

        if abs(coordinate.d) > track.half_width_m + 1e-6:
            return False

    return True


def generate_starting_grid(
    track: Track,
    vehicle_ids: Sequence[str],
    *,
    longitudinal_spacing_m: float = 8.0,
    lateral_offsets_m: Sequence[float] = (-2.0, 2.0),
    initial_speed_mps: float = 0.0,
    start_s_m: float = 0.0,
    footprint: VehicleFootprint | None = None,
) -> StartingGrid:
    """Generate deterministic, collision-free vehicle starting states.

    Vehicles are placed in rows along the track. Lateral offsets are
    assigned in order and repeated for subsequent rows.

    The first row is positioned at start_s_m. Each subsequent row
    is placed farther behind it along the direction of travel.

    Raises:
        ValueError: If configuration is invalid or any vehicle would
            overlap another vehicle or extend beyond track boundaries.
    """
    if not track.closed:
        raise ValueError("Starting grids require a closed racing track.")

    _validate_vehicle_ids(vehicle_ids)

    if not math.isfinite(longitudinal_spacing_m) or longitudinal_spacing_m <= 0:
        raise ValueError("Longitudinal spacing must be finite and positive.")

    if not lateral_offsets_m:
        raise ValueError("At least one lateral offset is required.")

    if not all(math.isfinite(offset) for offset in lateral_offsets_m):
        raise ValueError("Lateral offsets must be finite.")

    if not math.isfinite(initial_speed_mps) or initial_speed_mps < 0:
        raise ValueError("Initial speed must be finite and non-negative.")

    if not math.isfinite(start_s_m):
        raise ValueError("Starting position must be finite.")

    footprint = footprint or VehicleFootprint()

    states: list[tuple[str, VehicleState]] = []

    for index, vehicle_id in enumerate(vehicle_ids):
        row = index // len(lateral_offsets_m)
        lane = index % len(lateral_offsets_m)

        longitudinal_position = start_s_m - row * longitudinal_spacing_m
        lateral_offset = float(lateral_offsets_m[lane])

        sample = track.sample(longitudinal_position)

        x = sample.x + sample.normal_x * lateral_offset
        y = sample.y + sample.normal_y * lateral_offset
        heading = math.atan2(sample.tangent_y, sample.tangent_x)

        state = VehicleState(
            x=x,
            y=y,
            heading_rad=heading,
            speed_mps=initial_speed_mps,
        )

        if not _validate_vehicle_inside_track(state, track, footprint):
            raise ValueError(f"Vehicle '{vehicle_id}' would extend beyond track boundaries.")

        for existing_id, existing_state in states:
            if boxes_intersect(
                state,
                existing_state,
                footprint,
                footprint,
            ):
                raise ValueError(
                    f"Vehicle '{vehicle_id}' would overlap vehicle "
                    f"'{existing_id}' in the starting grid."
                )

        states.append((vehicle_id, state))

    return StartingGrid(vehicle_states=tuple(states))


@dataclass(frozen=True, slots=True)
class RaceScenario:
    """Configuration for a reusable multi-vehicle racing scenario."""

    track: Track
    vehicle_ids: tuple[str, ...] = (
        "car_1",
        "car_2",
        "car_3",
        "car_4",
    )
    longitudinal_spacing_m: float = 8.0
    lateral_offsets_m: tuple[float, ...] = (-2.0, 2.0)
    initial_speed_mps: float = 0.0
    start_s_m: float = 0.0
    parameters: VehicleParameters | None = None
    footprint: VehicleFootprint | None = None

    def generate_grid(self) -> StartingGrid:
        """Generate the scenario's deterministic starting grid."""
        return generate_starting_grid(
            self.track,
            self.vehicle_ids,
            longitudinal_spacing_m=self.longitudinal_spacing_m,
            lateral_offsets_m=self.lateral_offsets_m,
            initial_speed_mps=self.initial_speed_mps,
            start_s_m=self.start_s_m,
            footprint=self.footprint,
        )

    def create_simulation(self) -> SimulationManager:
        """Create and populate a simulation manager for this scenario."""
        manager = SimulationManager(
            track=self.track,
            parameters=self.parameters,
            footprint=self.footprint,
        )

        for vehicle_id, state in self.generate_grid().vehicle_states:
            manager.add_vehicle(vehicle_id, state)

        return manager
