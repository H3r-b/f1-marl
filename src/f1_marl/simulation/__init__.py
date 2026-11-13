"""Simulation components for F1-MARL."""

from f1_marl.simulation.collision import (
    TrackBoundaryEvent,
    VehicleCollisionEvent,
    VehicleFootprint,
    boxes_intersect,
    detect_vehicle_collisions,
    oriented_box_corners,
)
from f1_marl.simulation.manager import SimulationManager, VehicleSnapshot
from f1_marl.simulation.scenarios import (
    RaceScenario,
    StartingGrid,
    generate_starting_grid,
)
from f1_marl.simulation.track import FrenetCoordinate, Track, TrackSample
from f1_marl.simulation.tracks import create_oval_track, create_technical_track
from f1_marl.simulation.vehicle import ControlInput, VehicleParameters, VehicleState

__all__ = [
    "ControlInput",
    "FrenetCoordinate",
    "RaceScenario",
    "SimulationManager",
    "StartingGrid",
    "Track",
    "TrackBoundaryEvent",
    "TrackSample",
    "VehicleCollisionEvent",
    "VehicleFootprint",
    "VehicleParameters",
    "VehicleSnapshot",
    "VehicleState",
    "boxes_intersect",
    "create_oval_track",
    "create_technical_track",
    "detect_vehicle_collisions",
    "generate_starting_grid",
    "oriented_box_corners",
]
