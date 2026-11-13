"""Simulation components for F1-MARL."""

from f1_marl.simulation.collision import (
    TrackBoundaryEvent,
    VehicleCollisionEvent,
    VehicleFootprint,
    boxes_intersect,
    detect_vehicle_collisions,
    oriented_box_corners,
)

__all__ = [
    "TrackBoundaryEvent",
    "VehicleCollisionEvent",
    "VehicleFootprint",
    "boxes_intersect",
    "detect_vehicle_collisions",
    "oriented_box_corners",
]
