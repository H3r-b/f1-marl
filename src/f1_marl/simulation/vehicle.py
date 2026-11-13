"""Vehicle state, physical parameters, and control inputs."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VehicleParameters:
    """Configurable physical parameters of a racing vehicle."""

    wheelbase_m: float = 3.6
    max_steering_angle_rad: float = 0.55
    max_acceleration_mps2: float = 9.0
    max_braking_mps2: float = 12.0

    def __post_init__(self) -> None:
        values = (
            self.wheelbase_m,
            self.max_steering_angle_rad,
            self.max_acceleration_mps2,
            self.max_braking_mps2,
        )

        if not all(math.isfinite(value) for value in values):
            raise ValueError("Vehicle parameters must be finite.")

        if self.wheelbase_m <= 0:
            raise ValueError("Wheelbase must be positive.")

        if not 0 < self.max_steering_angle_rad < math.pi / 2:
            raise ValueError("Maximum steering angle must be between 0 and pi/2.")

        if self.max_acceleration_mps2 <= 0:
            raise ValueError("Maximum acceleration must be positive.")

        if self.max_braking_mps2 <= 0:
            raise ValueError("Maximum braking must be positive.")


@dataclass(frozen=True, slots=True)
class VehicleState:
    """Dynamic state of a vehicle in Cartesian coordinates."""

    x: float = 0.0
    y: float = 0.0
    heading_rad: float = 0.0
    speed_mps: float = 0.0

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.heading_rad, self.speed_mps)

        if not all(math.isfinite(value) for value in values):
            raise ValueError("Vehicle state values must be finite.")

        if self.speed_mps < 0:
            raise ValueError("Vehicle speed cannot be negative.")


@dataclass(frozen=True, slots=True)
class ControlInput:
    """Normalized accelerator and brake inputs, and steering angle."""

    steering_angle_rad: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.steering_angle_rad):
            raise ValueError("Steering angle must be finite.")

        if not math.isfinite(self.throttle):
            raise ValueError("Throttle must be finite.")

        if not math.isfinite(self.brake):
            raise ValueError("Brake must be finite.")

        if not 0.0 <= self.throttle <= 1.0:
            raise ValueError("Throttle must be between 0 and 1.")

        if not 0.0 <= self.brake <= 1.0:
            raise ValueError("Brake must be between 0 and 1.")
