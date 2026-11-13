"""Kinematic bicycle vehicle dynamics."""

import math

from f1_marl.simulation.vehicle import (
    ControlInput,
    VehicleParameters,
    VehicleState,
)


def step_vehicle(
    state: VehicleState,
    control: ControlInput,
    parameters: VehicleParameters,
    dt: float,
) -> VehicleState:
    """Advance vehicle state by one simulation timestep.

    Uses a kinematic bicycle model with midpoint integration.

    Args:
        state: Current vehicle state.
        control: Steering, throttle, and brake commands.
        parameters: Vehicle physical parameters.
        dt: Simulation timestep in seconds.

    Returns:
        Updated vehicle state.

    Raises:
        ValueError: If the timestep or control inputs are invalid.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Timestep must be finite and positive.")

    steering = max(
        -parameters.max_steering_angle_rad,
        min(
            control.steering_angle_rad,
            parameters.max_steering_angle_rad,
        ),
    )

    acceleration = (
        control.throttle * parameters.max_acceleration_mps2
        - control.brake * parameters.max_braking_mps2
    )

    next_speed = max(0.0, state.speed_mps + acceleration * dt)

    average_speed = (state.speed_mps + next_speed) / 2.0

    yaw_rate = average_speed / parameters.wheelbase_m * math.tan(steering)

    next_heading = state.heading_rad + yaw_rate * dt

    midpoint_heading = state.heading_rad + yaw_rate * dt / 2.0

    next_x = state.x + average_speed * math.cos(midpoint_heading) * dt

    next_y = state.y + average_speed * math.sin(midpoint_heading) * dt

    return VehicleState(
        x=next_x,
        y=next_y,
        heading_rad=next_heading,
        speed_mps=next_speed,
    )
