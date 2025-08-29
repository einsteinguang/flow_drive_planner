import torch


def smooth_trajectories(actions: torch.Tensor, smoothing_method: str = "multi_pass",
                       num_passes: int = 3, window_size: int = 5) -> torch.Tensor:
    """
    Enhanced trajectory smoothing with multiple approaches.
    The first state is kept unchanged, only subsequent states are adjusted.

    Args:
        actions: [S, B, T, A] or [B, T, A], A = [x, y, yaw] where T=40
        smoothing_method: 'multi_pass', 'gaussian', 'adaptive', or 'savgol'
        num_passes: Number of smoothing passes (for multi_pass method)
        window_size: Window size for smoothing (must be odd)

    Returns:
        Smoothed trajectories with the same shape as input
    """
    if actions.dim() == 3:
        # Handle [B, T, A] case
        is_single_batch = True
        actions = actions.unsqueeze(0)  # Convert to [1, B, T, A]
    else:
        is_single_batch = False

    S, B, T, A = actions.shape
    assert T > 5, f"Expected T > 5 for trajectory smoothing, got T = {T}"

    # Reshape to [S*B, T, A] for vectorized processing
    actions_flat = actions.view(-1, T, A)  # [S*B, T, A]

    # Extract positions and yaw for all trajectories at once
    positions = actions_flat[:, :, :2]  # [S*B, T, 2]
    yaw = actions_flat[:, :, 2]  # [S*B, T]

    if smoothing_method == "multi_pass":
        smoothed_positions, smoothed_yaw = _multi_pass_smoothing(
            positions, yaw, num_passes)
    elif smoothing_method == "gaussian":
        smoothed_positions, smoothed_yaw = _gaussian_smoothing(
            positions, yaw, window_size)
    elif smoothing_method == "adaptive":
        smoothed_positions, smoothed_yaw = _adaptive_smoothing(
            positions, yaw)
    elif smoothing_method == "savgol":
        smoothed_positions, smoothed_yaw = _savgol_smoothing(
            positions, yaw, window_size)
    else:
        raise ValueError(f"Unknown smoothing method: {smoothing_method}")

    # Ensure first point remains unchanged (as required)
    smoothed_positions[:, 0] = positions[:, 0]
    smoothed_yaw[:, 0] = yaw[:, 0]

    # Reconstruct the smoothed actions
    smoothed_actions_flat = torch.cat([smoothed_positions, smoothed_yaw.unsqueeze(-1)], dim=-1)

    smoothed_actions = smoothed_actions_flat.view(S, B, T, A)

    if is_single_batch:
        return smoothed_actions.squeeze(0)  # Convert back to [B, T, A]
    else:
        return smoothed_actions


def _multi_pass_smoothing(positions: torch.Tensor, yaw: torch.Tensor,
                         num_passes: int = 3) -> tuple:
    """Apply multiple passes of 3-point moving average for stronger smoothing."""
    smoothed_positions = positions.clone()
    smoothed_yaw = yaw.clone()

    for _ in range(num_passes):
        # Apply 3-point moving average to positions (vectorized)
        temp_positions = smoothed_positions.clone()
        temp_positions[:, 1:-1] = (
            0.25 * smoothed_positions[:, :-2] +
            0.5 * smoothed_positions[:, 1:-1] +
            0.25 * smoothed_positions[:, 2:]
        )
        smoothed_positions = temp_positions

        # Handle yaw with angle wrapping (vectorized)
        temp_yaw = smoothed_yaw.clone()
        yaw_windows = smoothed_yaw.unfold(1, 3, 1)  # [S*B, T-2, 3] - sliding windows
        yaw_complex = torch.exp(1j * yaw_windows)  # [S*B, T-2, 3]
        avg_complex = torch.mean(yaw_complex, dim=2)  # [S*B, T-2]
        temp_yaw[:, 1:-1] = torch.angle(avg_complex)
        smoothed_yaw = temp_yaw

    return smoothed_positions, smoothed_yaw


def _gaussian_smoothing(positions: torch.Tensor, yaw: torch.Tensor,
                       window_size: int = 5) -> tuple:
    """Apply Gaussian smoothing with configurable window size."""
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Create Gaussian kernel
    sigma = window_size / 4.0
    kernel_range = torch.arange(window_size, device=positions.device).float()
    kernel_center = (window_size - 1) / 2.0
    gaussian_kernel = torch.exp(-0.5 * ((kernel_range - kernel_center) / sigma) ** 2)
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    smoothed_positions = positions.clone()
    smoothed_yaw = yaw.clone()

    # Apply Gaussian smoothing to positions
    half_window = window_size // 2
    for i in range(half_window, positions.shape[1] - half_window):
        window_positions = positions[:, i - half_window:i + half_window + 1]  # [S*B, window_size, 2]
        weighted_positions = window_positions * gaussian_kernel.view(1, -1, 1)
        smoothed_positions[:, i] = weighted_positions.sum(dim=1)

    # Apply Gaussian smoothing to yaw with angle wrapping
    for i in range(half_window, yaw.shape[1] - half_window):
        window_yaw = yaw[:, i - half_window:i + half_window + 1]  # [S*B, window_size]
        yaw_complex = torch.exp(1j * window_yaw)  # [S*B, window_size]
        weighted_complex = yaw_complex * gaussian_kernel.view(1, -1)
        avg_complex = weighted_complex.sum(dim=1)  # [S*B]
        smoothed_yaw[:, i] = torch.angle(avg_complex)

    return smoothed_positions, smoothed_yaw


def _adaptive_smoothing(positions: torch.Tensor, yaw: torch.Tensor) -> tuple:
    """Apply adaptive smoothing based on trajectory curvature."""
    smoothed_positions = positions.clone()
    smoothed_yaw = yaw.clone()

    # Calculate curvature as a proxy for how much smoothing to apply
    # Higher curvature = more smoothing needed
    dx = positions[:, 1:] - positions[:, :-1]  # [S*B, T-1, 2]
    velocity = torch.norm(dx, dim=-1)  # [S*B, T-1]

    # Avoid division by zero
    velocity = torch.clamp(velocity, min=1e-6)

    # Calculate acceleration (change in velocity direction)
    dv = dx[:, 1:] - dx[:, :-1]  # [S*B, T-2, 2]
    acceleration = torch.norm(dv, dim=-1)  # [S*B, T-2]

    # Curvature approximation: acceleration / velocity^2
    curvature = acceleration / (velocity[:, 1:] ** 2 + 1e-6)  # [S*B, T-2]

    # Normalize curvature to [0, 1] range for smoothing weights
    curvature_norm = torch.tanh(curvature * 10)  # Scale and normalize

    # Apply adaptive smoothing
    for i in range(1, positions.shape[1] - 1):
        if i - 1 < curvature_norm.shape[1]:
            # Adaptive weight based on curvature
            adaptive_weight = 0.1 + 0.4 * curvature_norm[:, i - 1]  # [S*B]
            adaptive_weight = adaptive_weight.unsqueeze(-1)  # [S*B, 1]

            # Apply weighted smoothing
            smoothed_positions[:, i] = (
                (1 - adaptive_weight) * positions[:, i] +
                adaptive_weight * 0.5 * (positions[:, i - 1] + positions[:, i + 1])
            )

            # Handle yaw with angle wrapping
            prev_yaw = yaw[:, i - 1]
            curr_yaw = yaw[:, i]
            next_yaw = yaw[:, i + 1]

            # Convert to complex for proper angle averaging
            prev_complex = torch.exp(1j * prev_yaw)
            curr_complex = torch.exp(1j * curr_yaw)
            next_complex = torch.exp(1j * next_yaw)

            avg_complex = 0.5 * (prev_complex + next_complex)
            smoothed_complex = (1 - adaptive_weight.squeeze(-1)) * curr_complex + adaptive_weight.squeeze(-1) * avg_complex
            smoothed_yaw[:, i] = torch.angle(smoothed_complex)

    return smoothed_positions, smoothed_yaw


def _savgol_smoothing(positions: torch.Tensor, yaw: torch.Tensor,
                     window_size: int = 5) -> tuple:
    """Apply Savitzky-Golay smoothing (polynomial fitting)."""
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Simple implementation of Savitzky-Golay filter with polynomial order 2
    half_window = window_size // 2

    # Create Savitzky-Golay coefficients for polynomial order 2
    # This is a simplified version - for window_size=5, these are the coefficients
    if window_size == 5:
        coeffs = torch.tensor([-3, 12, 17, 12, -3], device=positions.device, dtype=torch.float32) / 35
    elif window_size == 7:
        coeffs = torch.tensor([-2, 3, 6, 7, 6, 3, -2], device=positions.device, dtype=torch.float32) / 21
    else:
        # Fall back to simple moving average for other window sizes
        coeffs = torch.ones(window_size, device=positions.device, dtype=torch.float32) / window_size

    smoothed_positions = positions.clone()
    smoothed_yaw = yaw.clone()

    # Apply Savitzky-Golay smoothing to positions
    for i in range(half_window, positions.shape[1] - half_window):
        window_positions = positions[:, i - half_window:i + half_window + 1]  # [S*B, window_size, 2]
        weighted_positions = window_positions * coeffs.view(1, -1, 1)
        smoothed_positions[:, i] = weighted_positions.sum(dim=1)

    # Apply Savitzky-Golay smoothing to yaw with angle wrapping
    for i in range(half_window, yaw.shape[1] - half_window):
        window_yaw = yaw[:, i - half_window:i + half_window + 1]  # [S*B, window_size]
        yaw_complex = torch.exp(1j * window_yaw)  # [S*B, window_size]
        weighted_complex = yaw_complex * coeffs.view(1, -1)
        avg_complex = weighted_complex.sum(dim=1)  # [S*B]
        smoothed_yaw[:, i] = torch.angle(avg_complex)

    return smoothed_positions, smoothed_yaw


def smooth_trajectories_preset(actions: torch.Tensor, preset: str = "default") -> torch.Tensor:
    """
    Apply trajectory smoothing with predefined presets for common use cases.

    Args:
        actions: [S, B, T, A] or [B, T, A], A = [x, y, yaw] where T=40
        preset: Smoothing preset - "default", "light", "medium", "strong", "adaptive", "high_quality"

    Returns:
        Smoothed trajectories with the same shape as input
    """
    preset_configs = {
        "default": {"smoothing_method": "multi_pass", "num_passes": 2},
        "light": {"smoothing_method": "gaussian", "window_size": 3},
        "medium": {"smoothing_method": "gaussian", "window_size": 5},
        "strong": {"smoothing_method": "multi_pass", "num_passes": 4},
        "adaptive": {"smoothing_method": "adaptive"},
        "high_quality": {"smoothing_method": "savgol", "window_size": 5},
    }

    if preset not in preset_configs:
        raise ValueError(f"Unknown preset: {preset}. Available presets: {list(preset_configs.keys())}")

    config = preset_configs[preset]
    return smooth_trajectories(actions, **config)


def bound_speed_and_acceleration(actions: torch.Tensor, ego_state_unnormalized: torch.Tensor, speed_limits: torch.Tensor) -> torch.Tensor:
    """
    Vectorized implementation of acceleration and velocity bounding for trajectory points.
    """
    # Constants
    max_accel = 2.2    # m/s^2
    min_accel = -3.8  # m/s^2
    dt = 0.1
    eps = 1e-6

    S, B, T, _ = actions.shape
    SB = S * B

    # Flatten
    actions = actions.view(SB, T, 3)
    traj_pos = actions[:, :, :2]  # [SB, T, 2]

    # Expand ego state and speed limits
    ego_pos = ego_state_unnormalized[:, :2].repeat(S, 1)          # [SB, 2]
    ego_vel = ego_state_unnormalized[:, 4:6].repeat(S, 1)         # [SB, 2]
    ego_speed = torch.norm(ego_vel, dim=1)                        # [SB]
    if speed_limits.dim() == 1:
        speed_limits = speed_limits.unsqueeze(0).expand(S, -1)    # [S, B]
    speed_limits = speed_limits.reshape(SB)                       # [SB]

    # Build full positions [SB, T+1, 2]
    full_pos = torch.cat([ego_pos.unsqueeze(1), traj_pos], dim=1)  # [SB, T+1, 2]

    # ----- Phase 1: Acceleration constraint -----
    # Process multiple iterations to handle cascading effects
    for iteration in range(3):
        violations_fixed = 0

        for t in range(T):
            # Compute previous and current positions
            prev_pos = full_pos[:, t]     # [SB, 2]
            curr_pos = full_pos[:, t+1]   # [SB, 2]

            # Compute current segment velocity and speed
            curr_vel = (curr_pos - prev_pos) / dt  # [SB, 2]
            curr_speed = torch.norm(curr_vel, dim=1)  # [SB]

            if t == 0:
                # Special handling for first point: transition from ego velocity to first trajectory velocity
                prev_speed = ego_speed  # [SB]

                # Check both speed acceleration and vector acceleration
                speed_accel = (curr_speed - prev_speed) / dt  # [SB]

                # Vector acceleration (change in velocity vector)
                vector_accel = (curr_vel - ego_vel) / dt  # [SB, 2]
                vector_accel_mag = torch.norm(vector_accel, dim=1)  # [SB]

                # Check violations for both speed and vector acceleration
                speed_violated = (speed_accel > max_accel) | (speed_accel < min_accel)
                vector_violated = vector_accel_mag > max(abs(max_accel), abs(min_accel))
                violated = speed_violated | vector_violated

                if violated.any():
                    violations_fixed += violated.sum().item()

                    # Handle speed acceleration violations
                    target_speed = torch.where(
                        speed_accel > max_accel,
                        prev_speed + max_accel * dt,
                        torch.where(speed_accel < min_accel,
                                    prev_speed + min_accel * dt,
                                    curr_speed)
                    ).clamp(min=0.1)

                    # Handle vector acceleration violations
                    max_vel_change = max(abs(max_accel), abs(min_accel)) * dt
                    vel_change = curr_vel - ego_vel  # [SB, 2]
                    vel_change_mag = torch.norm(vel_change, dim=1)  # [SB]

                    # Scale down excessive velocity changes
                    scale_factor = torch.where(
                        vel_change_mag > max_vel_change,
                        max_vel_change / (vel_change_mag + eps),
                        torch.ones_like(vel_change_mag)
                    )

                    # Apply scaling only where vector acceleration is violated
                    limited_vel_change = vel_change * scale_factor.unsqueeze(1)
                    target_vel = ego_vel + limited_vel_change

                    # Use the more restrictive constraint
                    vector_target_speed = torch.norm(target_vel, dim=1)
                    final_target_speed = torch.where(
                        vector_violated & (vector_target_speed < target_speed),
                        vector_target_speed,
                        target_speed
                    ).clamp(min=0.1)

                    # Direction: use limited velocity direction when vector acceleration is violated
                    speed_direction = curr_vel / (curr_speed.unsqueeze(1) + eps)
                    vector_direction = target_vel / (torch.norm(target_vel, dim=1, keepdim=True) + eps)
                    direction = torch.where(
                        vector_violated.unsqueeze(1),
                        vector_direction,
                        speed_direction
                    )

                    # Update position
                    corrected_pos = prev_pos + direction * final_target_speed.unsqueeze(1) * dt
                    delta = corrected_pos - curr_pos
                    full_pos[:, t+1:] += delta.unsqueeze(1)

            else:
                # For other points, check speed acceleration only
                prev_vel = (prev_pos - full_pos[:, t-1]) / dt  # [SB, 2]
                prev_speed = torch.norm(prev_vel, dim=1)  # [SB]

                # Speed acceleration
                speed_accel = (curr_speed - prev_speed) / dt  # [SB]
                violated = (speed_accel > max_accel) | (speed_accel < min_accel)

                if violated.any():
                    violations_fixed += violated.sum().item()

                    # Target speed
                    target_speed = torch.where(
                        speed_accel > max_accel,
                        prev_speed + max_accel * dt,
                        torch.where(speed_accel < min_accel,
                                    prev_speed + min_accel * dt,
                                    curr_speed)
                    ).clamp(min=0.1)

                    # Direction from previous to current point
                    direction = curr_vel / (curr_speed.unsqueeze(1) + eps)

                    # Update position
                    corrected_pos = prev_pos + direction * target_speed.unsqueeze(1) * dt
                    delta = corrected_pos - curr_pos
                    full_pos[:, t+1:] += delta.unsqueeze(1)

        # If no violations were fixed, we're done
        if violations_fixed == 0:
            break

    # ----- Phase 2: Speed limit constraint -----
    # Process multiple iterations to handle cascading effects
    for iteration in range(3):
        violations_fixed = 0

        for t in range(T):
            prev_pos = full_pos[:, t]
            curr_pos = full_pos[:, t+1]
            delta_pos = curr_pos - prev_pos
            seg_dist = torch.norm(delta_pos, dim=1)
            seg_speed = seg_dist / dt

            # Check for speed violations
            too_fast = seg_speed > speed_limits
            if not too_fast.any():
                continue

            violations_fixed += too_fast.sum().item()

            # Calculate previous speed for deceleration constraint
            if t == 0:
                prev_speed = ego_speed
            else:
                prev_seg = full_pos[:, t] - full_pos[:, t-1]
                prev_speed = torch.norm(prev_seg, dim=1) / dt

            # Minimum speed allowed by deceleration bounds
            min_speed_due_to_decel = (prev_speed + min_accel * dt).clamp(min=0.1)

            # Target speed: respect both speed limit AND deceleration constraint
            # If deceleration constraint prevents reaching speed limit, use the minimum allowed speed
            target_speed = torch.where(
                min_speed_due_to_decel > speed_limits,
                min_speed_due_to_decel,  # Can't reach speed limit due to deceleration constraint
                speed_limits             # Can safely reach speed limit
            )

            # Don't increase speed if already below limit
            target_speed = torch.min(seg_speed, target_speed).clamp(min=0.1)

            # Direction (from previous to current position)
            dir_unit = delta_pos / (seg_dist.unsqueeze(1) + eps)

            # New position that respects constraints
            corrected_pos = prev_pos + dir_unit * target_speed.unsqueeze(1) * dt
            delta = corrected_pos - curr_pos

            # Apply shift to current and all future points
            full_pos[:, t+1:] += delta.unsqueeze(1)

        # If no violations were fixed, we're done
        if violations_fixed == 0:
            break

    # Return updated actions
    new_traj_pos = full_pos[:, 1:]  # [SB, T, 2]
    actions_adjusted = actions.clone()
    actions_adjusted[:, :, :2] = new_traj_pos
    return actions_adjusted.view(S, B, T, 3)