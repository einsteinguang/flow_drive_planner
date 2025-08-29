import torch


def sample_action(params, noise_pred_net, obs_cond, sampler, ego_current_state):
    # ego_current_state: [B, 10] -> [B, 4], x, y, cos, sin
    sampler.set_timesteps(params.inference.flow_inference_iter)
    device = obs_cond["encoding"].device
    B = obs_cond["encoding"].shape[0]
    T = params.diffuser.pred_horizon
    A = 4

    x_t = torch.randn((B, T, A), device=device)  # [B, T, 4]
    x_t = torch.cat([ego_current_state.unsqueeze(1), x_t], dim=1)  # [B, T + 1, A]

    history = [x_t[:, 1:, :]]
    float_ts = sampler.timesteps
    int_ts = torch.linspace(
        0, sampler.config.num_train_timesteps - 1, steps=len(float_ts), device=device).long()
    for t_int, t_float in zip(int_ts, float_ts):
        with torch.no_grad():
            model_out = noise_pred_net(x_t, t_int, global_cond=obs_cond)
            x_t_minus_1 = sampler.step(model_out, t_float, x_t).prev_sample
            x_t_minus_1[:, 0, :] = ego_current_state  # Keep the first step as the current state
        x_t = x_t_minus_1.detach()
        history.append(x_t[:, 1:, :].detach())
    return x_t[:, 1:, :], history


def _apply_speed_and_lateral_adjustments(
        x_t,
        ego_current_state,
        speed_offsets,
        lateral_offsets,
        B, S, T,
        ego_current_speed):
    """
    Apply longitudinal and lateral adjustments to trajectories.

    Args:
        x_t: Trajectory tensor [S * B, T + 1, 4] (with ego state appended)
        ego_current_state: Current ego state [S * B, 4]
        speed_offsets: List of speed adjustment factors
        lateral_offsets: List of lateral offset values
        B: Batch size
        S: Number of offset combinations
        T: Prediction horizon
        ego_current_speed: Current speed of the ego vehicle [B]
    """
    # Extract trajectory points (skip ego state at index 0)
    trajectory = x_t[:, 1:, :].clone()  # [S * B, T, 4]

    # Reshape to separate samples and batches: [S, B, T, 4]
    trajectory = trajectory.view(S, B, T, 4)
    ego_state = ego_current_state.view(S, B, 4)  # [S, B, 4] (x, y, cos, sin)

    # Create offset combinations
    offset_combinations = []
    for i, speed_offset in enumerate(speed_offsets):
        for j, lateral_offset in enumerate(lateral_offsets):
            offset_combinations.append((speed_offset, lateral_offset))

    for s_idx, (speed_offset, lateral_offset) in enumerate(offset_combinations):
        # Get current sample trajectories: [B, T, 4]
        current_traj = trajectory[s_idx]  # [B, T, 4]
        current_ego_state = ego_state[s_idx]  # [B, 4]

        # 1. Use ego heading as longitudinal direction
        # Extract heading from ego state: cos and sin components
        ego_cos = current_ego_state[:, 2]  # [B]
        ego_sin = current_ego_state[:, 3]  # [B]
        longitudinal_unit = torch.stack([ego_cos, ego_sin], dim=-1)  # [B, 2]

        # Lateral direction (perpendicular to longitudinal)
        lateral_unit = torch.stack([-longitudinal_unit[:, 1], longitudinal_unit[:, 0]], dim=-1)  # [B, 2]

        # 2. Apply longitudinal stretching (speed adjustment) - BATCHED with negative offset protection
        if speed_offset != 1.0:
            # Compute average trajectory length for scaling
            last_point = current_traj[:, -1, :2]  # [B, 2]
            first_point = current_traj[:, 0, :2]  # [B, 2]
            trajectory_length = torch.norm(last_point - first_point, dim=-1, keepdim=True)  # [B, 1]
            trajectory_length = torch.clamp(trajectory_length, min=1e-6)

            stretch_per_step = trajectory_length * (speed_offset - 1.0) / T  # [B, 1]

            # Create time indices for all trajectory points: [T]
            time_indices = torch.arange(1, T + 1, dtype=torch.float32, device=x_t.device)  # [T]

            # Compute stretch amounts for all points: [B, T]
            stretch_amounts = stretch_per_step * time_indices.unsqueeze(0)  # [B, 1] * [1, T] = [B, T]

            # Apply stretching to all points at once: [B, T, 2]
            stretch_vecs = longitudinal_unit.unsqueeze(1) * stretch_amounts.unsqueeze(-1)  # [B, 1, 2] * [B, T, 1] = [B, T, 2]
            current_traj[:, :, :2] += stretch_vecs

        # 3. Apply lateral offset - BATCHED with velocity-dependent scaling
        if lateral_offset != 0:
            # Create progress values for all trajectory points: [T]
            lateral_progress = torch.arange(1, T + 1, dtype=torch.float32, device=x_t.device) / T  # [T]

            # Compute velocity-dependent lateral scaling factor
            ego_speed_batch = ego_current_speed  # [B]
            velocity_scale = 1.0 - 0.3 * torch.clamp(ego_speed_batch / 0.5, 0.0, 1.0)  # [B]

            # Apply velocity scaling to lateral offset
            scaled_lateral_offset = lateral_offset * velocity_scale  # [B]

            # Compute lateral displacements for all points: [B, T, 2]
            lateral_displacements = lateral_unit.unsqueeze(1) * (
                scaled_lateral_offset.unsqueeze(1).unsqueeze(-1) * lateral_progress.unsqueeze(0).unsqueeze(-1))  # [B, 1, 2] * [B, T, 1] = [B, T, 2]
            current_traj[:, :, :2] += lateral_displacements

        # Update trajectory
        trajectory[s_idx] = current_traj

    # Reshape back to original format and update x_t
    trajectory = trajectory.view(S * B, T, 4)
    x_t[:, 1:, :] = trajectory

    return x_t


def sample_action_with_speed_and_lateral_offsets(
        params,
        noise_pred_net,
        obs_cond,
        sampler,
        ego_current_state,
        speed_offsets,
        lateral_offsets
    ):
    sampler.set_timesteps(params.inference.flow_inference_iter)
    device = obs_cond["encoding"].device
    B, N, D = obs_cond["encoding"].shape
    T = params.diffuser.pred_horizon
    A = 4
    S = len(speed_offsets) * len(lateral_offsets)

    x_t = torch.randn((S * B, T, A), device=device)

    # Expand ego current state for all samples
    ego_current_v = ego_current_state[..., 4:6]  # [B, 2], vx, vy
    ego_current_speed = torch.norm(ego_current_v, dim=-1)  # [B]
    ego_current_state_expanded = \
        ego_current_state[..., :4].unsqueeze(0).expand(S, -1, -1).reshape(S * B, -1)
    x_t = torch.cat([ego_current_state_expanded.unsqueeze(1), x_t], dim=1)  # [S * B, T + 1, A]

    expanded_scene_embedding = {}
    # expand from [B, N, D] to [S * B, N, D]
    expanded_scene_embedding["encoding"] = \
        obs_cond["encoding"][None, :, :, :].expand(S, -1, -1, -1).reshape(S * B, N, D)
    # expand from [B, N] to [S * B, N]
    expanded_scene_embedding["mask"] = \
        obs_cond["mask"][None, :, :].expand(S, -1, -1).reshape(S * B, N)

    history = [x_t[:, 1:, :]]
    float_ts = sampler.timesteps  # e.g. tensor([σ_T, σ_t2, σ_t1, ...])
    int_ts = torch.linspace(
        0, sampler.config.num_train_timesteps - 1, steps=len(float_ts), device=device).long()

    applied_index = len(int_ts) // 2 - 1

    for step_idx, (t_int, t_float) in enumerate(zip(int_ts, float_ts)):
        with torch.no_grad():
            model_out = noise_pred_net(x_t, t_int, global_cond=expanded_scene_embedding)
            x_t_minus_1 = sampler.step(model_out, t_float, x_t).prev_sample
            x_t_minus_1[:, 0, :] = ego_current_state_expanded  # Keep the first step as the current state

        # Apply adjustments
        if step_idx == applied_index:
            x_t_minus_1 = _apply_speed_and_lateral_adjustments(
                x_t_minus_1,
                ego_current_state_expanded,
                speed_offsets,
                lateral_offsets,
                B, S, T,
                ego_current_speed
            )

        x_t = x_t_minus_1.detach()
        history.append(x_t[:, 1:, :].detach())

    # Extract the final trajectories
    final_trajectories = x_t[:, 1:, :]  # [S*B, T, A]

    return final_trajectories, history

