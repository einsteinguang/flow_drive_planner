import time
import torch
import torch.nn as nn
import numpy as np
from box import ConfigBox

from flow_drive.utils.dataset import ClusterStatsRetriever
from flow_drive.utils.normalizer import StateNormalizer, ObservationNormalizer
from flow_drive.utils.train_utils import \
    get_diffuser, get_encoder, get_noise_scheduler, load_trained_models, load_checkpoint_directly
from flow_drive.utils.infer_utils import \
    sample_action, sample_action_with_speed_and_lateral_offsets
from flow_drive.utils.post_processing import \
    smooth_trajectories_preset, bound_speed_and_acceleration


class FlowDrivePlanner(nn.Module):
    def __init__(self,
                 params: ConfigBox,
                 device: str = "cpu",
                 ckpt_path: str = "None",
                 mlflow_exp_name: str = None,
                 load_run_name: str = None,
                 load_epoch: int = 0):
        super().__init__()
        if ckpt_path != "None":
            self.encoder, self.decoder = load_checkpoint_directly(params, ckpt_path, device=device)
        elif load_run_name is not None and load_epoch > 0 and mlflow_exp_name is not None:
            params, self.encoder, self.decoder = \
                load_trained_models(params, mlflow_exp_name, load_run_name, load_epoch, device=device)
        else:
            self.encoder = get_encoder(params)
            self.decoder = get_diffuser(params)
        self.encoder = self.encoder.to(device).eval()
        self.decoder = self.decoder.to(device).eval()
        self.params = params
        self.device = device
        self.noise_scheduler = get_noise_scheduler(params)
        self.action_normalizer = StateNormalizer.from_json(params.data_processing)
        self.observation_normalizer = ObservationNormalizer.from_json(params.data_processing)
        self.cluster_retriever = ClusterStatsRetriever(params.data_processing.ego_future_clusters_path)

    def forward(self, inputs: dict):
        with torch.inference_mode():
            inputs = self.observation_normalizer(inputs)

            encoder_outputs = self.encoder(inputs)

            action_norm, _ = sample_action(
                self.params,
                self.decoder,
                encoder_outputs,
                self.noise_scheduler,
                inputs["ego_current_state"][:, :4],  # [B, 4], x, y, cos, sin
            )  # [B, future_len, 3 or 4]

            action = self.action_normalizer.inverse(action_norm)

            # transform cos and sin to heading
            heading = torch.arctan2(action[:, :, 3], action[:, :, 2])
            action = torch.cat([action[:, :, :2], heading.unsqueeze(-1)], dim=-1)  # [B, future_len, 3], x, y, yaw
        return action

    def plan_multiple_trajectories_with_moderated_offset(
            self,
            inputs: dict,
            speed_limit,  # [B], normalized speed limit between [0, 1]
            speed_offsets,
            lateral_offsets,
            ) -> torch.Tensor:
        """
        Plan multiple trajectories from the same input and return the one with the maximum reward.
        """
        with torch.inference_mode():
            ego_state_unnormalized = inputs["ego_current_state"].clone()  # [B, 4], x, y, cos, sin
            inputs = self.observation_normalizer(inputs)

            encoder_outputs = self.encoder(inputs)

            B, N, D = encoder_outputs["encoding"].shape
            T = self.params.diffuser.pred_horizon
            A = 4

            action_norm, _ = sample_action_with_speed_and_lateral_offsets(
                self.params,
                self.decoder,
                encoder_outputs,
                self.noise_scheduler,
                inputs["ego_current_state"][:, :6],  # [B, 4], x, y, cos, sin, vx, vy
                speed_offsets=speed_offsets,
                lateral_offsets=lateral_offsets
            )  # [S * B, future_len, 3 or 4]

            actions = self.action_normalizer.inverse(action_norm)  # [20 * B, T, A]
            actions = actions.view(-1, B, T, A)

            # transform cos and sin to heading
            heading = torch.arctan2(actions[:, :, :, 3], actions[:, :, :, 2])
            actions = torch.cat([actions[:, :, :, :2], heading.unsqueeze(-1)], dim=-1)

            # Add current ego state before smoothing and then remove it after
            # Convert ego_state_unnormalized to position format [x, y, yaw]
            ego_current_position = torch.stack([
                ego_state_unnormalized[:, 0],  # x
                ego_state_unnormalized[:, 1],  # y
                torch.atan2(ego_state_unnormalized[:, 3], ego_state_unnormalized[:, 2]),  # yaw
            ], dim=-1)  # [B, 3]

            # Expand to all trajectories and add as first timestep
            S = actions.shape[0]  # Number of sampled trajectories
            ego_current_expanded = ego_current_position[None, :, None, :].expand(S, -1, 1, -1)  # [S, B, 1, 3]
            actions_with_current = torch.cat([ego_current_expanded, actions], dim=2)  # [S, B, T+1, 3]

            # Options: "default", "light", "medium", "strong", "adaptive", "high_quality"
            actions = smooth_trajectories_preset(actions_with_current, preset="strong")[:, :, 1:, :]  # [S, B, T, 3]

            actions = bound_speed_and_acceleration(actions, ego_state_unnormalized, speed_limit)
        return actions
