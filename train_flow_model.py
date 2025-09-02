import os
import cv2
import sys
import csv
import time
import torch
import mlflow
import imageio
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from box import ConfigBox
from catalyst.data.sampler import DistributedSamplerWrapper
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from flow_drive.utils.dataset import FlowDriveDataset
from flow_drive.utils.train_utils import \
    load_params, batch_to_tensor, get_diffuser, get_encoder, get_noise_scheduler, set_seed
from flow_drive.utils.loss_function import compute_batch_loss


ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)


def training_dataloader(params: ConfigBox, device: str) -> DataLoader:
    dataset = FlowDriveDataset(params.data_processing, device=device)
    print("Dataset length:", len(dataset))
    if params.data_processing.weighted_sampling:
        sample_weights = torch.tensor(dataset.weights, dtype=torch.float)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    else:
        sampler = None
    if torch.distributed.is_initialized():
        if sampler is not None:
            sampler = DistributedSamplerWrapper(
                sampler,
                shuffle=True,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank())
        else:
            sampler = DistributedSampler(
                dataset,
                shuffle=True,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank())
    dataloader = DataLoader(
        dataset,
        batch_size=params.train.batch_size,
        sampler=sampler,
        num_workers=params.train.num_workers,
        pin_memory=params.train.pin_memory,
        persistent_workers=params.train.persistent_workers
    )
    return dataloader


def train_gpu_adaptive(params: ConfigBox):
    """
    GPU-adaptive training function that ensures consistent results regardless of the number of GPUs.
    Uses gradient accumulation and adaptive learning rate scaling to maintain equivalence to 8-GPU training.
    """
    set_seed(params.train.seed)

    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = torch.distributed.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    print(f"Local gpu rank: {local_rank}, World size: {world_size}")

    # Define target configuration (8 GPUs with specific batch size)
    target_world_size = 8
    target_global_batch_size = params.train.batch_size * target_world_size

    # Calculate gradient accumulation steps to maintain effective batch size
    current_global_batch_size = params.train.batch_size * world_size
    gradient_accumulation_steps = max(1, target_global_batch_size // current_global_batch_size)
    effective_batch_size = current_global_batch_size * gradient_accumulation_steps

    if local_rank == 0:
        print(f"Target global batch size: {target_global_batch_size}")
        print(f"Current global batch size: {current_global_batch_size}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Effective batch size: {effective_batch_size}")

    # Initialize or resume MLflow
    mlflow.set_tracking_uri(ROOT_DIR + "/mlruns")

    if local_rank == 0:
        mlflow.start_run()
        print("New mlflow run: ", mlflow.get_tracking_uri())
        for key1, value1 in params.to_dict().items():
            for key2, value2 in value1.items():
                mlflow.log_param(key1 + "_" + key2, value2)
        # Log adaptive training parameters
        mlflow.log_param("adaptive_world_size", world_size)
        mlflow.log_param("adaptive_gradient_accumulation_steps", gradient_accumulation_steps)
        mlflow.log_param("adaptive_effective_batch_size", effective_batch_size)

    # Create dataloaders
    scene_encoder = get_encoder(params).to(device)
    noise_pred_net = get_diffuser(params).to(device)
    ema_encoder = EMAModel(parameters=scene_encoder.parameters(), power=params.train.ema_power)
    ema_decoder = EMAModel(parameters=noise_pred_net.parameters(), power=params.train.ema_power)

    noise_scheduler = get_noise_scheduler(params)
    train_dataloader = training_dataloader(params, device)

    scene_encoder = DDP(scene_encoder, device_ids=[local_rank]).train()
    noise_pred_net = DDP(noise_pred_net, device_ids=[local_rank]).train()

    # Adaptive learning rate: scale based on effective batch size instead of current world size
    lr_scaler = effective_batch_size / 1536
    opt_params = list(noise_pred_net.module.parameters()) + list(scene_encoder.module.parameters())
    optimizer = torch.optim.AdamW(
        params=opt_params,
        lr=params.train.learning_rate * lr_scaler,
        weight_decay=params.train.weight_decay
    )
    if local_rank == 0:
        print(f"Adaptive learning rate: {params.train.learning_rate * lr_scaler}")

    # Adjust scheduler for gradient accumulation
    total_steps = len(train_dataloader) * params.train.num_epochs // gradient_accumulation_steps
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=params.train.warmup_steps // gradient_accumulation_steps,
        num_training_steps=total_steps
    )

    # Load states if resuming
    start_epoch = 0
    step = 0
    accumulated_loss = 0.0

    # Training loop
    for epoch_idx in range(start_epoch, params.train.num_epochs + 1):
        epoch_loss = list()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch_idx:02d}", disable=local_rank != 0)

        for batch_idx, batch in enumerate(batch_iterator):
            inputs = batch_to_tensor(batch, device)
            inputs, ego_future, _ = train_dataloader.dataset.transform_inputs_tensor(inputs)

            # Use standard loss function
            loss, _ = compute_batch_loss(
                params, scene_encoder.module, noise_pred_net.module, noise_scheduler, inputs, ego_future, device)

            # Scale loss by gradient accumulation steps to maintain average
            loss = loss / gradient_accumulation_steps

            loss.backward()

            accumulated_loss += loss.item()

            # Perform optimizer step only after accumulating enough gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(opt_params, 5.0)

                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                ema_encoder.step(scene_encoder.module.parameters())
                ema_decoder.step(noise_pred_net.module.parameters())

                # Log the accumulated loss
                epoch_loss.append(accumulated_loss)
                batch_iterator.set_postfix({"loss": f"{accumulated_loss:.4f}"})

                step += 1  # Increment step counter for logging

                if local_rank == 0:
                    # Log all metrics to MLflow
                    mlflow.log_metric("loss", accumulated_loss, step=step)

                step += 1
                accumulated_loss = 0.0

        if local_rank == 0:  # Only the rank 0 process will log and save models
            # Log epoch-level metrics
            mlflow.log_metric("epoch_train_loss", np.mean(epoch_loss), step=epoch_idx)

            # Save and log checkpoint
            if epoch_idx % 10 == 0 and epoch_idx > 0:
                checkpoint_path = f"checkpoints/epoch_{epoch_idx}_checkpoint.pth"
                os.makedirs("checkpoints", exist_ok=True)
                torch.save({
                    # "scene_encoder_state_dict": scene_encoder.module.state_dict(),
                    # "noise_pred_net_state_dict": noise_pred_net.module.state_dict(),
                    "ema_encoder_state_dict": ema_encoder.state_dict(),
                    "ema_decoder_state_dict": ema_decoder.state_dict(),
                    }, checkpoint_path)

                # Log to MLflow
                mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
                os.remove(checkpoint_path)  # Clean up local copy after logging
                print(f"Checkpoint for epoch {epoch_idx} saved to MLflow artifacts.")

    if local_rank == 0:
        mlflow.end_run()
    # Clean up DDP
    dist.destroy_process_group()


if __name__ == "__main__":
    params_path = "flow_drive/config/config.yaml"
    params = load_params(params_path)

    # Create training dataloader once to initializing the weights computation
    dataloader = training_dataloader(params, device='cpu')

    # Now start training
    train_gpu_adaptive(params)  # GPU-adaptive training function for consistent results across different GPU counts