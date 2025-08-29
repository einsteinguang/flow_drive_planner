import torch
import random
import yaml
import io
import os
import json
import mlflow
import numpy as np

from mmengine import fileio
from box import ConfigBox

from diffusers.training_utils import EMAModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import \
    FlowMatchEulerDiscreteScheduler


def load_params(params_file):
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
        params = ConfigBox(params)
    return params


def load_logged_parameters(experiment_name: str, run_name: str):
    """Load the logged parameters from MLflow."""
    mlflow.set_tracking_uri(os.path.join(os.path.dirname(__file__), "..", "..", "mlruns"))  # Set the MLflow tracking URI
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    runs = mlflow.search_runs(experiment_ids=[experiment_id], filter_string="run_name = '{}'".format(run_name))
    # Get the parameters of the latest run from runs
    latest_run_id = runs.iloc[0].run_id
    run = mlflow.get_run(latest_run_id)
    params = run.data.params  # Dictionary of parameters
    return params, latest_run_id


def correct_params_from_logged_params(params: ConfigBox, logged_params: dict) -> ConfigBox:
    for key1, value1 in params.to_dict().items():
        if key1 == "ddpo":
            # Skip ddpo parameters, they are not logged
            continue
        for key2, value2 in value1.items():
            if key1 + "_" + key2 in logged_params:
                # Replace the value in params with the logged value and convert to the correct type
                if isinstance(value2, bool):
                    params[key1][key2] = logged_params[key1 + "_" + key2] == "True"
                elif isinstance(value2, int):
                    params[key1][key2] = int(logged_params[key1 + "_" + key2])
                elif isinstance(value2, float):
                    params[key1][key2] = float(logged_params[key1 + "_" + key2])
                elif isinstance(value2, list):
                    params[key1][key2] = list(logged_params[key1 + "_" + key2])
                else:
                    params[key1][key2] = str(logged_params[key1 + "_" + key2])
    return params


def openjson(path):
    value  = fileio.get_text(path)
    dict = json.loads(value)
    return dict


def opendata(path):
    npz_bytes = fileio.get(path)
    buff = io.BytesIO(npz_bytes)
    npz_data = np.load(buff)
    return npz_data


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def batch_to_tensor(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def batch_to_non_batched_numpy(batch):
    return {k: v.squeeze(0).cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def get_diffuser(params: ConfigBox):
    action_dim = 4
    pred_horizon = params.diffuser.pred_horizon + 1  # Append the current ego state
    from flow_drive.model.module.dit import DiT
    noise_pred_net = DiT(
        n_blocks=params.diffuser.n_blocks,
        action_dim=action_dim,
        pred_horizon=pred_horizon,
        hidden_dim=params.diffuser.hidden_dim,
        heads=params.diffuser.num_heads,
        dropout=params.diffuser.dropout
    )
    return noise_pred_net


def get_encoder(params: ConfigBox):
    from flow_drive.model.module.encoder import Encoder
    encoder = Encoder(params.encoder)
    return encoder


def load_checkpoint(run_id, epoch_number=None, artifact_subdir="checkpoints", device="cpu"):
    # Construct the artifact path for the checkpoints folder
    artifact_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "mlruns",
        "0", run_id, "artifacts", artifact_subdir)
    # List all checkpoint files
    checkpoint_files = [f for f in os.listdir(artifact_path) if f.endswith(".pth")]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {artifact_path}")

    # If epoch_number is provided, use it to locate the specific checkpoint
    if epoch_number is not None:
        checkpoint_filename = f"epoch_{epoch_number}_checkpoint.pth"
        if checkpoint_filename not in checkpoint_files:
            raise FileNotFoundError(f"No checkpoint file found for epoch {epoch_number}")
        else:
            checkpoint_path = os.path.join(artifact_path, checkpoint_filename)
    else:
        # Load the latest checkpoint based on epoch number in filename
        checkpoint_files.sort(key=lambda f: int(f.split("_")[1]))  # Assumes filenames like 'epoch_XX_checkpoint.pth'
        checkpoint_path = os.path.join(artifact_path, checkpoint_files[-1])
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def load_checkpoint_directly(params, ckpt_path, device="cpu"):
    scene_encoder = get_encoder(params)
    noise_pred_net = get_diffuser(params)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "ema_encoder_state_dict" in checkpoint:
        ema = EMAModel(parameters=scene_encoder.parameters(), power=params.train.ema_power)
        ema.load_state_dict(checkpoint['ema_encoder_state_dict'])
        ema.copy_to(scene_encoder.parameters())
    else:
        print("No scene encoder state dict found in checkpoint, using default encoder.")
    if "ema_state_dict" in checkpoint or "ema_decoder_state_dict" in checkpoint:
        key = 'ema_decoder_state_dict' if 'ema_decoder_state_dict' in checkpoint else 'ema_state_dict'
        ema = EMAModel(parameters=noise_pred_net.parameters(), power=params.train.ema_power)
        ema.load_state_dict(checkpoint[key])
        ema.copy_to(noise_pred_net.parameters())
    else:
        print("No noise prediction net state dict found in checkpoint, using default noise prediction net.")
    return scene_encoder, noise_pred_net


def load_from_checkpoints(run_id, epoch, params):
    scene_encoder = get_encoder(params)
    noise_pred_net = get_diffuser(params)
    checkpoint = load_checkpoint(run_id, epoch)
    if "ema_encoder_state_dict" in checkpoint:
        ema = EMAModel(parameters=scene_encoder.parameters(), power=params.train.ema_power)
        ema.load_state_dict(checkpoint['ema_encoder_state_dict'])
        ema.copy_to(scene_encoder.parameters())
    else:
        print("No scene encoder state dict found in checkpoint, using default encoder.")
    if "ema_state_dict" in checkpoint or "ema_decoder_state_dict" in checkpoint:
        key = 'ema_decoder_state_dict' if 'ema_decoder_state_dict' in checkpoint else 'ema_state_dict'
        ema = EMAModel(parameters=noise_pred_net.parameters(), power=params.train.ema_power)
        ema.load_state_dict(checkpoint[key])
        ema.copy_to(noise_pred_net.parameters())
    else:
        print("No noise prediction net state dict found in checkpoint, using default noise prediction net.")
    return scene_encoder, noise_pred_net


def load_trained_models(params: ConfigBox, mlflow_exp_name: str, run_name: str, epoch: int, device: str = "cpu"):
    logged_params, run_id = load_logged_parameters(mlflow_exp_name, run_name)
    params = correct_params_from_logged_params(params, logged_params)
    encoder, decoder = load_from_checkpoints(run_id, epoch, params)
    return params, encoder.to(device), decoder.to(device)


def get_noise_scheduler(params: ConfigBox):
    return FlowMatchEulerDiscreteScheduler(num_train_timesteps=params.inference.flow_train_iter)


def delete_ghost_runs(experiment_name: str):
    # Get active run IDs from MLflow
    mlflow.set_tracking_uri(os.path.join(os.path.dirname(__file__), "..", "..", "mlruns"))  # Set the MLflow tracking URI
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    active_runs = mlflow.search_runs(experiment_ids=[experiment_id])
    active_run_ids = set(active_runs['run_id'].tolist())

    print(f"Active runs: {active_run_ids}")

    # Path to the experiment directory
    experiment_path = os.path.join("mlruns", experiment_id)

    # Iterate over each folder in the experiment directory
    for run_folder in os.listdir(experiment_path):
        run_folder_path = os.path.join(experiment_path, run_folder)

        # Check if it's a folder and not an active run
        if os.path.isdir(run_folder_path) and run_folder not in active_run_ids:
            print(f"Deleting ghost run folder: {run_folder}")
            # Delete the folder and its contents
            try:
                for root, dirs, files in os.walk(run_folder_path, topdown=False):
                    for file in files:
                        os.remove(os.path.join(root, file))
                    for dir in dirs:
                        os.rmdir(os.path.join(root, dir))
                os.rmdir(run_folder_path)
                print(f"Deleted: {run_folder_path}")
            except Exception as e:
                print(f"Error deleting {run_folder_path}: {e}")



