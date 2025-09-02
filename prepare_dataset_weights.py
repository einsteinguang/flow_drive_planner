from train_flow_model import training_dataloader
from flow_drive.utils.train_utils import load_params


if __name__ == "__main__":
    params_path = "flow_drive/config/config.yaml"
    params = load_params(params_path)

    print("Preparing dataset weights...")
    dataloader = training_dataloader(params, device='cpu')
    print("Dataset weights prepared.")