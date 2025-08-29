## Demo Videos
All videos locate in `flow_drive_planner/videos`

## Getting Started

- Setup the nuPlan dataset following the [offiical-doc](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html)

### Install dependencies

#### Install nuplan-devkit using the following commands or see the [official doc](https://nuplan-devkit.readthedocs.io/en/latest/installation.html)
```bash
git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
conda env create -f environment.yml
conda activate nuplan 
python -m pip install -e .
```
- Depending on your nuplan dataset location, you may need to add `splits` in the following two config files accordingly:
  - In `nuplan-devkit/nuplan/planning/script/config/common/scenario_builder/nuplan_challenge.yaml`, change the `data_root: ${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/test/` to `data_root: ${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/test/`.
  - In `nuplan-devkit/nuplan/planning/script/config/common/scenario_builder/nuplan.yaml`, change `data_root: ${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/trainval` to `data_root: ${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/trainval`.


#### setup InterPlan devkit
```bash
cd ..
git clone https://github.com/mh0797/interPlan.git && cd interPlan
python -m pip install -e .
```
- You may need to change the config file `interPlan/interplan/planning/script/config/common/scenario_builder/interplan.yaml`, change `data_root: ${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/trainval` to `data_root: ${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/test`.

#### setup TuPlan Garage
```bash
cd ..
git clone https://github.com/autonomousvision/tuplan_garage.git && cd tuplan_garage
python -m pip install -e .
```
- Replace the code `tuplan_garage/tuplan_garage/planning/simulation/planner/pdm_planner/observation/pdm_object_manager.py` by our code in `flow_drive_planner/assets/adapted_tuplan_code/pdm_object_manager.py`. The main fix with this code is that the objects predicted with constant velocity are now moving along their velocity direction, instead of the heading direction. This fix improves the performance of PDM and FlowDrive* significantly.

#### setup FlowDrive Planner
```bash
cd flow_drive_planner && python -m pip install -r requirements.txt
python -m pip install -e .
```

### Closed-loop Evaluation
- Modify the evaluation script `sim_flow_drive_planner.sh` accordingly
- Run the evaluation script
```bash
chmod +x sim_flow_drive_planner.sh
./sim_flow_drive_planner.sh
```

### Visualize the evaluation results
- Modify the visualization script `run_nuboard_flow.py` accordingly
- Run the visualization script `python run_nuboard_flow.py`

### Training
- Configure the dara processing script `data_process.sh` and preprocess the training data
```bash
chmod +x data_process.sh
./data_process.sh
```
- Configure the paths in `flow_drive_planner/flow_drive/config/config.yaml` accordingly
- Run the training code. Note that the first time you train with `weighted_sampling=True` in `flow_drive_planner/flow_drive/config/config.yaml`, it will take a while to compute the sampling weights and save the weights to the same folder as the processed training data.
```bash
chmod +x train.sh
./train.sh 0,1,2,3,4,5,6,7  # specify the GPU ids
```

## Acknowledgement
FlowDrive Planner is greatly inspired by the following open-source projects and we reuse some code from them
: [Diffusion-Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner), [nuplan-devkit](https://github.com/motional/nuplan-devkit), [tuplan_garage](https://github.com/autonomousvision/tuplan_garage), [planTF](https://github.com/jchengai/planTF), [pluto](https://github.com/jchengai/pluto), [DiT](https://github.com/facebookresearch/DiT)