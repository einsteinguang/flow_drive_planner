<div align="center">
    <h2>FlowDrive:<br/>moderated flow matching with data balancing for trajectory planning
    <br/>
    <br/>
    <a href="https://arxiv.org/abs/2509.21961"><img src='https://img.shields.io/badge/arXiv-Page-aff'></a>
    </h2>
</div>

## News <a name="news"></a>
- **`2026/02/17`** Initial code release
- **`2025/09/26`** FlowDrive [paper](https://arxiv.org/abs/2509.21961) published on arXiv.

## Demo Figures & Videos
All demonstration GIFs correspond to figures in the paper.

### Figure 7: Challenging Scenarios on nuPlan (Rows 1–4)
| Row 1 | Row 2 |
|:-----:|:-----:|
| ![Fig7 Row1](videos/challenging_scenarios/Fig7_row1.gif) | ![Fig7 Row2](videos/challenging_scenarios/Fig7_row2.gif) |
| Row 3 | Row 4 |
| ![Fig7 Row3](videos/challenging_scenarios/Fig7_row3.gif) | ![Fig7 Row4](videos/challenging_scenarios/Fig7_row4.gif) |

### Figure 8: Challenging Scenarios on InterPlan (Rows 1–2)
| Row 1 | Row 2 |
|:-----:|:-----:|
| ![Fig8 Row1](videos/challenging_scenarios/Fig8_row1.gif) | ![Fig8 Row2](videos/challenging_scenarios/Fig8_row2.gif) |

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

## License <a name="license"></a>

All assets and code are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.

## Citation <a name="citation"></a>
If you find this work useful, please consider citing:
```bibtex
@article{wang2025flowdrivemoderatedflowmatching,
      title={FlowDrive: moderated flow matching with data balancing for trajectory planning}, 
      author={Lingguang Wang and Ömer Şahin Taş and Marlon Steiner and Christoph Stiller},
      year={2025},
      eprint={2509.21961},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.21961}, 
}
```

## Acknowledgement
FlowDrive Planner is greatly inspired by the following open-source projects and we reuse some code from them
: [Diffusion-Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner), [nuplan-devkit](https://github.com/motional/nuplan-devkit), [tuplan_garage](https://github.com/autonomousvision/tuplan_garage), [planTF](https://github.com/jchengai/planTF), [pluto](https://github.com/jchengai/pluto), [DiT](https://github.com/facebookresearch/DiT)