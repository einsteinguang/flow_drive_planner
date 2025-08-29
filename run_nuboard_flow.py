# Useful imports
import os
from pathlib import Path
import tempfile
import hydra
import sys

port = "6600"
is_interplan = False                          # Set to True if visualizing interplan results
nuplan_data_root = "path/to/nuplan/data"      # nuplan dataset absolute path (e.g. "/data")
nuplan_maps_root = "path/to/maps"             # nuplan maps absolute path (e.g. "/data/nuplan-v1.1/maps")
nuplan_exp_dir = "path/to/save/exp"           # Directory to save simulation results for nuplan and interplan
                                              # (e.g. .../nuplan_exp/)
nuplan_devkit_root = "path/to/nuplan-devkit"  # nuplan-devkit absolute path
result_folder = "path/to/exp/folder"          # simulation result absolute path
                                              # (e.g. .../nuplan_exp/exp/simulation/closed_loop_reactive_agents/flow_drive/val14/exp_name)

interplan_exp_dir = os.path.join(nuplan_exp_dir, "interplan_exp") # Directory to save simulation results for interplan

env_variables = {
    "NUPLAN_DEVKIT_ROOT": nuplan_devkit_root,
    "NUPLAN_DATA_ROOT": nuplan_data_root,
    "NUPLAN_MAPS_ROOT": nuplan_maps_root,
    "NUPLAN_EXP_ROOT": nuplan_exp_dir,
    "NUPLAN_SIMULATION_ALLOW_ANY_BUILDER":"1"
}

if is_interplan:
    env_variables["NUPLAN_EXP_ROOT"] = interplan_exp_dir

for k, v in env_variables.items():
    os.environ[k] = v

# Location of path with all nuBoard configs
CONFIG_PATH = f'{nuplan_devkit_root}/nuplan/planning/script/config/nuboard' # relative path to nuplan-devkit
CONFIG_NAME = 'default_nuboard'

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
hydra.initialize(config_path=CONFIG_PATH)

ml_planner_simulation_folder = result_folder
ml_planner_simulation_folder = [dp for dp, _, fn in os.walk(ml_planner_simulation_folder) if True in ['.nuboard' in x for x in fn]]

# Compose the configuration
cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
    f'scenario_builder=nuplan',  # set the database (same as simulation) used to fetch data for visualization
    f'simulation_path={ml_planner_simulation_folder}',  # nuboard file path(s), if left empty the user can open the file inside nuBoard
    'hydra.searchpath=[pkg://flow_drive.config.scenario_filter, pkg://flow_drive.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]',
    f'port_number={port}'
])

from nuplan.planning.script.run_nuboard import main as main_nuboard

# Run nuBoard
main_nuboard(cfg)