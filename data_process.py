import os
import argparse
import json

from flow_drive.data_process.data_processor import DataProcessor
from flow_drive.utils.train_utils import load_params

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder

def get_filter_parameters(num_scenarios_per_type=None, limit_total_scenarios=None, shuffle=True, scenario_tokens=None, log_names=None):

    scenario_types = None

    scenario_tokens                      # List of scenario tokens to include
    log_names = log_names                # Filter scenarios by log names
    map_names = None                     # Filter scenarios by map names

    num_scenarios_per_type               # Number of scenarios per type
    limit_total_scenarios                # Limit total scenarios (float = fraction, int = num) - this filter can be applied on top of num_scenarios_per_type
    timestamp_threshold_s = None         # Filter scenarios to ensure scenarios have more than `timestamp_threshold_s` seconds between their initial lidar timestamps
    ego_displacement_minimum_m = None    # Whether to remove scenarios where the ego moves less than a certain amount

    expand_scenarios = True              # Whether to expand multi-sample scenarios to multiple single-sample scenarios
    remove_invalid_goals = False          # Whether to remove scenarios where the mission goal is invalid
    shuffle                              # Whether to shuffle the scenarios

    ego_start_speed_threshold = None     # Limit to scenarios where the ego reaches a certain speed from below
    ego_stop_speed_threshold = None      # Limit to scenarios where the ego reaches a certain speed from above
    speed_noise_tolerance = None         # Value at or below which a speed change between two timepoints should be ignored as noise.

    return scenario_types, scenario_tokens, log_names, map_names, num_scenarios_per_type, limit_total_scenarios, timestamp_threshold_s, ego_displacement_minimum_m, \
           expand_scenarios, remove_invalid_goals, shuffle, ego_start_speed_threshold, ego_stop_speed_threshold, speed_noise_tolerance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--data_path', default='/data/nuplan-v1.1/trainval', type=str, help='path to raw data')
    parser.add_argument('--map_path', default='/data/nuplan-v1.1/maps', type=str, help='path to map data')

    parser.add_argument('--save_path', default='./cache', type=str, help='path to save processed data')
    parser.add_argument('--scenarios_per_type', type=int, default=None, help='number of scenarios per type')
    parser.add_argument('--total_scenarios', type=int, default=10, help='limit total number of scenarios')
    parser.add_argument('--shuffle_scenarios', type=bool, default=False, help='shuffle scenarios')

    parser.add_argument('--agent_num', type=int, help='number of agents', default=32)
    parser.add_argument('--static_objects_num', type=int, help='number of static objects', default=5)

    parser.add_argument('--lane_len', type=int, help='number of lane point', default=20)
    parser.add_argument('--lane_num', type=int, help='number of lanes', default=70)

    parser.add_argument('--route_len', type=int, help='number of route lane point', default=20)
    parser.add_argument('--route_num', type=int, help='number of route lanes', default=25)
    args = parser.parse_args()

    # create save folder
    os.makedirs(args.save_path, exist_ok=True)

    params = load_params(
            os.path.join(os.path.dirname(__file__), "flow_drive", "config", "config.yaml"))
    params.data_processing.save_path = args.save_path
    processor = DataProcessor(params.data_processing)

    sensor_root = None
    db_files = None
    log_names = None

    # Only preprocess the training data
    with open(os.path.join(os.path.dirname(__file__), 'nuplan_train.json'), "r", encoding="utf-8") as file:
        log_names = json.load(file)

    map_version = "nuplan-maps-v1.0"
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, sensor_root, db_files, map_version)
    scenario_filter = ScenarioFilter(*get_filter_parameters(args.scenarios_per_type, args.total_scenarios, args.shuffle_scenarios, log_names=log_names))

    print("Loading scenarios...")
    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Total number of scenarios: {len(scenarios)}")

    # process data
    del worker, builder, scenario_filter

    # processor.work(scenarios)
    processor.work_multiprocess(scenarios, n_processes=64)

    npz_files = [f for f in os.listdir(args.save_path) if f.endswith('.npz')]

    # Save the list to a JSON file
    json_list_path = os.path.join(args.save_path, 'scenarios_training.json')
    with open(json_list_path, 'w') as json_file:
        json.dump(npz_files, json_file, indent=4)

    print(f"Saved {len(npz_files)} .npz file names")