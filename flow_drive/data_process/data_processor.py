import numpy as np
from tqdm import tqdm

from nuplan.common.actor_state.state_representation import Point2D

from flow_drive.data_process.roadblock_utils import route_roadblock_correction
from flow_drive.data_process.agent_process import (
agent_past_process,
sampled_tracked_objects_to_array_list,
sampled_static_objects_to_array_list,
agent_future_process
)
from flow_drive.data_process.map_process import get_neighbor_vector_set_map, map_process
from flow_drive.data_process.ego_process import \
    get_ego_past_array_from_scenario, get_ego_future_array_from_scenario, \
        get_ego_past_array_from_history_buffer, calculate_additional_ego_states
from flow_drive.data_process.utils import convert_to_model_inputs
from flow_drive.data_process.utils import convert_absolute_quantities_to_relative

import multiprocessing
from multiprocessing import Process, Queue
import queue
import time


class DataProcessor(object):
    def __init__(self, config):

        self._save_dir = getattr(config, "save_path", None)

        self.past_time_horizon = config.past_time_horizon # [seconds]
        self.num_past_poses = 10 * self.past_time_horizon
        self.future_time_horizon = config.future_time_horizon # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon

        self.num_agents = config.agent_num
        self.num_static = config.static_objects_num
        self.max_ped_bike = 10 # Limit the number of pedestrians and bicycles in the agent.
        self._radius = 100 # [m] query radius scope relative to the current pose.

        self._map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES'] # name of map features to be extracted.
        self._max_elements = {'LANE': config.lane_num, 'LEFT_BOUNDARY': config.lane_num, 'RIGHT_BOUNDARY': config.lane_num, 'ROUTE_LANES': config.route_num} # maximum number of elements to extract per feature layer.
        self._max_points = {'LANE': config.lane_len, 'LEFT_BOUNDARY': config.lane_len, 'RIGHT_BOUNDARY': config.lane_len, 'ROUTE_LANES': config.route_len} # maximum number of points per feature to extract per feature layer.

    # Use for inference
    def observation_adapter(self, history_buffer, traffic_light_data, map_api, route_roadblock_ids, device='cpu'):

        '''
        ego
        '''
        ego_state = history_buffer.current_state[0]
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        anchor_ego_state = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading], dtype=np.float64)
        ego_agent_past, time_stamps_past = get_ego_past_array_from_history_buffer(history_buffer, self.num_past_poses)

        '''
        neighbor
        '''
        observation_buffer = history_buffer.observation_buffer # Past observations including the current
        neighbor_agents_past, neighbor_agents_types = sampled_tracked_objects_to_array_list(observation_buffer)
        static_objects, static_objects_types = sampled_static_objects_to_array_list(observation_buffer[-1])
        ego_agent_past, neighbor_agents_past, _, static_objects = \
            agent_past_process(ego_agent_past, neighbor_agents_past, neighbor_agents_types, self.num_agents, static_objects, static_objects_types, self.num_static, self.max_ped_bike, anchor_ego_state)

        '''
        Map
        '''
        # Simply fixing disconnected routes without pre-searching for reference lines
        route_roadblock_ids = route_roadblock_correction(
            ego_state, map_api, route_roadblock_ids
        )
        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            map_api, self._map_features, ego_coords, self._radius, traffic_light_data
        )
        vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords, traffic_light_data, speed_limit, lane_route, self._map_features,
                                    self._max_elements, self._max_points)

        ego_current_state = calculate_additional_ego_states(ego_agent_past, time_stamps_past)

        # print("ego current state:", ego_current_state)

        data = {"neighbor_agents_past": neighbor_agents_past[:, -21:],
                "ego_current_state": ego_current_state, # ego centric x, y, cos, sin, vx, vy, ax, ay, steering angle, yaw rate, we only use x, y, cos, sin during inference
                "static_objects": static_objects}
        data.update(vector_map)
        data = convert_to_model_inputs(data, device)

        return data


    def process_scenario(self, scenario):
        map_name = scenario._map_name
        token = scenario.token
        map_api = scenario.map_api
        scenario_type = scenario.scenario_type

        '''
        ego & agents past
        '''
        ego_state = scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        anchor_ego_state = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading], dtype=np.float64)
        ego_agent_past, time_stamps_past = get_ego_past_array_from_scenario(scenario, self.num_past_poses, self.past_time_horizon)  # list[EgoState]

        present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_past_tracked_objects(
                iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
            )
        ]
        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        neighbor_agents_past, neighbor_agents_types = \
            sampled_tracked_objects_to_array_list(sampled_past_observations)

        static_objects, static_objects_types = sampled_static_objects_to_array_list(present_tracked_objects)

        ego_agent_past, neighbor_agents_past, neighbor_indices, static_objects = \
            agent_past_process(ego_agent_past, neighbor_agents_past, neighbor_agents_types, self.num_agents, static_objects, static_objects_types, self.num_static, self.max_ped_bike, anchor_ego_state)

        '''
        Map
        '''
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))

        if route_roadblock_ids != ['']:
            route_roadblock_ids = route_roadblock_correction(
                ego_state, map_api, route_roadblock_ids
            )

        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            map_api, self._map_features, ego_coords, self._radius, traffic_light_data
        )

        vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords, traffic_light_data, speed_limit, lane_route, self._map_features,
                                self._max_elements, self._max_points)

        '''
        ego & agents future
        '''
        ego_agent_future = get_ego_future_array_from_scenario(scenario, ego_state, self.num_future_poses, self.future_time_horizon)

        present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
                iteration=0, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
            )
        ]

        sampled_future_observations = [present_tracked_objects] + future_tracked_objects
        future_tracked_objects_array_list, _ = sampled_tracked_objects_to_array_list(sampled_future_observations)
        neighbor_agents_future = agent_future_process(anchor_ego_state, future_tracked_objects_array_list, self.num_agents, neighbor_indices)


        '''
        ego current
        '''
        ego_current_state = calculate_additional_ego_states(ego_agent_past, time_stamps_past)

        # gather data
        data = {"map_name": map_name, "token": token, "scenario_type": scenario_type,
                "ego_current_state": ego_current_state, "ego_agent_future": ego_agent_future,
                "neighbor_agents_past": neighbor_agents_past, "neighbor_agents_future": neighbor_agents_future, "static_objects": static_objects}
        data.update(vector_map)

        self.save_to_disk(self._save_dir, data)


    def exclude_existing_scenarios(self, scenarios, saved_npz_files):
        """
        Exclude scenarios that have already been processed and saved to disk.
        :param scenarios: List of scenario objects to process.
        :param saved_npz_files: List of already saved npz files.
        :return: Filtered list of scenarios.
        """
        existing_scenarios = set()
        for file in saved_npz_files:
            map_name, token = file.split('_')[:2]
            # remove the .npz extension
            token = token.split('.')[0]
            existing_scenarios.add((map_name, token))

        filtered_scenarios = [scen for scen in scenarios if (scen._map_name, scen.token) not in existing_scenarios]
        return filtered_scenarios


    # Use for data preprocess
    def work(self, scenarios):
        for scenario in tqdm(scenarios):
            self.process_scenario(scenario)


    def save_to_disk(self, dir, data):
        np.savez(f"{dir}/{data['map_name']}_{data['token']}.npz", **data)


    def work_multiprocess(self, scenarios, n_processes):
        """
        Parallelize processing of a list of scenarios using multiprocessing.
        Each worker processes a subset of scenarios and has its own progress bar.
        :param scenarios: List of scenario objects to process.
        :param n_processes: Number of worker processes to launch.
        """
        # on some platforms you must set the start method before creating any queues or processes
        multiprocessing.set_start_method('spawn', force=True)

        scenarios_per_process = np.array_split(scenarios, n_processes)

        processes = []
        for i in range(n_processes):
            p = Process(
                target=self._worker,
                args=(list(scenarios_per_process[i]), i), # Pass scenario chunk and process_id
                name=f"worker_{i}"
            )
            p.daemon = True # Daemonize processes so they exit when the main program exits
            p.start()
            processes.append(p)

        # Wait for all workers (allow Ctrl-C)
        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("Terminating all processes...")
            for p in processes:
                p.terminate() # Forcefully terminate
                p.join() # Wait for termination to complete
        finally:
            # A small delay to ensure tqdm progress bars from child processes have finished printing
            time.sleep(1.)
            print("All processes finished.")


    def _worker(self, scenarios_chunk, process_id: int):
        """
        Worker loop: processes its assigned chunk of scenarios with its own progress bar.
        :param scenarios_chunk: List of scenario objects for this worker to process.
        :param process_id: The ID of this process, used for positioning the progress bar.
        """
        # Ensure tqdm is imported within the worker if not globally available in the spawned process context
        # from tqdm import tqdm
        # import multiprocessing # Already imported at the top level

        process_name = multiprocessing.current_process().name
        for scenario in tqdm(scenarios_chunk, desc=f"{process_name}", position=process_id, leave=True):
            try:
                self.process_scenario(scenario)
            except Exception as e:
                # you may want to log this to a file instead
                # Using print here might interleave output with tqdm, consider logging
                print(f"[{process_name}] Error on {scenario.token if hasattr(scenario, 'token') else 'unknown scenario'}: {e}")