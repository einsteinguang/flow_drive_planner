
import os
import warnings
import torch
import time
import numpy as np
import numpy.typing as npt
from shapely.geometry import Point
from typing import Deque, Dict, List, Type, Optional, Tuple
import threading
import pickle
import uuid
import glob
import atexit

warnings.filterwarnings("ignore")

from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.observation_type import Observation, DetectionsTracks
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput
)
from nuplan.common.maps.abstract_map_objects import (
    LaneGraphEdgeMapObject,
    RoadBlockGraphEdgeMapObject,
)

from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_states_to_state_array,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    normalize_angle,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
    PDMOccupancyMap,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation import (
    PDMObservation,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.route_utils import (
    route_roadblock_correction,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.graph_search.dijkstra import (
    Dijkstra,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_emergency_brake import (
    PDMEmergencyBrake,
)

from flow_drive.model.flow_drive_planner import FlowDrivePlanner
from flow_drive.data_process.data_processor import DataProcessor
from flow_drive.utils.train_utils import load_params, set_seed
from flow_drive.utils.plot_dataset_scenario import plot_scenario


def outputs_to_trajectory(
    outputs: torch.Tensor,
    ego_state_history: Deque[EgoState],
    future_horizon: float,
    step_interval: float
    ) -> List[InterpolatableState]:
    predictions = outputs[0].detach().cpu().numpy().astype(np.float64) # [T, 3]
    # transform relative poses to absolute poses
    states = transform_predictions_to_states(predictions, ego_state_history, future_horizon, step_interval)
    return states


class FlowDrivePlannerWrapper(AbstractPlanner):
    def __init__(
            self,
            device: str = "cpu",
            mlflow_exp_name: str = "None",
            ckpt_path: str = "None",
            load_run_name: str = None,
            load_epoch: int = 0,
            post_mode: int = 0,
            render: bool = False,
            video_dir: str = None,
        ):
        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"

        self._master_seed = 520  # master seed for reproducibility
        self._device = device
        self._post_process = post_mode
        self._ckpt_path = ckpt_path
        # print(f"Post-process: {self._post_process}")

        self._lateral_offset = [0, 6.5/20, -6.5/20, 12.5/20, -12.5/20]
        self._speed_offsets = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        self._planner = None
        self._planner_id = str(uuid.uuid4())[:8]
        self._mlflow_exp_name = mlflow_exp_name
        self._load_run_name = load_run_name
        self._load_epoch = load_epoch
        self._future_horizon = None
        self._step_interval = 0.1
        self._data_processor = None

        self._map_api: Optional[AbstractMap] = None
        self._route_roadblock_ids = None
        self._trajectory_scorer = TrajectoryScorer()

        self._params = load_params(
            os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"))

        self._render = render
        self._video_dir = video_dir

    def name(self) -> str:
        """
        Inherited.
        """
        return "flow_drive"

    def observation_type(self) -> Type[Observation]:
        """
        Inherited.
        """
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Inherited.
        """
        self._iteration = 0
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids
        self._trajectory_scorer.initialize(self._map_api, self._route_roadblock_ids)

        self._planner = FlowDrivePlanner(
            self._params,
            self._device,
            self._ckpt_path,
            self._mlflow_exp_name,
            self._load_run_name,
            self._load_epoch)
        self._data_processor = DataProcessor(self._planner.params.data_processing)

        self._future_horizon = self._planner.params.data_processing.future_time_horizon
        self._initialization = initialization

    def planner_input_to_model_inputs(self, planner_input: PlannerInput) -> Dict[str, torch.Tensor]:
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)
        model_inputs = self._data_processor.observation_adapter(
            history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)
        return model_inputs

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Inherited.
        """
        set_seed(self._master_seed + self._iteration)  # Set seed for reproducibility

        inputs = self.planner_input_to_model_inputs(current_input)

        if self._post_process == 1:
            current_lane = self._trajectory_scorer.prepare_scoring(current_input)
            speed_limit = current_lane.speed_limit_mps
            if speed_limit is None:
                speed_limit = 15.
            speed_limit = torch.tensor([speed_limit], device=self._device)  # [B] normalized speed limit
            outputs = self._planner.plan_multiple_trajectories_with_moderated_offset(
                inputs, speed_limit, self._speed_offsets, self._lateral_offset)  # (S, B, T, [x, y, heading]), B = 1
            scores, ego_states_list = self._trajectory_scorer.score_plans(outputs)
            index = np.argmax(scores)
            ego_states = ego_states_list[index]
            trajectory = InterpolatedTrajectory(trajectory=ego_states)
        else:
            outputs = self._planner(inputs)
            trajectory = InterpolatedTrajectory(
                trajectory=outputs_to_trajectory(outputs, current_input.history.ego_states,
                                                 self._future_horizon, self._step_interval)
            )

        if self._render:
            inputs_for_plotting = {k: v[0].cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if self._post_process > 0:
                inputs_for_plotting["ego_plan"] = outputs[index, 0].cpu().numpy()
            else:
                inputs_for_plotting["ego_plan"] = outputs[0].cpu().numpy()
            fig_dir = os.path.join(self._video_dir, self._planner_id)
            os.makedirs(fig_dir, exist_ok=True)
            fig_path = os.path.join(fig_dir, f"{self._iteration}.png")
            plot_scenario(inputs_for_plotting, fig_path)

        self._iteration += 1
        return trajectory

    # make video when deleting the planner
    def __del__(self):
        if self._render:
            img_paths = sorted(glob.glob(os.path.join(self._video_dir, self._planner_id, "*.png")),
                               key=lambda x: int(os.path.basename(x).split(".")[0]))
            if len(img_paths) > 0:
                import imageio
                video_path = os.path.join(self._video_dir, f"{self._planner_id}.mp4")
                with imageio.get_writer(video_path, fps=10) as video_writer:
                    for img_path in img_paths:
                        image = imageio.v2.imread(img_path)
                        video_writer.append_data(image)
                print(f"Saved video to {video_path}")


class TrajectoryScorer:
    def __init__(self):
        self._iteration: int = 0
        self._map_radius: int = 50  # [m]
        self._map_api: Optional[AbstractMap] = None
        self._route_roadblock_ids = None
        self._route_roadblock_dict: Optional[
            Dict[str, RoadBlockGraphEdgeMapObject]
        ] = None
        self._route_lane_dict: Optional[Dict[str, LaneGraphEdgeMapObject]] = None
        self._centerline: Optional[PDMPath] = None
        self._drivable_area_map: Optional[PDMOccupancyMap] = None
        trajectory_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        self._observation = PDMObservation(
            trajectory_sampling, proposal_sampling, self._map_radius
        )
        self._simulator = PDMSimulator(proposal_sampling)
        self._scorer = PDMScorer(proposal_sampling)
        self._emergency_brake = PDMEmergencyBrake(trajectory_sampling)
        self._current_input: Optional[PlannerInput] = None


    def _load_route_dicts(self, route_roadblock_ids: List[str]) -> None:
        """
        Loads roadblock and lane dictionaries of the target route from the map-api.
        :param route_roadblock_ids: ID's of on-route roadblocks
        """
        # remove repeated ids while remaining order in list
        route_roadblock_ids = list(dict.fromkeys(route_roadblock_ids))
        self._route_roadblock_dict = {}
        self._route_lane_dict = {}
        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(
                id_, SemanticMapLayer.ROADBLOCK_CONNECTOR
            )
            self._route_roadblock_dict[block.id] = block
            for lane in block.interior_edges:
                self._route_lane_dict[lane.id] = lane


    def _route_roadblock_correction(self, ego_state: EgoState) -> None:
        """
        Corrects the roadblock route and reloads lane-graph dictionaries.
        :param ego_state: state of the ego vehicle.
        """
        route_roadblock_ids = route_roadblock_correction(
            ego_state, self._map_api, self._route_roadblock_dict
        )
        self._load_route_dicts(route_roadblock_ids)


    def _get_discrete_centerline(
        self, current_lane: LaneGraphEdgeMapObject, search_depth: int = 30
        ) -> List[StateSE2]:
        """
        Applies a Dijkstra search on the lane-graph to retrieve discrete centerline.
        :param current_lane: lane object of starting lane.
        :param search_depth: depth of search (for runtime), defaults to 30
        :return: list of discrete states on centerline (x,y,θ)
        """

        roadblocks = list(self._route_roadblock_dict.values())
        roadblock_ids = list(self._route_roadblock_dict.keys())

        # find current roadblock index
        start_idx = np.argmax(
            np.array(roadblock_ids) == current_lane.get_roadblock_id()
        )
        roadblock_window = roadblocks[start_idx : start_idx + search_depth]

        graph_search = Dijkstra(current_lane, list(self._route_lane_dict.keys()))
        route_plan, path_found = graph_search.search(roadblock_window[-1])

        centerline_discrete_path: List[StateSE2] = []
        for lane in route_plan:
            centerline_discrete_path.extend(lane.baseline_path.discrete_path)

        return centerline_discrete_path


    def _get_starting_lane(self, ego_state: EgoState) -> LaneGraphEdgeMapObject:
        """
        Returns the most suitable starting lane, in ego's vicinity.
        :param ego_state: state of ego-vehicle
        :return: lane object (on-route)
        """
        starting_lane: LaneGraphEdgeMapObject = None
        on_route_lanes, heading_error = self._get_intersecting_lanes(ego_state)

        if on_route_lanes:
            # 1. Option: find lanes from lane occupancy-map
            # select lane with lowest heading error
            starting_lane = on_route_lanes[np.argmin(np.abs(heading_error))]
            return starting_lane

        else:
            # 2. Option: find any intersecting or close lane on-route
            closest_distance = np.inf
            for edge in self._route_lane_dict.values():
                if edge.contains_point(ego_state.center):
                    starting_lane = edge
                    break

                distance = edge.polygon.distance(ego_state.car_footprint.geometry)
                if distance < closest_distance:
                    starting_lane = edge
                    closest_distance = distance

        return starting_lane


    def _get_intersecting_lanes(
        self, ego_state: EgoState
    ) -> Tuple[List[LaneGraphEdgeMapObject], List[float]]:
        """
        Returns on-route lanes and heading errors where ego-vehicle intersects.
        :param ego_state: state of ego-vehicle
        :return: tuple of lists with lane objects and heading errors [rad].
        """
        assert (
            self._drivable_area_map
        ), "AbstractPDMPlanner: Drivable area map must be initialized first!"

        ego_position_array: npt.NDArray[np.float64] = ego_state.rear_axle.array
        ego_rear_axle_point: Point = Point(*ego_position_array)
        ego_heading: float = ego_state.rear_axle.heading

        intersecting_lanes = self._drivable_area_map.intersects(ego_rear_axle_point)

        on_route_lanes, on_route_heading_errors = [], []
        for lane_id in intersecting_lanes:
            if lane_id in self._route_lane_dict.keys():
                # collect baseline path as array
                lane_object = self._route_lane_dict[lane_id]
                lane_discrete_path: List[
                    StateSE2
                ] = lane_object.baseline_path.discrete_path
                lane_state_se2_array = np.array(
                    [state.array for state in lane_discrete_path], dtype=np.float64
                )
                # calculate nearest state on baseline
                lane_distances = (
                    ego_position_array[None, ...] - lane_state_se2_array
                ) ** 2
                lane_distances = lane_distances.sum(axis=-1) ** 0.5

                # calculate heading error
                heading_error = (
                    lane_discrete_path[np.argmin(lane_distances)].heading - ego_heading
                )
                heading_error = np.abs(normalize_angle(heading_error))

                # add lane to candidates
                on_route_lanes.append(lane_object)
                on_route_heading_errors.append(heading_error)

        return on_route_lanes, on_route_heading_errors


    def initialize(self, map_api: AbstractMap, route_roadblock_ids: List[str]) -> None:
        """
        Initializes the trajectory scorer with map API and route roadblock IDs.
        :param map_api: map API to access map objects
        :param route_roadblock_ids: list of roadblock IDs on the route
        """
        self._map_api = map_api
        self._route_roadblock_ids = route_roadblock_ids
        self._load_route_dicts(route_roadblock_ids)


    def prepare_scoring(self, current_input: PlannerInput) -> LaneGraphEdgeMapObject:
        ego_state, observation = current_input.history.current_state

        if self._iteration == 0:
            self._route_roadblock_correction(ego_state)

        # Update/Create drivable area polygon map
        self._drivable_area_map = get_drivable_area_map(
            self._map_api, ego_state, self._map_radius
        )

        # 1. Environment forecast and observation update
        self._observation.update(
            ego_state,
            observation,
            current_input.traffic_light_data,
            self._route_lane_dict,
        )

        current_lane = self._get_starting_lane(ego_state)
        self._centerline = PDMPath(self._get_discrete_centerline(current_lane))

        self._current_input = current_input

        return current_lane


    def score_plans(self, planner_outputs: torch.Tensor) \
        -> Tuple[np.ndarray, List[List[InterpolatableState]]]:
        """
        planner_outputs: (S, B, T, A) where S = number of plans, B = batch size (1)
        """
        plans_list = []
        ego_states_list = []
        for i in range(planner_outputs.shape[0]):
            ego_states = outputs_to_trajectory(
                planner_outputs[i], self._current_input.history.ego_states, 4., 0.1)
            plan_state_array = ego_states_to_state_array(ego_states)  # (40, 11)
            plans_list.append(plan_state_array)
            ego_states_list.append(ego_states)
        plans_array = np.array(plans_list)

        ego_state, _ = self._current_input.history.current_state

        # Simulate proposals
        simulated_proposals_array = self._simulator.simulate_proposals(
            plans_array, ego_state
        )

        # Score proposals
        proposal_scores = self._scorer.score_proposals(
            simulated_proposals_array,
            ego_state,
            self._observation,
            self._centerline,
            self._route_lane_dict,
            self._drivable_area_map,
            self._map_api,
        )  # [S * B] where S = number of plans, B = batch size (1)
        # Apply brake if emergency is expected
        trajectory = self._emergency_brake.brake_if_emergency(
            ego_state, proposal_scores, self._scorer
        )
        if trajectory is not None:
            ego_states_list = [trajectory.get_sampled_trajectory()] * len(ego_states_list)

        self._iteration += 1
        return proposal_scores, ego_states_list

