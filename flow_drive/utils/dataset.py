import os
default_n_threads = 8
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"
import pickle
import time
import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from typing import List
from sklearn.cluster import KMeans
from multiprocessing import Pool, cpu_count, Process
from torch.utils.data import Dataset, DataLoader

from flow_drive.utils.train_utils import openjson, opendata, load_params
from flow_drive.utils.normalizer import StateNormalizer, ObservationNormalizer
from flow_drive.utils.data_augmentation import StatePerturbation

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder


class ClusterStatsRetriever:
    """
    A class to efficiently retrieve cluster statistics for batches of ego future trajectories.
    Pre-loads cluster statistics at initialization for fast inference.
    """

    def __init__(self, cluster_stats_path=None, cluster_stats=None):
        """
        Initialize the cluster stats retriever.

        Args:
            cluster_stats_path: Path to saved cluster statistics file
            cluster_stats: Pre-loaded cluster statistics (alternative to path)
        """
        if cluster_stats is None and cluster_stats_path is None:
            raise ValueError("Either cluster_stats or cluster_stats_path must be provided")

        if cluster_stats is None:
            with open(cluster_stats_path, 'rb') as f:
                saved_data = pickle.load(f)
            self.cluster_centers = saved_data['cluster_centers']
            self.cluster_stats = saved_data['cluster_stats']
            self.n_clusters = saved_data['n_clusters']
        else:
            # Extract cluster centers from provided stats
            self.cluster_centers = np.array([stats['center'].flatten() for stats in cluster_stats.values()])
            self.cluster_stats = cluster_stats
            self.n_clusters = len(cluster_stats)


    def get_cluster_stats(self, cluster_indices: torch.Tensor):
        """
        Retrieve statistics for given cluster indices.

        Args:
            cluster_indices: List or tensor of cluster indices

        Returns:
            cluster_mean_std: Tensor of shape [B, 2, 3] with [mean, std] for each cluster (x, y, yaw)
        """
        device = cluster_indices.device
        cluster_indices = cluster_indices.cpu().numpy()

        B = len(cluster_indices)
        cluster_mean_std = np.zeros((B, 2, 3))
        for i, cluster_id in enumerate(cluster_indices):
            if cluster_id < 0 or cluster_id >= self.n_clusters:
                raise ValueError(f"Cluster index {cluster_id} out of bounds for n_clusters={self.n_clusters}")
            cluster_mean_std[i, 0, :] = self.cluster_stats[cluster_id]['mean']
            cluster_mean_std[i, 1, :] = self.cluster_stats[cluster_id]['std']
        return torch.from_numpy(cluster_mean_std).float().to(device)


    def __call__(self, ego_future_batch):
        """
        Retrieve cluster statistics for a batch of normalized ego future trajectories.

        Args:
            ego_future_batch: Tensor of shape [B, T, 4] - normalized ego futures

        Returns:
            cluster_indices: Tensor of shape [B] - cluster assignment for each trajectory
            cluster_mean_std: Tensor of shape [B, 2, 3] - [mean, std] for each trajectory's cluster (x, y, yaw)
        """
        batch_size = ego_future_batch.shape[0]
        device = ego_future_batch.device

        # Extract x, y coordinates and flatten for comparison with cluster centers (only use x,y for classification)
        xy_trajectories = ego_future_batch[:, :, :2].cpu().numpy()  # [B, T, 2]
        xy_flattened = xy_trajectories.reshape(batch_size, -1)  # [B, T*2]

        # Compute distances to all cluster centers (only using x,y coordinates)
        distances = np.linalg.norm(
            xy_flattened[:, np.newaxis, :] - self.cluster_centers[np.newaxis, :, :],
            axis=2
        )  # [B, n_clusters]

        # Find closest cluster for each trajectory
        cluster_indices = np.argmin(distances, axis=1)  # [B]

        # Extract corresponding mean and std for each trajectory (including yaw statistics)
        cluster_mean_std = np.zeros((batch_size, 2, 3))  # [B, 2, 3] for [mean, std] with [x, y, yaw]

        for i, cluster_id in enumerate(cluster_indices):
            cluster_mean_std[i, 0, :] = self.cluster_stats[cluster_id]['mean']  # mean [3] - x, y, yaw
            cluster_mean_std[i, 1, :] = self.cluster_stats[cluster_id]['std']   # std [3] - x, y, yaw

        # Convert to tensors
        cluster_indices = torch.from_numpy(cluster_indices).to(device)
        cluster_mean_std = torch.from_numpy(cluster_mean_std).float().to(device)

        return cluster_indices, cluster_mean_std


class FlowDriveDataset(Dataset):
    def __init__(self, config, device="cpu"):
        self.weights_file_extension = "training"
        self.data_dir = config.data_processed_path
        self.data_list = openjson(config.data_processed_list)
        self.config = config
        self._past_neighbor_num = config.agent_num
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_time_horizon * 10  # 10Hz
        self._dt = 0.1  # 10Hz

        self.action_normalizer = StateNormalizer.from_json(config)
        self.observation_normalizer = ObservationNormalizer.from_json(config)
        self.state_augmentation = StatePerturbation(augment_prob=config.augment_prob,
                                                    augment_mode=config.augment_mode,
                                                    device=device)

        if config.weighted_sampling:
            if config.balance_mode == "cluster":
                print("Using cluster-balanced weight computation...")
                self.weights = self._compute_weights_for_balanced_clusters()  # shape: [N,]
            elif config.balance_mode == "scenario_type":
                print("Using scenario type-based weight computation...")
                self._get_all_training_scenario_types()
                self.weights = self._compute_weights_for_balanced_scenario_types()
            else:
                raise ValueError(f"Unknown balance mode: {config.balance_mode}")

    def _compute_weights_for_balanced_clusters(self):
        """
        Compute weights to balance cluster distribution.
        Gives higher weights to trajectories from underrepresented clusters.
        Saves and loads precomputed weights to avoid recomputation.
        """
        print("Computing cluster-balanced weights for each trajectory...")

        # Define file path for saving/loading cluster-balanced weights
        cluster_weights_file = self.data_dir + f"/cluster_weights_{self.weights_file_extension}.npy"

        # Check if precomputed cluster weights exist
        if os.path.exists(cluster_weights_file):
            print("Cluster-balanced weights: pre-computed weights found, loading...")
            trajectory_weights = np.load(cluster_weights_file, allow_pickle=True)
            print(f"Loaded {len(trajectory_weights)} precomputed cluster-balanced weights")
            return trajectory_weights

        # If precomputed weights don't exist, compute them
        print("Cluster-balanced weights: computing from scratch...")

        # Load cluster statistics
        cluster_stats_path = os.path.join(os.path.dirname(__file__), "..", "config", "ego_future_clusters.pkl")
        if not os.path.exists(cluster_stats_path):
            raise FileNotFoundError(f"Cluster statistics file not found: {cluster_stats_path}")

        cluster_retriever = ClusterStatsRetriever(cluster_stats_path=cluster_stats_path)

        # Process all trajectories in batches to get cluster assignments
        batch_size = 500
        all_cluster_indices = []

        from torch.utils.data import DataLoader
        dataloader = DataLoader(self, batch_size=batch_size, shuffle=False, num_workers=16)

        print(f"Processing {len(self)} trajectories in batches of {batch_size}...")
        for batch_idx, batch in enumerate(dataloader):
            print(f"Processing batch {batch_idx + 1}/{len(dataloader)}", end='\r')

            # Transform inputs to get normalized ego future
            _, ego_future_normalized, _ = self.transform_inputs_tensor(batch)

            # Get cluster assignments
            cluster_indices, _ = cluster_retriever(ego_future_normalized)
            all_cluster_indices.extend(cluster_indices.cpu().numpy())

        all_cluster_indices = np.array(all_cluster_indices)
        print(f"\nCollected cluster assignments for {len(all_cluster_indices)} trajectories")

        # Compute cluster distribution
        n_clusters = cluster_retriever.n_clusters
        cluster_counts = np.bincount(all_cluster_indices, minlength=n_clusters)
        cluster_frequencies = cluster_counts / len(all_cluster_indices)

        print("Current cluster distribution:")
        for i in range(n_clusters):
            print(f"  Cluster {i}: {cluster_counts[i]} samples ({cluster_frequencies[i]*100:.2f}%)")

        # Compute inverse frequency weights for balancing
        # Add small epsilon to avoid division by zero
        epsilon = 1e-4
        inv_freq_weights = 1.0 / (cluster_frequencies + epsilon)

        # Normalize weights so that the average weight is 1.0
        inv_freq_weights = inv_freq_weights / np.mean(inv_freq_weights)

        print("Inverse frequency weights for clusters:")
        for i in range(n_clusters):
            print(f"  Cluster {i}: {inv_freq_weights[i]:.3f}")

        # Assign weights to each trajectory based on its cluster
        trajectory_weights = inv_freq_weights[all_cluster_indices]

        # Save the computed weights for future use
        print("Saving cluster-balanced weights...")
        np.save(cluster_weights_file, trajectory_weights, allow_pickle=True)
        print(f"Saved cluster weights to: {cluster_weights_file}")

        return trajectory_weights


    def _compute_weights_for_balanced_scenario_types(self):
        """
        Compute weights to balance scenario type distribution.
        Gives higher weights to trajectories from underrepresented scenario types.
        """
        print("Computing scenario type-balanced weights for each trajectory...")

        # Convert to numpy array if it's a list
        scenario_types_array = np.array(self.scenario_types)

        # Get unique scenario types and their counts
        unique_types, counts = np.unique(scenario_types_array, return_counts=True)
        total_samples = len(scenario_types_array)

        # Calculate frequencies
        frequencies = counts / total_samples

        print("Current scenario type distribution:")
        for i, (scenario_type, count, freq) in enumerate(zip(unique_types, counts, frequencies)):
            print(f"  {scenario_type}: {count} samples ({freq*100:.2f}%)")

        # Compute inverse frequency weights for balancing
        # Add small epsilon to avoid division by zero
        epsilon = 1e-4
        inv_freq_weights = 1.0 / (frequencies + epsilon)

        # Normalize weights so that the average weight is 1.0
        inv_freq_weights = inv_freq_weights / np.mean(inv_freq_weights)

        print("Inverse frequency weights for scenario types:")
        for scenario_type, weight in zip(unique_types, inv_freq_weights):
            print(f"  {scenario_type}: {weight:.3f}")

        # Create a mapping from scenario type to weight
        type_to_weight = dict(zip(unique_types, inv_freq_weights))

        # Assign weights to each trajectory based on its scenario type
        trajectory_weights = np.array([type_to_weight[scenario_type] for scenario_type in scenario_types_array])

        print(f"Computed weights for {len(trajectory_weights)} trajectories")
        return trajectory_weights


    def _get_all_training_scenarios(self):
        builder = NuPlanScenarioBuilder(
            self.config.data_original_path, self.config.map_original_path, None, None, "nuplan-maps-v1.0")
        worker = SingleMachineParallelExecutor(use_process_pool=True, max_workers=int(0.8 * cpu_count()))
        t1 = time.time()
        tokens = []
        for idx in range(len(self.data_list)):
            token = self.data_list[idx].split('_')[-1].split('.')[0]
            tokens.append(token)
        all_scenarios = []
        max_tokens = 100000  # maximum tokens per query chunk to avoid too many SQL variables
        for i in tqdm(range(0, len(tokens), max_tokens), desc="Querying scenarios"):
            chunk_tokens = tokens[i:i+max_tokens]
            scenario_filter = ScenarioFilter(None, chunk_tokens, None, None, None, None, None, None,
                                             False, False, False, None, None, None, None, None, None)
            chunk_scenarios = builder.get_scenarios(scenario_filter, worker)
            if len(chunk_scenarios) != len(chunk_tokens):
                raise ValueError(f"Number of scenarios does not match number of tokens in chunk starting at index {i}")
            all_scenarios.extend(chunk_scenarios)
        # Reorder scenarios to match tokens order
        token_to_scenario = {scenario.token: scenario for scenario in all_scenarios}
        scenarios = [token_to_scenario[token] for token in tokens]
        print(f"Finish getting all scenarios in {time.time() - t1:.2f} seconds.")

        token_to_scenario = {scenario.token: scenario for scenario in scenarios}
        scenarios = [token_to_scenario[token] for token in tokens]
        return scenarios


    def _get_all_training_scenario_types(self):
        scenario_types_file = os.path.join(self.data_dir, f"scenario_types_{self.weights_file_extension}.npy")
        if os.path.exists(scenario_types_file):
            self.scenario_types = np.load(scenario_types_file, allow_pickle=True)
            return
        self.scenario_types = []
        scenarios = self._get_all_training_scenarios()
        self.scenario_types = [scenario.scenario_type for scenario in scenarios]
        np.save(scenario_types_file, self.scenario_types, allow_pickle=True)


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        data = opendata(os.path.join(self.data_dir, self.data_list[idx]))
        token = self.data_list[idx].split('_')[-1].split('.')[0]

        lanes_is_route = data['lanes_is_route']

        ego_current_state = data['ego_current_state']
        ego_agent_future = data['ego_agent_future'][:self._future_len]

        neighbor_agents_past = data['neighbor_agents_past'][:self._past_neighbor_num]
        neighbor_agents_future = data['neighbor_agents_future'][:self._predicted_neighbor_num, :self._future_len]

        lanes = data['lanes']
        lanes_speed_limit = data['lanes_speed_limit']
        lanes_has_speed_limit = data['lanes_has_speed_limit']

        route_lanes = data['route_lanes']
        route_lanes_speed_limit = data['route_lanes_speed_limit']
        route_lanes_has_speed_limit = data['route_lanes_has_speed_limit']

        static_objects = data['static_objects']

        data = {
            "idx": idx,
            "token": token,
            "ego_current_state": ego_current_state,
            "ego_future_gt": ego_agent_future,
            "neighbor_agents_past": neighbor_agents_past,
            "neighbors_future_gt": neighbor_agents_future,
            "lanes": lanes,
            "lanes_speed_limit": lanes_speed_limit,
            "lanes_has_speed_limit": lanes_has_speed_limit,
            "lanes_is_route": lanes_is_route,
            "route_lanes": route_lanes,
            "route_lanes_speed_limit": route_lanes_speed_limit,
            "route_lanes_has_speed_limit": route_lanes_has_speed_limit,
            "static_objects": static_objects
        }
        return data


    def inputs_augmentation(self, inputs):
        inputs, ego_future, neighbors_future_not_normalized = self.state_augmentation(inputs)
        # heading to cos sin
        ego_future = torch.cat([
            ego_future[..., :2],
            torch.stack([ego_future[..., 2].cos(),
                            ego_future[..., 2].sin()], dim=-1),],dim=-1,)  # [B, T, 3] -> [B, T, 2 + 2]
        return inputs, ego_future, neighbors_future_not_normalized


    def inputs_normalization(self, inputs, ego_future):
        inputs = self.observation_normalizer(inputs)
        ego_future = self.action_normalizer(ego_future)
        return inputs, ego_future


    def transform_inputs_tensor(self, inputs):
        inputs, ego_future, neighbors_future_not_normalized = self.inputs_augmentation(inputs)
        inputs, ego_future_normalized = self.inputs_normalization(inputs, ego_future)
        return inputs, ego_future_normalized, neighbors_future_not_normalized


def compute_ego_future_clusters(dataset, n_clusters=20, save_path=None):
    """
    Cluster ego future trajectories and visualize cluster centers with statistics.

    Args:
        dataset: FlowDriveDataset instance
        n_clusters: Number of clusters to create
        save_path: Path to save cluster statistics and plot

    Returns:
        cluster_stats: Dictionary containing cluster centers, labels, and statistics
    """
    print(f"Computing ego future clusters for {len(dataset)} samples...")

    # Set OpenBLAS thread limit to avoid memory issues
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    # Collect all normalized ego futures (full [x, y, cos, sin] data)
    all_ego_futures_full = []  # Store full trajectories [T, 4]
    all_ego_futures_xy = []    # Store only x,y for clustering [T*2]
    all_indices = []

    # Create a DataLoader to efficiently process the dataset
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=16)

    for batch_idx, batch in enumerate(dataloader):
        print(f"Processing batch {batch_idx + 1}/{len(dataloader)}", end='\r')

        # Transform inputs to get normalized ego future
        _, ego_future_normalized, _ = dataset.transform_inputs_tensor(batch)

        # ego_future_normalized shape: [B, T, 4] where T=40, 4=[x, y, cos, sin]
        batch_size = ego_future_normalized.shape[0]

        for i in range(batch_size):
            ego_future = ego_future_normalized[i].cpu().numpy()  # [T, 4]
            # Save full trajectory data
            all_ego_futures_full.append(ego_future)  # [T, 4]
            # Extract only x, y coordinates for clustering
            xy_trajectory = ego_future[:, :2].flatten()  # [T*2] = [80]
            all_ego_futures_xy.append(xy_trajectory)
            all_indices.append(batch_idx * 100 + i)

        if batch_idx > 500:
            break

    print(f"\nCollected {len(all_ego_futures_xy)} trajectories")

    # Convert to numpy arrays
    ego_futures_full_array = np.array(all_ego_futures_full)  # [N, T, 4]
    ego_futures_xy_array = np.array(all_ego_futures_xy)      # [N, T*2]

    # Perform K-means clustering (using only x,y coordinates)
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(ego_futures_xy_array)
    cluster_centers = kmeans.cluster_centers_  # [n_clusters, 80]

    # Compute cluster statistics (mean and std of positions and yaw angles)
    cluster_stats = {}
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_trajectories_xy = ego_futures_xy_array[cluster_mask]  # [n_samples_in_cluster, 80]
        cluster_trajectories_full = ego_futures_full_array[cluster_mask]  # [n_samples_in_cluster, T, 4]

        if len(cluster_trajectories_full) > 0:
            # Reshape xy trajectories back to [n_samples, T, 2] for computing position statistics
            cluster_trajectories_reshaped = cluster_trajectories_xy.reshape(-1, 40, 2)
            # Compute mean and std across all positions in the cluster
            all_positions = cluster_trajectories_reshaped.reshape(-1, 2)  # [n_samples*T, 2]
            cluster_pos_mean = np.mean(all_positions, axis=0)  # [2]
            cluster_pos_std = np.std(all_positions, axis=0)   # [2]

            # Compute yaw angle statistics from cos/sin values directly from saved data
            cos_vals = cluster_trajectories_full[:, :, 2].flatten()  # [n_samples*T]
            sin_vals = cluster_trajectories_full[:, :, 3].flatten()  # [n_samples*T]
            yaw_vals = np.arctan2(sin_vals, cos_vals)  # [n_samples*T]

            # Handle circular statistics for yaw angles
            # Convert to complex numbers for circular mean and std
            complex_yaw = np.exp(1j * yaw_vals)
            circular_mean_complex = np.mean(complex_yaw)
            cluster_yaw_mean = np.angle(circular_mean_complex)  # circular mean

            # Circular standard deviation
            R = np.abs(circular_mean_complex)  # resultant length
            cluster_yaw_std = np.sqrt(-2 * np.log(R)) if R > 0 else np.pi  # circular std

            cluster_stats[cluster_id] = {
                'mean': np.concatenate([cluster_pos_mean, [cluster_yaw_mean]]),  # [3] - x, y, yaw
                'std': np.concatenate([cluster_pos_std, [cluster_yaw_std]]),     # [3] - x, y, yaw
                'center': cluster_centers[cluster_id].reshape(40, 2),  # [T, 2] - only x,y for clustering
                'count': np.sum(cluster_mask)
            }
        else:
            # Fallback if no trajectories found
            cluster_stats[cluster_id] = {
                'mean': np.array([0.0, 0.0, 0.0]),  # [3] - x, y, yaw
                'std': np.array([1.0, 1.0, np.pi]), # [3] - x, y, yaw
                'center': cluster_centers[cluster_id].reshape(40, 2),
                'count': np.sum(cluster_mask)
            }

    # Save cluster statistics
    save_data = {
        'cluster_centers': cluster_centers,
        'cluster_labels': cluster_labels,
        'cluster_stats': cluster_stats,
        'n_clusters': n_clusters,
        'indices': all_indices
    }
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Cluster statistics saved to: {save_path}")

    return cluster_stats


if __name__ == "__main__":
    # perform_clustering()
    params_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    params = load_params(params_path)
    compute_ego_future_clusters(
        FlowDriveDataset(params.data_processing, device="cpu", training=True),
        n_clusters=20,
        save_path=os.path.join(os.path.dirname(__file__), "..", "config", "ego_future_clusters.pkl")
    )
