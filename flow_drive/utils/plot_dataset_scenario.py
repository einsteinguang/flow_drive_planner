import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define constants for colors and dimensions
COLOR_LANES = 'gray'
COLOR_ROUTE_LANES = 'blue'
COLOR_EGO_CURRENT = 'red'
COLOR_EGO_FUTURE = 'blue'
COLOR_EGO_PLAN = "green"
COLOR_NEIGHBORS = 'black'

STATIC_OBJECT_COLORS = {
    "CZONE_SIGN": 'orange',
    "BARRIER": 'orange',
    "TRAFFIC_CONE": 'orange',
    "UNKNOWN_STATIC": 'orange'
}

TRAFFIC_LIGHT_COLORS = {
    "GREEN": 'green',
    "YELLOW": 'yellow',
    "RED": 'red'
}

# Default dimensions for ego vehicle
EGO_LENGTH = 5.176  # meters
EGO_WIDTH = 2.3   # meters
EGO_REAR_CENTER_OFFSET = 1.461  # meters, distance from rear axle to center of the vehicle

def _plot_agent_rectangle(ax, x, y, cos_h, sin_h, length, width, color, alpha=1.0, zorder=10, fill=True, show_heading=False):
    """Plot a rotated rectangle representing an agent with an optional hollow front triangle.

    Args:
        ax: Matplotlib axis.
        x, y: Center position (vehicle center) in meters.
        cos_h, sin_h: Cosine and sine of heading.
        length, width: Dimensions of the rectangle.
        color: Base color for rectangle (edge & face if fill=True).
        alpha: Opacity of the rectangle body.
        zorder: Base z-order.
        fill: Whether to fill the rectangle.
        show_heading: If True, draw a small hollow gray triangle in front indicating orientation.
    """
    if length <= 0 or width <= 0:  # Skip if dimensions are invalid
        return

    angle_rad = np.arctan2(sin_h, cos_h)
    angle_deg = np.rad2deg(angle_rad)

    # Base rectangle centered at origin before transform
    rect = patches.Rectangle((-length / 2, -width / 2), length, width,
                             linewidth=1, edgecolor=color, facecolor=color,
                             alpha=alpha, zorder=zorder, fill=fill)

    transform = plt.matplotlib.transforms.Affine2D().rotate_deg(angle_deg).translate(x, y)
    rect.set_transform(transform + ax.transData)
    ax.add_patch(rect)

    if show_heading:
        # Small hollow triangle sitting flush at the front edge (inside/just at boundary), not protruding forward
        tri_len = min(length, width) * 0.5   # depth of triangle into the body
        half_base = width * 0.46              # half of triangle base width
        # Apex at the midpoint of the front edge, base slightly behind inside the rectangle
        apex = (length / 2, 0.0)
        base_left = (length / 2 - tri_len, -half_base)
        base_right = (length / 2 - tri_len, half_base)
        triangle = patches.Polygon([apex, base_right, base_left], closed=True,
                                   facecolor='none', edgecolor='#4a4a4a', linewidth=1,
                                   zorder=zorder + 1, joinstyle='miter')
        triangle.set_transform(transform + ax.transData)
        ax.add_patch(triangle)


def _plot_lanes_data(ax, lanes_data, color, alpha=0.2):
    """Helper function to plot lanes (regular or route)."""
    if lanes_data is None or lanes_data.shape[0] == 0:
        return

    for lane_idx in range(lanes_data.shape[0]):
        lane_points = lanes_data[lane_idx]  # Shape (20, 12)

        # Heuristic: if the first point of the centerline is (0,0), assume it's an invalid/padded lane slot
        if np.all(lane_points[0, :2] == 0): # Check x_center, y_center of the first point
            # print(f"Skipping lane {lane_idx} with all zero values.")
            continue

        center_x = lane_points[:, 0]
        center_y = lane_points[:, 1]
        left_x = lane_points[:, 4] + center_x
        left_y = lane_points[:, 5] + center_y
        right_x = lane_points[:, 6] + center_x
        right_y = lane_points[:, 7] + center_y

        # Plot lane lines
        ax.plot(center_x, center_y, linestyle='--', color=color, linewidth=1, zorder=1, alpha=alpha)
        ax.plot(left_x, left_y, linestyle='-', color=color, linewidth=1.5, zorder=1, alpha=alpha)
        ax.plot(right_x, right_y, linestyle='-', color=color, linewidth=1.5, zorder=1, alpha=alpha)

        # Traffic lights: check features of the last point entry in the lane_points array
        last_point_features = lane_points[-1]
        tl_status = last_point_features[8:12]  # green, yellow, red, unknown

        tl_plot_color = None
        if tl_status[0] == 1:  # Green
            tl_plot_color = TRAFFIC_LIGHT_COLORS["GREEN"]
        elif tl_status[1] == 1:  # Yellow
            tl_plot_color = TRAFFIC_LIGHT_COLORS["YELLOW"]
        elif tl_status[2] == 1:  # Red
            tl_plot_color = TRAFFIC_LIGHT_COLORS["RED"]

        if tl_plot_color:
            # Plot circle at the x,y coordinates of the first point of the centerline
            ax.plot(center_x[0], center_y[0], 'o', markersize=8,
                    color=tl_plot_color, markeredgecolor='black', zorder=20)


def plot_scenario(data, output_filename="scenario.png"):
    """
    Plots a driving scenario from the given data dictionary.

    Args:
        data (dict): A dictionary containing scenario data as NumPy arrays.
        output_filename (str): The filename to save the plot to.
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    # 1. Plot regular lanes
    if "lanes" in data:
        _plot_lanes_data(ax, data["lanes"], COLOR_LANES, alpha=0.7)

    # # 2. Plot route lanes
    if "route_lanes" in data:
        _plot_lanes_data(ax, data["route_lanes"], COLOR_ROUTE_LANES, alpha=0.7)

    # 3. Plot static objects
    # static_objects: (N, 10), [x, y, cos, sin, width, length, type(4)]
    if "static_objects" in data and data["static_objects"] is not None and data["static_objects"].shape[0] > 0:
        static_objects_data = data["static_objects"]
        for i in range(static_objects_data.shape[0]):
            obj = static_objects_data[i]
            # Heuristic for validity: width and length must be positive.
            # Also skip if x,y,w,l are all zero (common padding for unused slots).
            if np.all(obj[0:2] == 0) and np.all(obj[4:6] == 0): # x,y and width,length are all zero
                 continue

            obj_type_enc = obj[6:10]
            obj_color = STATIC_OBJECT_COLORS["UNKNOWN_STATIC"]
            if obj_type_enc[0] == 1: obj_color = STATIC_OBJECT_COLORS["CZONE_SIGN"]
            elif obj_type_enc[1] == 1: obj_color = STATIC_OBJECT_COLORS["BARRIER"]
            elif obj_type_enc[2] == 1: obj_color = STATIC_OBJECT_COLORS["TRAFFIC_CONE"]

            _plot_agent_rectangle(ax, obj[0], obj[1], obj[2], obj[3],  # x, y, cos, sin
                                  obj[5], obj[4],  # length, width
                                  obj_color, zorder=10, show_heading=False)

    # 4. Plot neighbor agents (current state from past history)
    # neighbor_agents_past: (N, 21_steps, 11_features)
    # Features: [x, y, cos, sin, vx, vy, width, length, type(3)]
    if "neighbor_agents_past" in data.keys() and \
       data["neighbor_agents_past"] is not None and \
       data["neighbor_agents_past"].shape[0] > 0 and \
       data["neighbor_agents_past"].shape[1] > 0: # Ensure there's at least one history step

        neighbor_agents_current = data["neighbor_agents_past"][:, -1, :] # Current state is the last step, shape (N, 11)
        for i in range(neighbor_agents_current.shape[0]):
            agent = neighbor_agents_current[i]
            width, length = agent[6], agent[7]
            if np.all(agent[0:2]==0) and np.all(agent[6:8]==0): # x,y and width,length are all zero
                continue

            _plot_agent_rectangle(ax, agent[0], agent[1], agent[2], agent[3],  # x, y, cos, sin
                                  length, width,
                                  COLOR_NEIGHBORS, alpha=0.4, zorder=5, show_heading=True)
        # Plot neighbor history
        # neighbor_agents_history = data["neighbor_agents_past"][:, :-1, :]  # exclude the current state already plotted
        # for agent_history in neighbor_agents_history:
        #     for state in agent_history:
        #         # Skip invalid or padded entries
        #         if np.all(state[0:2] == 0) and np.all(state[6:8] == 0):
        #             continue
        #         # Scale down dimensions for history
        #         scaled_length = state[7] * 0.4
        #         scaled_width = state[6] * 0.4
        #         _plot_agent_rectangle(ax, state[0], state[1], state[2], state[3],
        #                               scaled_length, scaled_width,
        #                               COLOR_NEIGHBORS, alpha=1, zorder=10, fill=False)

    # 5. Plot ego agent (current state)
    # ego_current_state: (10,), [x, y, cos, sin, ...]
    if "ego_current_state" in data and data["ego_current_state"] is not None:
        ego_state = data["ego_current_state"]
        # compute center of the ego vehicle based on rear axle position
        ego_center_x = ego_state[0] + np.cos(np.arctan2(ego_state[3], ego_state[2])) * EGO_REAR_CENTER_OFFSET
        ego_center_y = ego_state[1] + np.sin(np.arctan2(ego_state[3], ego_state[2])) * EGO_REAR_CENTER_OFFSET
        _plot_agent_rectangle(ax, ego_center_x, ego_center_y, ego_state[2], ego_state[3],
                              EGO_LENGTH, EGO_WIDTH, COLOR_EGO_CURRENT, zorder=5, fill=False, show_heading=False)

        # 6. Plot ego agent (future trajectory)
        # ego_future_gt: (80_steps, 3_features: x, y, yaw)
        if "ego_future_gt" in data and \
           data["ego_future_gt"] is not None and \
           data["ego_future_gt"].shape[0] > 0:

            ego_future_traj = data["ego_future_gt"]
            for i in range(ego_future_traj.shape[0]):
                future_pose = ego_future_traj[i]
                x, y, yaw_rad = future_pose[0], future_pose[1], future_pose[2]
                # cos_yaw = np.cos(yaw_rad)
                # sin_yaw = np.sin(yaw_rad)
                # _plot_agent_rectangle(ax, x, y, cos_yaw, sin_yaw,
                #                      EGO_LENGTH, EGO_WIDTH, COLOR_EGO_FUTURE,
                #                      alpha=0.3, zorder=15)
                # Plot a circle instead of a rectangle at position (x, y)
                # circle = patches.Circle((x, y), radius=0.8, color=COLOR_EGO_FUTURE, alpha=0.8, zorder=15, fill=False)
                # ax.add_patch(circle)
    if "ego_plan" in data:
            ego_future_traj = data["ego_plan"]
            colors =  ['blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'gray']
            for i in range(ego_future_traj.shape[0]):
                future_pose = ego_future_traj[i]
                x, y, yaw_rad = future_pose[0], future_pose[1], future_pose[2]
                # cos_yaw = np.cos(yaw_rad)
                # sin_yaw = np.sin(yaw_rad)
                # color = colors[i % len(colors)]  # Cycle through colors
                # _plot_agent_rectangle(ax, x, y, cos_yaw, sin_yaw,
                #                       2., 1., color=color, alpha=0.3, zorder=16)
                circle = patches.Circle((x, y), radius=0.2, color=COLOR_EGO_PLAN, zorder=16)
                ax.add_patch(circle)
    if "ego_plans" in data:
            ego_future_trajs = data["ego_plans"]
            # list a set of colors for different plans
            colors =  ["red", "orange", "blue", "green"]
            for j in range(ego_future_trajs.shape[0]):  # Iterate over each plan
                ego_future_traj = ego_future_trajs[j]  # Shape (T, 3) with x, y, yaw
                for i in range(ego_future_traj.shape[0]):
                    future_pose = ego_future_traj[i]
                    color = colors[(j // 5) % len(colors)]  # Change color every 5 trajectories
                    x, y, yaw_rad = future_pose[0], future_pose[1], future_pose[2]
                    # cos_yaw = np.cos(yaw_rad)
                    # sin_yaw = np.sin(yaw_rad)
                    # _plot_agent_rectangle(ax, x, y, cos_yaw, sin_yaw,
                    #                       2., 1., COLOR_EGO_PLAN, alpha=0.7, zorder=16)
                    # alpha = 1.0 if j == 0 else 1.0  # Make the first plan more prominent
                    # zorder = 16 if j == 0 else 15  # First plan on top
                    # color = "black" if j == 0 else "green"  # First plan is black
                    circle = patches.Circle((x, y), radius=0.1, color=color, alpha=0.6, zorder=16)
                    ax.add_patch(circle)
            if "ego_plan_scores" in data:
                ego_plan_scores = data["ego_plan_scores"]  # Shape (N,) where N is number of plans
                for j, score in enumerate(ego_plan_scores):
                    # Display the score as text next to the plan circle
                    ax.text(ego_future_trajs[j, -1, 0], ego_future_trajs[j, -1, 1],
                            f"{score:.2f}", fontsize=6, color='black', ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.4, edgecolor='none', boxstyle='round,pad=0.2'),
                            zorder=17)

    # 7. Plot neighbors' future ground truth trajectories
    # "neighbors_future_gt",  # [Pn, T, 3] with x, y, yaw
    # if "neighbors_future_gt" in data:
    #     neighbors_future_gt = data["neighbors_future_gt"]
    #     for i in range(neighbors_future_gt.shape[0]):  # Iterate over each neighbor
    #         for j in range(neighbors_future_gt.shape[1]):  # Iterate over each future
    #             future_pose = neighbors_future_gt[i, j]
    #             x, y = future_pose[0], future_pose[1]
    #             if np.all(future_pose == 0):  # Skip if all zeros
    #                 continue
    #             circle = patches.Circle((x, y), radius=0.3, color=COLOR_NEIGHBORS, alpha=0.5, zorder=12, fill=False)
    #             ax.add_patch(circle)

    # Setup plot appearance
    ax.set_aspect('equal', adjustable='box')
    if "ego_current_state" in data and data["ego_current_state"] is not None:
        ego_x = data["ego_current_state"][0]
        ego_y = data["ego_current_state"][1]
        VIEW_RANGE = 60  # meters in each direction from ego
        ax.set_xlim(ego_x - VIEW_RANGE, ego_x + VIEW_RANGE)
        ax.set_ylim(ego_y - VIEW_RANGE, ego_y + VIEW_RANGE)
    else:
        # Fallback if ego state is not available for centering the view
        ax.autoscale_view()

    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("Scenario Visualization")
    plt.grid(False)

    try:
        plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    except Exception as e:
        # Consider logging the error if a logger is available
        print(f"Error saving plot to {output_filename}: {e}")
    finally:
        plt.close(fig) # Close the figure to free memory, crucial if called in a loop

