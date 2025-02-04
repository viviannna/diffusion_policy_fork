import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ------------------------------------------------------------------
# Functions to compute distances between blocks and targets, 
# determine the closest target for each block, and evaluate 
# whether the blocks have successfully reached their targets.
# ------------------------------------------------------------------

def is_successful(obs):
    """
    Determines if both blocks are inside different targets.
    """
    distances = get_blocks_to_target_distance(obs)
    
    b0_closest_dist, b0_closest_target, b0_in_target = find_closest_target(distances, 0)
    b1_closest_dist, b1_closest_target, b1_in_target = find_closest_target(distances, 1)
    
    success = b0_in_target and b1_in_target and (b0_closest_target != b1_closest_target)
    
    return success, b0_in_target, b1_in_target

def calculate_distance(p1, p2):
    """
    Calculate the Euclidean distance between two points.
    """
    return np.sqrt((p2['x'] - p1['x']) ** 2 + (p2['y'] - p1['y']) ** 2)

def extract_block_and_target_positions(obs_after):
    """
    Extracts block and target positions from observation data.
    """
    blocks = [
        {'x': obs_after[0], 'y': obs_after[1], 'orientation': obs_after[2]},
        {'x': obs_after[3], 'y': obs_after[4], 'orientation': obs_after[5]}
    ]
    
    targets = [
        {'x': obs_after[10], 'y': obs_after[11]},
        {'x': obs_after[13], 'y': obs_after[14]}
    ]
    
    return blocks, targets

def get_blocks_to_target_distance(obs_after):
    """
    Calculates the distance between each block and each target.
    """
    blocks, targets = extract_block_and_target_positions(obs_after)
    
    distances = {
        (b_idx, t_idx): calculate_distance(block, target)
        for b_idx, block in enumerate(blocks)
        for t_idx, target in enumerate(targets)
    }
    
    return distances

def find_closest_target(distances, block_idx):
    """
    Determines the closest target for a given block and whether it is within tolerance.
    """
    goal_dist_tolerance = 0.05
    target_keys = ["target", "target2"]
    
    block_distances = {key: val for (b, key), val in distances.items() if b == block_idx}
    closest_target_idx = min(block_distances, key=block_distances.get)
    
    return block_distances[closest_target_idx], target_keys[closest_target_idx], block_distances[closest_target_idx] < goal_dist_tolerance



# ------------------------------------------------------------------
# Generic Plotting Utilities
# ------------------------------------------------------------------
# These functions add blocks, targets, and rectangles to a given
# Matplotlib axis (ax). They do not create new figures but instead
# modify existing plots.
# ------------------------------------------------------------------

# Stores the current vertical offset for text labels on each axis
AX_TEXT_OFFSET = {}

def initialize_text_offset(ax):
    """
    Initializes the text offset for labels on a given axis.
    
    Parameters:
        ax (matplotlib.axes.Axes): The axis to track offset for.
    """
    AX_TEXT_OFFSET[ax] = TEXT_Y_START

def increment_text_offset(ax, step=0.025):
    """
    Increments and returns the next available text offset for the given axis.

    Parameters:
        ax (matplotlib.axes.Axes): The axis to track offset for.
        step (float, optional): The spacing between text labels. Default is 0.025.

    Returns:
        float: The updated text offset for the next label.
    """
    if ax not in AX_TEXT_OFFSET:
        initialize_text_offset(ax)
    
    AX_TEXT_OFFSET[ax] -= step
    return AX_TEXT_OFFSET[ax]

def draw_blocks(obs_before, ax, show_labels=False):
    """
    Draws the initial positions of blocks.

    Parameters:
        obs_before (array-like): Initial observation data of the blocks.
        ax (matplotlib.axes.Axes): Matplotlib axis for plotting.
        show_labels (bool, optional): Whether to display block positions as text. Default is False.
    """
    blocks = [
        {'x': obs_before[0], 'y': obs_before[1], 'orientation': obs_before[2]},
        {'x': obs_before[3], 'y': obs_before[4], 'orientation': obs_before[5]}
    ]

    colors = ['blue', 'orange']
    labels = ['Block 1', 'Block 2']

    for i, block in enumerate(blocks):
        draw_rectangle(
            x=block['x'], y=block['y'], orientation=block['orientation'], 
            color=colors[i], label=labels[i], opacity=1, is_block=True, ax=ax
        )

    if show_labels:
        ax.text(TEXT_X_START, increment_text_offset(ax), 
                f'Block 1 Position: ({blocks[0]["x"]:.4f}, {blocks[0]["y"]:.4f})', 
                color='black', fontsize=9, transform=ax.transAxes)
        ax.text(TEXT_X_START, increment_text_offset(ax), 
                f'Block 2 Position: ({blocks[1]["x"]:.4f}, {blocks[1]["y"]:.4f})', 
                color='black', fontsize=9, transform=ax.transAxes)

def draw_targets(obs, ax):
    """
    Draws the target locations.

    Parameters:
        obs (array-like): Observation data containing target positions.
        ax (matplotlib.axes.Axes): Matplotlib axis for plotting.
    """
    targets = [
        {'x': obs[10], 'y': obs[11]},
        {'x': obs[13], 'y': obs[14]}
    ]

    colors = ['lightpink', 'lightgreen']
    labels = ['Target 1', 'Target 2']

    ax.scatter([t['x'] for t in targets], [t['y'] for t in targets], 
               color='lightgray', marker='x', s=100, label='Targets')

    for i, target in enumerate(targets):
        draw_rectangle(
            x=target['x'], y=target['y'], orientation=0, 
            color=colors[i], label=labels[i], opacity=0.3, ax=ax
        )

def draw_rectangle(x, y, orientation, color, label, goal_dist_tolerance=0.05, opacity=1.0, is_block=False, ax=None):
    """
    Draws a rotated rectangle representing a block or a target.

    Parameters:
        x (float): X-coordinate of the center.
        y (float): Y-coordinate of the center.
        orientation (float): Rotation angle in radians.
        color (str): Color of the rectangle.
        label (str): Label for the rectangle.
        goal_dist_tolerance (float, optional): Defines rectangle size. Default is 0.05.
        opacity (float, optional): Transparency level. Default is 1.0.
        is_block (bool, optional): Determines block or target size. Default is False.
        ax (matplotlib.axes.Axes): Matplotlib axis for plotting.
    """
    if ax is None:
        raise ValueError("An ax object must be provided for plotting.")

    # Define rectangle dimensions
    width, height = (goal_dist_tolerance, goal_dist_tolerance) if is_block else (2 * goal_dist_tolerance, 2 * goal_dist_tolerance)
    hw, hh = width / 2, height / 2  # Half-width and half-height

    # Define rectangle corners
    corners = np.array([
        [-hw, -hh], [ hw, -hh], [ hw,  hh], [-hw,  hh]
    ])

    # Apply rotation transformation
    rotation_matrix = np.array([
        [np.cos(orientation), -np.sin(orientation)],
        [np.sin(orientation),  np.cos(orientation)]
    ])
    
    rotated_corners = corners @ rotation_matrix.T + [x, y]

    # Ensure valid shape
    if len(rotated_corners.shape) == 3:
        rotated_corners = rotated_corners[0]

    # Create and add the rectangle polygon
    polygon = patches.Polygon(rotated_corners, closed=True, edgecolor=color, 
                               facecolor=color, alpha=opacity, label=label)
    ax.add_patch(polygon)
    ax.set_aspect('equal', adjustable='box')


# ------------------------------------------------------------------
# Full Trajectory Plot Setup
# ------------------------------------------------------------------
# These functions initialize and configure plots that visualize 
# an entire trajectory in a single figure. Unlike step-by-step 
# plots, these provide an overview by displaying all relevant 
# steps on one shared axis.
# ------------------------------------------------------------------

#Dictionary to store figure and axis references for trajectory plots
PLOT_REGISTRY = {}

# Default text starting positions on the plot
TEXT_X_START, TEXT_Y_START = 0.05, 0.95  

def setup_full_trajectory_plot(obs_before, demo_num):
    """
    Initializes a plot for visualizing an entire trajectory in a single figure.

    Parameters:
        obs_before (array-like): Initial state observation data.
        demo_num (int): Demonstration number for labeling.

    Returns:
        tuple: (fig, ax) Matplotlib figure and axis objects.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title(f"Training Demonstration {demo_num}")
    ax.set_xlim(-0.5, 1.0)
    ax.set_ylim(-1.0, 0.5)
    ax.set_xlabel("X-axis Label")
    ax.set_ylabel("Y-axis Label")

    plot_key = f"demo_{demo_num}"
    PLOT_REGISTRY[plot_key] = (fig, ax)

    draw_blocks(obs_before, ax, show_labels=True)
    draw_targets(obs_before, ax)

    return fig, ax

def finalize_full_trajectory_plot(obs, demo_num, coloring):
    """
    Finalizes and saves a full trajectory plot by adding final block positions
    and applying optional color coding.

    Parameters:
        obs (array-like): Final state observation data.
        demo_num (int): Demonstration number for labeling and file naming.
        coloring (str): Coloring mode, either "3_segments" or "at_k".
    """
    fig, ax = PLOT_REGISTRY[f"demo_{demo_num}"]

    draw_blocks(obs, ax, show_labels=False)  # Add final block positions

    # Apply color coding based on the specified mode
    if coloring == "3_segments":
        add_3_segment_legend(demo_num)
    elif coloring == "at_k":
        add_at_k_color_legend(demo_num)

    # Save and close the figure
    fig.savefig(f"global_plots/demo_{demo_num}.png", bbox_inches='tight')
    plt.close(fig)

# ------------------------------------------------------------------
# Color-Coded Legend for Trajectory Plots
# ------------------------------------------------------------------
# These functions add labeled legends to a trajectory plot, 
# explaining the meaning of different color-coded paths or 
# segments. They do not determine the actual colors used 
# for the trajectory, which is handled elsewhere.
# ------------------------------------------------------------------


def add_3_segment_legend(demo_num):
    """
    Adds a color-coded legend to indicate three key movement segments in the trajectory plot.

    Segments:
    - **Red**: First block to target
    - **Yellow**: Return to center
    - **Green**: Pivot point
    - **Blue**: Second block to target

    Parameters:
        demo_num (int): Demonstration number for labeling.
    """
    fig, ax = PLOT_REGISTRY[f"demo_{demo_num}"]

    segment_labels = [
        ("Red: First block to target", 'red'),
        ("Yellow: Return to center", 'yellow'),
        ("Green: Pivot Point", 'green'),
        ("Blue: Second block to target", 'blue'),
    ]

    for text, color in segment_labels:
        ax.text(TEXT_X_START, increment_text_offset(ax), text, color=color, fontsize=9, transform=ax.transAxes)

def add_at_k_color_legend(demo_num):
    """
    Adds a color-coded legend explaining the meaning of path divisions at step k.

    Legend Assignments:
    - **Cyan**: Time step k
    - **Black**: Path division
    - **Red → Orange**: Path 0
    - **Blue → Purple**: Path 1

    Parameters:
        demo_num (int): Demonstration number for labeling.
    """
    fig, ax = PLOT_REGISTRY[f"demo_{demo_num}"]

    coloring_labels = [
        ("Cyan: Time Step k", 'cyan'),
        ("Black: Path Division", 'black'),
        ("Path 0: Red -> Orange", 'red'),
        ("Path 1: Blue -> Purple", 'blue'),
    ]

    for text, color in coloring_labels:
        ax.text(TEXT_X_START, increment_text_offset(ax), text, color=color, fontsize=9, transform=ax.transAxes)

def label_environment_distance(current_num, target_num, dist):
    """
    Adds a label on the plot indicating the distance between 
    the current environment and the target environment.

    This visually represents how similar the two environments are.

    Parameters:
        current_num (int): The current environment's demonstration number.
        target_num (int): The target environment's demonstration number.
        dist (float): The computed distance between the two environments.
    """
    fig, ax = PLOT_REGISTRY[f"demo_{current_num}"]

    # Ensure text offset tracking is initialized
    
    # Add distance label with dynamically adjusted positioning
    ax.text(TEXT_X_START, increment_text_offset(ax), 
            f"Distance to Target Num ({target_num}): {dist:.4f}", 
            color='black', fontsize=9, transform=ax.transAxes)



# ------------------------------------------------------------------
# Visualization of Effector Actions
# ------------------------------------------------------------------
# These functions plot the movement steps of the effector 
# based on the action dictionary, showing how the effector 
# moves throughout an episode. 
# 
# The plotted points represent discrete actions taken by the 
# effector at each step, allowing for a clear visualization 
# of its movement patterns. The color scheme used for these 
# actions is determined separately, ensuring consistency 
# across different plots.
# ------------------------------------------------------------------

def plot_effector_actions(action, run_step, demo_num, color=None):
    """
    Plot the effector's movement steps based on the action dictionary.

    Args:
        action (tuple): A tuple containing x and y coordinates of the effector's movement.
        run_step (int): The current step relative to the episode (starts at 0).
        demo_num (int): The demo number for referencing the plot.
        color (str, optional): The color for plotting the actions. Defaults to gradient coloring.

    Raises:
        AssertionError: If TOTAL_NUM_STEPS is not defined when using gradient coloring.
    """

    # Retrieve figure and axis for the given demo number
    fig, ax = PLOT_REGISTRY[f"demo_{demo_num}"]

    # Extract x and y coordinates (horizon of 1)
    x_coords, y_coords = action

    # Determine color
    if not color or color == 'gradient':
        if TOTAL_NUM_STEPS is None: 
            TOTAL_NUM_STEPS = 130
            
        color = COLOR_GRADIENT[run_step % TOTAL_NUM_STEPS]

    # Plot effector movement
    ax.scatter(x_coords, y_coords, color=color, alpha=0.5, linewidth=2, label='Effector Actions')
