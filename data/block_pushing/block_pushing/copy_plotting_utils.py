# NOTE: At some point should just move plotting utils to root and share between the two files. 

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgba
import torch
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
global INNER_STEP
# Default values for run
# DISPLAY_BATCHES = [6, 7, 8, 9]
DISPLAY_BATCHES = [7] 

START_TIMESTEP = None
END_TIMESTEP = None


DISPLAY_RUN_STEPS = [8] # which run_steps to plot
SPECIAL_DENOISING_STEPS = 25 # how many denoising steps to take at each DISPLAY_RUN_STEPS (really only works if you want the same amount of denoising for each override). useful for arbitrarily plotting the first and last 2 denoising steps.
DISPLAY_DENOISING_STEPS = [] # which denoising steps to plot (should be variably set in policy)
PLOT_REFERENCES = {} # on subsequent calls will hold references to plots per batch per step

NUM_TESTING_BATCHES = 4 # 6, 7, 8, 9

NUM_BATCHES = 6 + NUM_TESTING_BATCHES
INNER_STEP = [0] * NUM_BATCHES
TEXT_X_START, TEXT_Y_START = 0.05, 0.95 # Starting coordinates for text on plot

# Needed to condition our moves 
TOTAL_BLOCK_DISTANCE_TRAVELED = dict() 
TOTAL_BLOCK2_DISTANCE_TRAVELED = dict() 
TOTAL_EFFECTOR_DISTANCE_TRAVELED = dict()

# TOTAL_BLOCK_TO_TARGET_DISTANCE = dict()
# TOTAL_BLOCK_TO_TARGET2_DISTANCE = dict()
# TOTAL_BLOCK2_TO_TARGET_DISTANCE = dict()
# TOTAL_BLOCK2_TO_TARGET2_DISTANCE = dict()

for batch in range(NUM_BATCHES): 
    TOTAL_BLOCK_DISTANCE_TRAVELED[batch] = []
    TOTAL_BLOCK2_DISTANCE_TRAVELED[batch] = []
    TOTAL_EFFECTOR_DISTANCE_TRAVELED[batch] = []
    
    # TOTAL_BLOCK_TO_TARGET_DISTANCE[batch] = []
    # TOTAL_BLOCK_TO_TARGET2_DISTANCE[batch] = []
    # TOTAL_BLOCK2_TO_TARGET_DISTANCE[batch] = []
    # TOTAL_BLOCK2_TO_TARGET2_DISTANCE[batch] = []


OVERRIDEN_STEPS = []


NUM_PLOTS = 0
VERTICAL_OFFSETS = [0] * NUM_PLOTS

def initialize_global_plotting(num_plots=None):
    if num_plots is not None:
        NUM_PLOTS = num_plots


# For custom, define which rotations to do per batch. Of form (step, degree, )

# Rotate Lie about the effector by (deg=0, dist=0.1) at each step on the past 1 observation. (Right) 

# For each step, rotate the effector by (deg=0, dist=0.1) at each step on the past 1 observation. (Right)

rotate_at_each_step = []
for i in range(3, 170, 1): 
    rotate_at_each_step.append((i, 180, 0.1))

ROTATIONS_PER_BATCH = [[], [], [], [], [], [], rotate_at_each_step, [], [], []] 

NUM_STEPS = 130 # normally defaults to 130
GLOBAL_STEP_COUNTER = 0 
# Get the "rainbow" colormap
if GLOBAL_STEP_COUNTER <= 130: 
    colormap = plt.cm.get_cmap("rainbow")
else:
    colormap = plt.cm.get_cmap("viridis")

# Generate evenly spaced points in the range [0, 1] and reverse them
points = np.linspace(0, 1, NUM_STEPS)[::-1]

# Map the reversed points to colors using the colormap
COLOR_GRADIENT = colormap(points)


def plot_rectangles(x, y, orientation, color, label, goal_dist_tolerance=0.05, opacity=1.0, is_block=False, ax=None):
    """
    Plots generic rectangles centered at (x, y) with a given rotation and size defined by goal_dist_tolerance.
    """

    # Ensure ax is provided for plotting
    if ax is None:
        raise ValueError("An ax object must be provided for plotting.")

    if is_block: 
        width = goal_dist_tolerance
        height = goal_dist_tolerance
    else: 
        width = 2 * goal_dist_tolerance
        height = 2 * goal_dist_tolerance

    if isinstance(x, torch.Tensor):
        x = x.item()
        y = y.item()
        orientation = orientation.item()

    # Calculate half-width and half-height
    hw, hh = width / 2, height / 2

    # Define the rectangle's corners relative to the center
    corners = np.array([
        [-hw, -hh],
        [ hw, -hh],
        [ hw,  hh],
        [-hw,  hh]
    ])

    # Define the rotation matrix
    angle_rad = orientation
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])

    # Rotate and translate the corners
    rotated_corners = corners @ rotation_matrix.T
    rotated_corners += [x, y]

    if len(rotated_corners.shape) == 3:

        rotated_corners = rotated_corners[0]

    # Create the rotated rectangle as a Polygon
    polygon = patches.Polygon(rotated_corners, closed=True, edgecolor=color, facecolor=color, alpha=opacity, label=label)
    ax.add_patch(polygon)

    # Set aspect ratio to be equal to maintain the shape of the rectangle
    ax.set_aspect('equal', adjustable='box')

def plot_targets (obs, ax): 
    """
    Plot the location of the targets. 
    """


    target_x = obs[10]
    target_y = obs[11]
    target2_x = obs[13]
    target2_y = obs[14]

    # Plot the center of the targets (x) 
    ax.scatter([target_x, target2_x], [target_y, target2_y], color=['lightgray', 'lightgray'], marker='x', s=100, label='Targets')

    # Plot the rectangles using plot_rectangles
    plot_rectangles(target_x, target_y, orientation=0, color='lightpink', label='Target 1', opacity=0.3, ax=ax)
    plot_rectangles(target2_x, target2_y, orientation=0, color='lightgreen', label='Target 2', opacity=0.3, ax=ax)


def get_distance_between(x1, y1, x2, y2):
    """
    Calculate the distance between two points. 
    """

    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    

def plot_blocks(obs_before, obs_after, batch, ax, plot=True):
        '''
        Plots the positions of the blocks. If plot=false, just updates the distance traveled by the blocks and doesn't plot. 
        '''

        obs_step = 1 # Which step of the observation we're at (0 = t-1, the first observation)

        block_before= {
            'x': obs_before[batch][obs_step][0].item(), 
            'y': obs_before[batch][obs_step][1].item(),
            'orientation': obs_before[batch][obs_step][2].item()
        }
        block2_before = {
            'x': obs_before[batch][obs_step][3].item(), 
            'y': obs_before[batch][obs_step][4].item(),
            'orientation': obs_before[batch][obs_step][5]
        }
        
        block_after = {
            'x': obs_after[batch][obs_step][0], 
            'y': obs_after[batch][obs_step][1],
            'orientation': obs_after[batch][obs_step][2]
        }
        block2_after = {
            'x': obs_after[batch][obs_step][3], 
            'y': obs_after[batch][obs_step][4],
            'orientation': obs_after[batch][obs_step][5]
        }

        if plot and batch in DISPLAY_BATCHES: 

            # Plot blocks before
            plot_rectangles(block_before['x'], block_before['y'], block_before['orientation'], 'blue', 'Block Before', opacity=0.5, is_block=True, ax=ax)
            plot_rectangles(block2_before['x'], block2_before['y'], block2_before['orientation'], 'orange', 'Block2 Before', opacity=0.5, is_block=True, ax=ax)
            
            # Plot blocks after
            plot_rectangles(block_after['x'], block_after['y'], block_after['orientation'], 'blue', 'Block After', opacity=1.0, is_block=True, ax=ax)
            plot_rectangles(block2_after['x'], block2_after['y'], block2_after['orientation'], 'orange', 'Block2 After', opacity=1.0, is_block=True, ax=ax)

            # Print the values onto the plot
            ax.text(TEXT_X_START, TEXT_Y_START - (get_vertical_offset(batch=batch)), f'Block 1 Position (After): ({block_before["x"]:.4f}, {block_after["y"]:.4f})', color='black', fontsize=9, transform=ax.transAxes)
            ax.text(TEXT_X_START, TEXT_Y_START - (get_vertical_offset(batch=batch)), f'Block 2 Position (After): ({block2_before["x"]:.4f}, {block2_after["y"]:.4f})', color='black', fontsize=9, transform=ax.transAxes)

        # Calculate the distance traveled by each block. Useful for conditioning when we should lie. 
        block_distance = get_distance_between(block_before['x'], block_before['y'], block_after['x'], block_after['y'])
        block2_distance = get_distance_between(block2_before['x'], block2_before['y'], block2_after['x'], block2_after['y'])
        TOTAL_BLOCK_DISTANCE_TRAVELED[batch].append(block_distance)
        TOTAL_BLOCK2_DISTANCE_TRAVELED[batch].append(block2_distance)
        
        # TODO: on the 'Compute distance traveled by blocks and update rolling lists' update the way we use get_total_block_distance_traveled. 
    
        if plot: 
            # Print the distance traveled
            ax.text(TEXT_X_START, TEXT_Y_START - (get_vertical_offset(batch=batch)), f'Block 1 Distance Traveled: {block_distance:.4f}', color='black', fontsize=9, transform=ax.transAxes)
            ax.text(TEXT_X_START, TEXT_Y_START - (get_vertical_offset(batch=batch)), f'Block 2 Distance Traveled: {block2_distance:.4f}', color='black', fontsize=9, transform=ax.transAxes)

            # Print the total distance traveled
            ax.text(TEXT_X_START, TEXT_Y_START - (get_vertical_offset(batch=batch)), f'Total Distance Blocks Traveled: {block_distance + block2_distance:.4f}', color='black', fontsize=9, transform=ax.transAxes)


def plot_effector(obs_before, obs_after, batch, ax, plot=True):
    obs_step = 1

    starting_effector_actual = {
        'x': obs_before[batch][obs_step][6], 
        'y': obs_before[batch][obs_step][7],
    }
    starting_effector_target = {
        'x': obs_before[batch][obs_step][8],
        'y': obs_before[batch][obs_step][9],
    }

    effector_actual = {
        'x': obs_after[batch][obs_step][6], 
        'y': obs_after[batch][obs_step][7],
    }
    effector_target = {
        'x': obs_after[batch][obs_step][8],
        'y': obs_after[batch][obs_step][9],
    }

    if plot: 
        # Plot the starting effector position
        ax.scatter(
            starting_effector_actual["x"], starting_effector_actual['y'], 
            color='teal', marker='o', s=100, label='Starting Actual Effector', alpha=0.5
        )
        ax.scatter(
            starting_effector_target['x'], starting_effector_target['y'], 
            color='purple', marker='o', s=100, label='Starting Target Effector', alpha=0.5
        )

        # Plot the final effector position
        ax.scatter(
            effector_actual['x'], effector_actual['y'], 
            color='green', marker='o', s=100, label='Actual Effector', alpha=0.5
        )
        ax.scatter(
            effector_target['x'], effector_target['y'], 
            color='red', marker='o', s=100, label='Target Effector', alpha=0.5
        )

    # Calculate effector distance traveled
    effector_distance = get_distance_between(starting_effector_actual['x'], starting_effector_actual['y'], effector_actual['x'], effector_actual['y'])
    
    TOTAL_EFFECTOR_DISTANCE_TRAVELED[batch].append(effector_distance)

    if plot:
        # Print final effector values
        ax.text(
            TEXT_X_START, TEXT_Y_START, 
            f'Effector Value (Actual): ({effector_actual["x"]:.4f}, {effector_actual["y"]:.4f})', 
            color='green', fontsize=9, transform=ax.transAxes
        )
        ax.text(
            TEXT_X_START, TEXT_Y_START - get_vertical_offset(batch=batch), 
            f'Effector Value (Target): ({effector_target["x"]:.4f}, {effector_target["y"]:.4f})', 
            color='red', fontsize=9, transform=ax.transAxes
        )

        # Print starting effector values
        ax.text(
            TEXT_X_START, TEXT_Y_START - get_vertical_offset(batch=batch), 
            f'Effector Value (Actual, At Start): ({starting_effector_actual["x"]:.4f}, {starting_effector_actual["y"]:.4f})', 
            color='teal', fontsize=9, transform=ax.transAxes
        )
        ax.text(
            TEXT_X_START, TEXT_Y_START - get_vertical_offset(batch=batch), 
            f'Effector Value (Target, At Start): ({starting_effector_target["x"]:.4f}, {starting_effector_target["y"]:.4f})', 
            color='purple', fontsize=9, transform=ax.transAxes
        )

        ax.text(
            TEXT_X_START, TEXT_Y_START - get_vertical_offset(batch=batch), 
            f'Effector Distance Traveled: {effector_distance:.4f}', 
            color='black', fontsize=9, transform=ax.transAxes
        )

def get_blocks_to_target_distance(obs_after):
    """
    Helper function for 'plot_distance_from_target'. Calculates the distance between each of the blocks and each of the targets.
    """

    obs_step = 2

    block = {
    'x': obs_after[0], 
    'y': obs_after[1],
    'orientation': obs_after[2]
    }
        
    block2 = {
        'x': obs_after[3], 
        'y': obs_after[4],
        'orientation': [5]
    }

    target = {
        'x': obs_after[10], 
        'y': obs_after[11],
    }
    target2 = {
        'x': obs_after[13], 
        'y': obs_after[14],
    }

    block_to_target = get_distance_between(block['x'], block['y'], target['x'], target['y'])
    block_to_target2 = get_distance_between(block['x'], block['y'], target2['x'], target2['y'])
    block2_to_target = get_distance_between(block2['x'], block2['y'], target['x'], target['y'])
    block2_to_target2 = get_distance_between(block2['x'], block2['y'], target2['x'], target2['y'])

    return block_to_target, block_to_target2, block2_to_target, block2_to_target2

def plot_distance_from_target(batch, obs_after, ax):
    """
    Calculate the distance between each of the blocks and each of the targets and plot them on the plot.
    """

    block_to_target, block_to_target2, block2_to_target, block2_to_target2 = get_blocks_to_target_distance(obs_after=obs_after)

    #NOTE: Not sure why I was appending block_to_target on this?? I think I was trying to append this to TOTAL_BLOCK_TO_TARGET_DISTANCE
    # TOTAL_BLOCK_DISTANCE_TRAVELED[batch].append(block_to_target)

    
    # TOTAL_BLOCK_TO_TARGET2_DISTANCE[batch].append(block_to_target2)
    # TOTAL_BLOCK2_TO_TARGET_DISTANCE[batch].append(block2_to_target)
    # TOTAL_BLOCK2_TO_TARGET2_DISTANCE[batch].append(block2_to_target2)

    def in_target_range(distance):
        return 'green' if distance <= 0.05 else 'black'

    ax.text(
        TEXT_X_START, TEXT_Y_START - get_vertical_offset(batch=batch),
        f'Block 1 to Target 1: {block_to_target:.4f}',
        color=in_target_range(block_to_target), fontsize=9, transform=ax.transAxes
    )
    ax.text(
        TEXT_X_START, TEXT_Y_START - get_vertical_offset(batch=batch),
        f'Block 1 to Target 2: {block_to_target2:.4f}',
        color=in_target_range(block_to_target2), fontsize=9, transform=ax.transAxes
    )
    ax.text(
        TEXT_X_START, TEXT_Y_START - get_vertical_offset(batch=batch),
        f'Block 2 to Target 1: {block2_to_target:.4f}',
        color=in_target_range(block2_to_target), fontsize=9, transform=ax.transAxes
    )
    ax.text(
        TEXT_X_START, TEXT_Y_START - get_vertical_offset(batch=batch),
        f'Block 2 to Target 2: {block2_to_target2:.4f}',
        color=in_target_range(block2_to_target2), fontsize=9, transform=ax.transAxes
    )

    return block_to_target, block_to_target2, block2_to_target, block2_to_target2

def is_successful(obs):
    """
    Calculates whether we were successful and the closest target per block. 
    """
    targets = ["target", "target2"]
    goal_dist_tolerance = 0.05
    block_to_target, block_to_target2, block2_to_target, block2_to_target2 = get_blocks_to_target_distance(obs_after=obs)

    def _closest_target(block):
        if block == "block":
            dists = [block_to_target, block_to_target2]
        else:
            dists = [block2_to_target, block2_to_target2]

        closest_target = targets[np.argmin(dists)]
        closest_dist = np.min(dists)

        # Is it in the closest target?
        in_target = closest_dist < goal_dist_tolerance
        return closest_dist, closest_target, in_target

    b0_closest_dist, b0_closest_target, b0_in_target = _closest_target("block")
    b1_closest_dist, b1_closest_target, b1_in_target = _closest_target("block2")

    if b0_in_target and b1_in_target and (b0_closest_target != b1_closest_target):
        # return True, b0_closest_dist, b0_closest_target, b1_closest_dist, b1_closest_target
        return True, b0_in_target, b1_in_target

    # return False, b0_closest_dist, b0_closest_target, b1_closest_dist, b1_closest_target
    return False, b0_in_target, b1_in_target


def plot_isolated_blocks(obs_before, ax, label_position=False):
    """
    Only plot the block location, no need to track across time.
    """

    # obs_step = 1 # Which step of the observation we're at (0 = t-1, the first observation)

    block_before= {
        'x': obs_before[0], 
        'y': obs_before[1],
        'orientation': [2]
    }
    block2_before = {
        'x': obs_before[3], 
        'y': obs_before[4],
        'orientation': obs_before[5]
    }

    plot_rectangles(block_before['x'], block_before['y'], block_before['orientation'], 'blue', 'Block Before', opacity=1, is_block=True, ax=ax)
    plot_rectangles(block2_before['x'], block2_before['y'], block2_before['orientation'], 'orange', 'Block2 Before', opacity=1, is_block=True, ax=ax)

    if label_position: 
        # Print the values onto the plot
        ax.text(TEXT_X_START, TEXT_Y_START-0.175, f'Block 1 Position: ({block_before["x"]:.4f}, {block_before["y"]:.4f})', color='black', fontsize=9, transform=ax.transAxes)
        ax.text(TEXT_X_START, TEXT_Y_START-0.2, f'Block 2 Position: ({block2_before["x"]:.4f}, {block2_before["y"]:.4f})', color='black', fontsize=9, transform=ax.transAxes)


def color_code_at_k_labels(demo_num):
    """
    Color code the labels at k. 
    """
    (fig, ax) = PLOT_REFERENCES[f"demo_num_{demo_num}"]

    # Color Code at k 
    ax.text(TEXT_X_START, TEXT_Y_START, 'Cyan: Time Step k', color='cyan', fontsize=9, transform=ax.transAxes)
    ax.text(TEXT_X_START, TEXT_Y_START - 0.025, 'Black: Path Division', color='black', fontsize=9, transform=ax.transAxes)
    ax.text(TEXT_X_START, TEXT_Y_START - 0.05, 'Path 0: Red -> Orange', color='red', fontsize=9, transform=ax.transAxes)
    ax.text(TEXT_X_START, TEXT_Y_START - 0.075, 'Path 1: Blue -> Purple', color='blue', fontsize=9, transform=ax.transAxes)

    # ax.text(TEXT_X_START, TEXT_Y_START - 0.1, 'path1_before_k, path0_after_k, path0_before_k, path1_after_k ', color='black', fontsize=9, transform=ax.transAxes)

    # ax.text(TEXT_X_START, TEXT_Y_START - .125, 'i.e. should correspond to blue -> orange -> red -> purple on the original graph', color='black', fontsize=9, transform=ax.transAxes)

    
def color_code_3_segments_labels(demo_num):
    """
    Color code the labels for the 3 segments. 
    """
    (fig, ax) = PLOT_REFERENCES[f"demo_num_{demo_num}"]

    # Color code 3 segments
    # Blue is first block to target
    # Yellow is return to center
    # Green is pivot point
    # Red is second block to target

    ax.text(TEXT_X_START, TEXT_Y_START, 'Red: First block to traget', color='red', fontsize=9, transform=ax.transAxes)
    ax.text(TEXT_X_START, TEXT_Y_START - 0.025, 'Yellow: Return to center', color='yellow', fontsize=9, transform=ax.transAxes)
    ax.text(TEXT_X_START, TEXT_Y_START - 0.05, 'Green: Pivot Point', color='green', fontsize=9, transform=ax.transAxes)
    ax.text(TEXT_X_START, TEXT_Y_START - 0.075, 'Blue: Second block to target', color='blue', fontsize=9, transform=ax.transAxes)
    

def write_dist_to_target_demo(target_num, current_num, dist):
    """
    Write on the plot the distance of the current environment to the target environment. 
    """
    
    (fig, ax) = PLOT_REFERENCES[f"demo_num_{current_num}"]

    ax.text(TEXT_X_START, TEXT_Y_START - .150, f"Distance to Target Num ({target_num}): {dist:.4f}", color='black', fontsize=9, transform=ax.transAxes)
    


def close_global_plots(obs, demo_num, coloring):
    
    (fig, ax) = PLOT_REFERENCES[f"demo_num_{demo_num}"]
    plot_isolated_blocks(obs_before=obs, ax=ax, label_position=False)
    # plot_labels(ax)
    if coloring == "3_segments":
        color_code_3_segments_labels(demo_num)
    elif coloring == "at_k":
        color_code_at_k_labels(demo_num)
    fig.savefig(f"global_plots/demo_num_{demo_num}.png", bbox_inches='tight')
    plt.close()

def init_global_plots(obs_before, demo_num):
    

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title(f"Training Demonstration {demo_num}")
    ax.set_xlim(-0.5, 1.0)
    ax.set_ylim(-1.0, 0.5)
    ax.set_xlabel("X-axis Label")
    ax.set_ylabel("Y-axis Label")

    plot_key = f"demo_num_{demo_num}"
    PLOT_REFERENCES[plot_key] = (fig, ax)

    plot_isolated_blocks(obs_before=obs_before, ax=ax, label_position=True)
    plot_targets(obs=obs_before, ax=ax)

    return fig, ax
       
def init_denoising_trajectories(obs, run_step, eff_x, eff_y): 
    """
    We're going to plot all the trajectories being generated through the denoising process within a given run step. 
    """
    obs = obs.detach().cpu().numpy()
    for batch in DISPLAY_BATCHES:
        for denoising_step in DISPLAY_DENOISING_STEPS:
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.set_title(f"Batch {batch} at Run Step {run_step}, Denoising Step {denoising_step}")
            ax.set_xlim(-0.5, 1.0)
            ax.set_ylim(-1.0, 0.5)
            ax.set_xlabel("X-axis Label")
            ax.set_ylabel("Y-axis Label")

            plot_key = f"batch_{batch}_run_step_{run_step}_denoising_{denoising_step}_diff"
            PLOT_REFERENCES[plot_key] = (fig, ax)

            plot_isolated_blocks(batch=batch, obs_before=obs, ax=ax, obs_step=2)
            plot_targets(obs=obs, batch=batch, ax=ax)

            ax.scatter(
                eff_x.item(), eff_y.item(), color='black', marker='o', s=100, label='Effector', alpha=0.5
            )

def plot_denoising_trajectories(trajectory, run_step, demo_num, color=None):
    """
    Plot the generated trajectory at denoising step, denoising_step. Used for global_plots.

    NOTE: Assumes that run_step is relative to its episode (i.e. starts at 0)
    """
    
    (fig, ax) = PLOT_REFERENCES[f"demo_num_{demo_num}"]
    action_horizon = 0 # (Here, we are only plotting a single step taken.)
    x_coords = trajectory[0]
    y_coords = trajectory[1]


    if color is None or color == 'gradient':
        assert NUM_STEPS is not None, "NUM_STEPS must be defined for gradient coloring."
        color = COLOR_GRADIENT[run_step % NUM_STEPS]

    ax.scatter(x_coords, y_coords, color=color, alpha=0.5, linewidth=2, label='Denoised Trajectory')
    




