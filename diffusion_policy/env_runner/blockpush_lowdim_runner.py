import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.block_pushing.block_pushing_multimodal import BlockPushMultimodal
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from gym.wrappers import FlattenObservation

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.colors as mcolors

from collections import deque
import matplotlib
matplotlib.use('Agg')

global global_translate, global_translation_angle, global_translation_distance

class BlockPushLowdimRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            fps=5,
            crf=22,
            past_action=False,
            abs_action=False,
            obs_eef_target=True,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)
        

        if n_envs is None:
            n_envs = n_train + n_test

        task_fps = 10
        steps_per_render = max(10 // fps, 1)

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    FlattenObservation(
                        BlockPushMultimodal(
                            control_frequency=task_fps,
                            shared_memory=False,
                            seed=seed,
                            abs_action=abs_action
                        )
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
        
        

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            # enable_render = i < n_train_vis
            if i in [7, 8, 10, 21, 37, 43]:
                enable_render = True
            else:
                enable_render = False

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    # filename = pathlib.Path(output_dir).joinpath(
                    #     'media', wv.util.generate_id() + ".mp4")
                    # filename.parent.mkdir(parents=False, exist_ok=True)
                    # filename = str(filename)

                    filename = self.generate_sequential_filename(n_train, output_dir, i, test=False)
                    print("filename:    ", filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            # enable_render = i < n_test_vis

            if (i+n_train) in [7, 8, 10, 21, 37, 43]:
                enable_render = True
            else:
                enable_render = False

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    # filename = pathlib.Path(output_dir).joinpath(
                    #     'media', wv.util.generate_id() + ".mp4")
                    # filename.parent.mkdir(parents=False, exist_ok=True)

                    filename = self.generate_sequential_filename(n_train, output_dir, i, test=True)
                    # filename = str(filename)
                    env.env.file_path = filename
                    print("filename:    ", filename)

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # env = AsyncVectorEnv(env_fns)
        env = SyncVectorEnv(env_fns)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.obs_eef_target = obs_eef_target

        




    def generate_sequential_filename(self, n_train, output_dir, i, test=False):
        media_dir = pathlib.Path(output_dir).joinpath('media')
        media_dir.mkdir(parents=True, exist_ok=True)
        
        if test: 
            batch = i + n_train  # offset by the number of training batches
        else:
            batch = i
        
        # Create the batch directory
        batch_dir = media_dir.joinpath(f'batch_{batch}')
        batch_dir.mkdir(parents=True, exist_ok=True)  # Ensure the batch folder is created

        # Create the new file path with the next number
        filename = batch_dir.joinpath(f'sim_batch_{batch}.mp4')

        return str(filename)


    
    def plot_desired_trajectory(self, desired_trajectory, batch):

        action_horizon = desired_trajectory.shape[1]
        x_coords = desired_trajectory[batch, :, 0] 
        y_coords = desired_trajectory[batch, :, 1]

        # Define a custom red colormap from light to dark red
        red_cmap = mcolors.LinearSegmentedColormap.from_list(
            'red_gradient', [(1, 0.8, 0.8), (1, 0, 0)], N=256
        )

        time_steps = np.arange(action_horizon)  # Create time step indices

        # Normalize time steps to range [0, 1] for colormap
        norm = plt.Normalize(time_steps.min(), time_steps.max())
        colors = red_cmap(norm(time_steps))

        # Plot each point with the corresponding color
        for j in range(action_horizon):
            plt.scatter(x_coords[j], y_coords[j], color=colors[j], label=f'Batch {batch+1}' if j == 0 and batch == 0 else "", edgecolor='k')

       
        plt.text(self.text_x, self.text_y_start - (self.get_vertical_offset(batch=batch)), f'Desired Trajectory End Point: ({x_coords[-1]:.4f}, {y_coords[-1]:.4f})', color='red', fontsize=9, transform=plt.gca().transAxes)


    def plot_rectangles(self, x, y, orientation, color, label, goal_dist_tolerance=0.05, opacity=1.0, is_block=False):

        if is_block: 
            width =  goal_dist_tolerance
            height =  goal_dist_tolerance
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

        # Create the rotated rectangle as a Polygon
        polygon = patches.Polygon(rotated_corners, closed=True, edgecolor=color, facecolor=color, alpha=opacity, label=label)
        plt.gca().add_patch(polygon)

        # # Annotate the coordinates
        # plt.text(x, y, f'({x:.2f}, {y:.2f})', color='black', fontsize=9, ha='center', va='bottom')

        # Set aspect ratio to be equal to maintain the shape of the rectangle
        plt.gca().set_aspect('equal', adjustable='box')

    def get_total_block_distance_traveled(self, obs_before, obs_after, batch): 
        """
        Used only when doing total distance traveled for blocks on the outside loop. Similar functionality to plot blocks and its use of get_block_distance() but without plotting. 
        """
        obs_step = 0

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

        block_1_distance = self.get_distance_between_blocks(block_before, block_after)
        block_2_distance = self.get_distance_between_blocks(block2_before, block2_after)

        block_distance = block_1_distance + block_2_distance

        return block_distance
    
    def get_blocks_to_target_distance(self, obs_after, batch):

        # obs_step = 2 # (Most recent) # for transformer? 
        obs_step = 1 # (Most recent) # for CNN based

        block = {
        'x': obs_after[batch][obs_step][0], 
        'y': obs_after[batch][obs_step][1],
        'orientation': obs_after[batch][obs_step][2]
        }
         
        block2 = {
            'x': obs_after[batch][obs_step][3], 
            'y': obs_after[batch][obs_step][4],
            'orientation': obs_after[batch][obs_step][5]
        }

        target = {
            'x': obs_after[batch][obs_step][10], 
            'y': obs_after[batch][obs_step][11],
        }
        target2 = {
            'x': obs_after[batch][obs_step][13], 
            'y': obs_after[batch][obs_step][14],
        }

        block_to_target = self.get_distance_between_blocks(block, target)
        block_to_target2 = self.get_distance_between_blocks(block, target2)

        block2_to_target = self.get_distance_between_blocks(block2, target)
        block2_to_target2 = self.get_distance_between_blocks(block2, target2)

        return block_to_target, block_to_target2, block2_to_target, block2_to_target2

    
    def plot_distance_from_target(self, batch, obs_after):

        block_to_target, block_to_target2, block2_to_target, block2_to_target2 = self.get_blocks_to_target_distance(obs_after=obs_after, batch=batch)

        self.total_block_to_target_distance[batch].append(block_to_target)
        self.total_block_to_target2_distance[batch].append(block_to_target2)
        self.total_block2_to_target_distance[batch].append(block2_to_target)
        self.total_block2_to_target2_distance[batch].append(block2_to_target2)

        def in_target_range(distance):
            if distance <= 0.05:
                return 'green'
            else:
                return 'black'

        plt.text(self.text_x, self.text_y_start - (self.get_vertical_offset(batch=batch)), f'Block 1 to Target 1: {block_to_target:.4f}', color=in_target_range(block_to_target), fontsize=9, transform=plt.gca().transAxes)
        plt.text(self.text_x, self.text_y_start - (self.get_vertical_offset(batch=batch)), f'Block 1 to Target 2: {block_to_target2:.4f}', color=in_target_range(block_to_target2), fontsize=9, transform=plt.gca().transAxes)

        plt.text(self.text_x, self.text_y_start - (self.get_vertical_offset(batch=batch)), f'Block 2 to Target 1: {block2_to_target:.4f}', color=in_target_range(block2_to_target), fontsize=9, transform=plt.gca().transAxes)
        plt.text(self.text_x, self.text_y_start - (self.get_vertical_offset(batch=batch)), f'Block 2 to Target 2: {block2_to_target2:.4f}', color=in_target_range(block2_to_target2), fontsize=9, transform=plt.gca().transAxes)


        return block_to_target, block_to_target2, block2_to_target, block2_to_target2
        

    def get_distance_between_blocks(self, block_before, block_after):
        x1, y1 = block_before['x'], block_before['y']
        x2, y2 = block_after['x'], block_after['y']

        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    

    

    def plot_blocks(self, obs_before, obs_after, batch):
        '''
        Plots the positions of the blocks. 
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

        # Plot blocks before
        self.plot_rectangles(block_before['x'], block_before['y'], block_before['orientation'], 'blue', 'Block Before', opacity=0.5, is_block=True)
        self.plot_rectangles(block2_before['x'], block2_before['y'], block2_before['orientation'], 'orange', 'Block2 Before', opacity=0.5, is_block=True)
        
        # Plot blocks after
        self.plot_rectangles(block_after['x'], block_after['y'], block_after['orientation'], 'blue', 'Block After', opacity=1.0, is_block=True)
        self.plot_rectangles(block2_after['x'], block2_after['y'], block2_after['orientation'], 'orange', 'Block2 After', opacity=1.0, is_block=True)

        # Print values
        plt.text(self.text_x, self.text_y_start - (self.get_vertical_offset(batch=batch)), f'Block 1 Position (After): ({block_before["x"]:.4f}, {block_after["y"]:.4f})', color='black', fontsize=9, transform=plt.gca().transAxes)
        plt.text(self.text_x, self.text_y_start - (self.get_vertical_offset(batch=batch)), f'Block 2 Position (After): ({block2_before["x"]:.4f}, {block2_after["y"]:.4f})', color='black', fontsize=9, transform=plt.gca().transAxes)

        block_distance = self.get_distance_between_blocks(block_before, block_after)
        block2_distance = self.get_distance_between_blocks(block2_before, block2_after)

        self.total_block_distance_traveled[batch].append(block_distance)
        self.total_block2_distance_traveled[batch].append(block2_distance)

        # Print the distance traveled
        plt.text(self.text_x, self.text_y_start - (self.get_vertical_offset(batch=batch)), f'Block 1 Distance Traveled: {block_distance:.4f}', color='black', fontsize=9, transform=plt.gca().transAxes)
        plt.text(self.text_x, self.text_y_start - (self.get_vertical_offset(batch=batch)), f'Block 2 Distance Traveled: {block2_distance:.4f}', color='black', fontsize=9, transform=plt.gca().transAxes)

        # Print the total distance traveled
        plt.text(self.text_x, self.text_y_start - (self.get_vertical_offset(batch=batch)), f'Total Distance Blocks Traveled: {block_distance + block2_distance:.4f}', color='black', fontsize=9, transform=plt.gca().transAxes)

        

   
    def plot_targets(self, obs_after, batch):
        
        target_x = obs_after[batch][0][10]
        target_y = obs_after[batch][0][11]
        target2_x = obs_after[batch][0][13]
        target2_y = obs_after[batch][0][14]

        # Plot the center of the targets (x) 
        plt.scatter([target_x, target2_x], [target_y, target2_y], color=['lightgray', 'lightgray'], marker='x', s=100, label='Targets')

        # Plot the rectangles using plot_rectangles
        self.plot_rectangles(target_x, target_y, orientation=0, color='lightgray', label='Target 1', opacity=0.3)
        self.plot_rectangles(target2_x, target2_y, orientation=0, color='lightgray', label='Target 2', opacity=0.3)

    

    def get_effector_distance(self, starting_effector_actual, effector_actual):

        x1, y1 = starting_effector_actual['x'], starting_effector_actual['y']
        x2, y2 = effector_actual['x'], effector_actual['y']

        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def get_effector_distance_traveled(self, obs_before, obs_after, batch):

        obs_step = 1

        starting_effector_actual = {
            'x': obs_before[batch][obs_step][6], 
            'y': obs_before[batch][obs_step][7],
        }

        effector_actual = {
            'x': obs_after[batch][obs_step][6], 
            'y': obs_after[batch][obs_step][7],
        }

        return self.get_effector_distance(starting_effector_actual, effector_actual)


    def plot_effector(self, obs_before, obs_after, batch):

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

        # Plot the starting effector position
        plt.scatter(starting_effector_actual["x"], starting_effector_actual['y'], color='teal', marker='o', s=100, label='Starting Actual Effector', alpha=0.5)

        plt.scatter(starting_effector_target['x'], starting_effector_target['y'], color='purple', marker='o', s=100, label='Starting Target Effector', alpha=0.5)


        # Plot the final effector position
        plt.scatter(effector_actual['x'], effector_actual['y'], color='green', marker='o', s=100, label='Actual Effector', alpha=.5)

        plt.scatter(effector_target['x'], effector_target['y'], color='red', marker='o', s=100, label='Target Effector', alpha=0.5)

        # Calculate efector distance traveled
        effector_distance = self.get_effector_distance(starting_effector_actual, effector_actual)

        self.total_effector_distance[batch].append(effector_distance)

        # Print final effector values 
        plt.text(self.text_x, self.text_y_start, f'Effector Value (Actual): ({effector_actual["x"]:.4f}, {effector_actual["y"]:.4f})', color='green', fontsize=9, transform=plt.gca().transAxes)

        plt.text(self.text_x, self.text_y_start - self.get_vertical_offset(batch=batch), f'Effector Value (Target): ({effector_target["x"]:.4f}, {effector_target["y"]:.4f})', color='red', fontsize=9, transform=plt.gca().transAxes)

        # Print starting effector values
        plt.text(self.text_x, self.text_y_start - (self.get_vertical_offset(batch=batch)), f'Effector Value (Actual, At Start): ({starting_effector_actual["x"]:.4f}, {starting_effector_actual["y"]:.4f})', color='teal', fontsize=9, transform=plt.gca().transAxes)
        plt.text(self.text_x, self.text_y_start - (self.get_vertical_offset(batch=batch)), f'Effector Value (Target, At Start): ({starting_effector_target["x"]:.4f}, {starting_effector_target["y"]:.4f})', color='purple', fontsize=9, transform=plt.gca().transAxes)

        plt.text(self.text_x, self.text_y_start - (self.get_vertical_offset(batch=batch)), f'Effector Distance Traveled: {effector_distance:.4f}', color='black', fontsize=9, transform=plt.gca().transAxes)

    def translate_at_angle(self, x, y, deg, distance):
        angle_radians = np.radians(deg)

        # Calculate the new coordinates using numpy's cos and sin functions
        new_x = x + distance * np.cos(angle_radians)
        new_y = y + distance * np.sin(angle_radians)
        
        return new_x, new_y
    

    def plot_lie_step(self, step, batch, last_lie_step, obs_before):

        # ONLY IF ACTUALLY DOING ROTATIONS
        if last_lie_step[batch] == step:
            plt.text(self.text_x, self.text_y_start - (self.get_vertical_offset(batch=batch)), 'Lying...', color='red', fontsize=9, transform=plt.gca().transAxes)

            global global_translate, global_translation_angle, global_translation_distance

            if global_translate: 

                

                obs_step = 1
                
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

                block_before['x'], block_before['y'] = self.translate_at_angle(block_before['x'], block_before['y'], global_translation_angle, global_translation_distance)


                self.plot_rectangles(block_before['x'], block_before['y'], block_before['orientation'], 'cyan', 'Block Before Rotated', opacity=0.25, is_block=True)

                block2_before['x'], block2_before['y'] = self.translate_at_angle(block2_before['x'], block2_before['y'], global_translation_angle, global_translation_distance)

                self.plot_rectangles(block2_before['x'], block2_before['y'], block2_before['orientation'], 'olive', 'Block2 Before Rotated', opacity=0.25, is_block=True)

                plt.text(self.text_x, self.text_y_start - (self.get_vertical_offset(batch=batch)), f'(Lied Observation) Block 1 Position: ({block_before["x"]:.4f}, {block_before["y"]:.4f})', color='cyan', fontsize=9, transform=plt.gca().transAxes)

                plt.text(self.text_x, self.text_y_start - (self.get_vertical_offset(batch=batch)), f'(Lied Observation) Block 2 Position: ({block2_before["x"]:.4f}, {block2_before["y"]:.4f})', color='olive', fontsize=9, transform=plt.gca().transAxes)




    def plot_successful(self, obs_after, batch):

        if self.is_successful(obs=obs_after, batch=batch)[0]:
            plt.text(self.text_x, self.text_y_start - (self.get_vertical_offset(batch=batch)), 'Successful!', color='green', fontsize=9, transform=plt.gca().transAxes)
            return True
        else:
            return False




    def plot_env_after_step(self, step, desired_trajectory, obs_before, obs_after, batch, last_lie_step):
        """

        Plots the observations after a step (target, block, and effector postiions). Also calculates distances from blocks to targets. Plots the desired trajectory and the target start and end positions compared to the actual start and end positions. 
        
        Plots everything relative to batch. (So should pass the entire observation across all batches and index inside this function)
        """
        
        
        plt.figure(figsize=(12, 12))
        self.plot_targets(obs_after=obs_after, batch=batch)
        self.plot_blocks(obs_before=obs_before, obs_after=obs_after, batch=batch)
        self.plot_desired_trajectory(desired_trajectory=desired_trajectory, batch=batch)
        self.plot_effector(obs_before=obs_before, obs_after=obs_after, batch=batch)
        self.plot_distance_from_target(batch=batch, obs_after=obs_after)
        if not (self.plot_successful(batch=batch, obs_after=obs_after)):
            self.plot_lie_step(step=step, batch=batch, last_lie_step=last_lie_step, obs_before=obs_before)
        
        # # Adjust legend and labels as needed
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        title = f"Batch {batch}; Step: {step}" 
        plt.title(title)
        plt.xlim(-0.5, 1.0)
        plt.ylim(-1.0, 0.5)
        plt.savefig(f"plots/batch_{batch}/step_{step}.png", bbox_inches='tight')
        plt.close()

    def plot_total_distances(self):

        for batch in [7, 8, 10, 21, 37, 43]:

            effector_distances = self.total_effector_distance[batch]
            block_distances_traveled = self.total_block_distance_traveled[batch]
            block2_distances_traveled = self.total_block2_distance_traveled[batch]

            block_to_target_distances = self.total_block_to_target_distance[batch]
            block_to_target2_distances = self.total_block_to_target2_distance[batch]
            block2_to_target_distances = self.total_block2_to_target_distance[batch]
            block2_to_target2_distances = self.total_block2_to_target2_distance[batch]
            
            plt.figure(figsize=(12, 6))
            plt.plot(effector_distances, marker='o', linestyle='-', color='blue', label='Effector Distance Traveled')
            plt.plot(block_distances_traveled, marker='o', linestyle='-', color='green', label='Block 1 Distance Traveled')
            plt.plot(block2_distances_traveled, marker='o', linestyle='-', color='red', label='Block 2 Distance Traveled')
            plt.plot(block_to_target_distances, marker='o', linestyle='-', color='purple', label='Block 1 to Target 1 Distance')
            plt.plot(block_to_target2_distances, marker='o', linestyle='-', color='orange', label='Block 1 to Target 2 Distance')
            plt.plot(block2_to_target_distances, marker='o', linestyle='-', color='brown', label='Block 2 to Target 1 Distance')
            plt.plot(block2_to_target2_distances, marker='o', linestyle='-', color='pink', label='Block 2 to Target 2 Distance')

            # shade the area where the distance is less than 0.05
            plt.axhspan(0, 0.05, color='green', alpha=0.1)
            plt.xlabel('Time Step')
            plt.ylabel('Effector Distance Traveled')
            plt.title(f'Batch: {batch}; Distances Over Time')
            plt.legend(loc='upper right')
            plt.grid(True)

            media_dir = pathlib.Path(self.output_dir).joinpath('media')
            dist_plot_destination = media_dir.joinpath(f'batch_{batch}/dist_over_time_batch{batch}.png')
            plt.savefig(dist_plot_destination, bbox_inches='tight')
            plt.close()
            

    def get_vertical_offset(self, batch):

        self.inner_step[batch] += 1
        return 0.025 * self.inner_step[batch]
    
    def is_successful(self, obs, batch):

        targets = ["target", "target2"]
        goal_dist_tolerance = 0.05
        block_to_target, block_to_target2, block2_to_target, block2_to_target2 = self.get_blocks_to_target_distance(obs_after=obs, batch=batch)

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
            return True, b0_closest_dist, b0_closest_target, b1_closest_dist, b1_closest_target
        
        return False, b0_closest_dist, b0_closest_target, b1_closest_dist, b1_closest_target


        
    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        last_info = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            step = 1
    
            # start rollout
            obs = env.reset()
            obs_before = obs
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval BlockPushLowdimRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False

            # Initialize trackers for plotting across steps
            num_batches = obs.shape[0]
            blocks_dist = {batch: deque(maxlen=5) for batch in range(num_batches)}
            effector_dist = {batch: deque(maxlen=5) for batch in range(num_batches)}


            self.total_effector_distance = dict()
            self.total_block_distance_traveled = dict()
            self.total_block2_distance_traveled = dict()
            self.total_block_to_target_distance = dict()
            self.total_block_to_target2_distance = dict()
            self.total_block2_to_target_distance = dict()
            self.total_block2_to_target2_distance = dict()

            for batch in [7, 8, 10, 21, 37, 43]:
                self.total_effector_distance[batch] = []
                self.total_block_distance_traveled[batch] = []
                self.total_block2_distance_traveled[batch] = []
                self.total_block_to_target_distance[batch] = []
                self.total_block_to_target2_distance[batch] = []
                self.total_block2_to_target_distance[batch] = []
                self.total_block2_to_target2_distance[batch] = []

            self.text_x, self.text_y_start = 0.05, 0.95
            self.vertical_offset = 0.025  # Vertical space between lines

            
            re_done_per_batch = [False] * num_batches
            closest_dist_per_batch = [(0, 0)] * num_batches
            closest_target_per_batch = [(0,0)] * num_batches
            last_lie_step = [0] * num_batches

            lie_conds = {'blocks_dist_5': 0, 'effector_dist_5': 1}
            

            while not done:
                self.inner_step = [0] * num_batches

                # Create obs dict
                if not self.obs_eef_target:
                    obs[..., 8:10] = 0

                # Create a 2D array for distance rolling lists
                rolling_blocks_dist = np.zeros((num_batches, 5), dtype=np.float32)
                for batch in range(num_batches):
                    # Ensure the deque has exactly 5 elements, fill with zeros if not
                    distances = np.array(blocks_dist[batch])
                    if len(distances) < 5:
                        distances = np.concatenate([np.zeros(5 - len(distances)), distances])
                    rolling_blocks_dist[batch, :] = distances
                
                # Create an array for rolling list of effector distances on the past 5 steps
                rolling_effector_dist = np.zeros((num_batches, 5), dtype=np.float32)
                for batch in range(num_batches):
                    distances = np.array(effector_dist[batch])
                    if len(distances) < 5:
                        distances = np.concatenate([np.zeros(5 - len(distances)), distances])
                    rolling_effector_dist[batch, :] = distances

                # Add step to np_obs_dict
                np_obs_dict = {
                    'obs': obs.astype(np.float32),
                    'blocks_dist': rolling_blocks_dist,
                    'effector_dist': rolling_effector_dist,
                    
                    'step': np.array([step], dtype=np.float32),  # Step should be a 1D array for consistency
                    
                    # Condition lie is based on -- (1) cummulative distance traveled by blocks in past 5 steps (2) cummulative distance traveled by effector in past 5 steps
                    'lie_cond': np.array(int(lie_conds['blocks_dist_5']), dtype=np.float32),
                    'last_lie_step': np.array(last_lie_step), 
                    
                    # Rotate the observed effector positions
                    'rotate' : np.array([0], dtype=np.float32),
                    'rotation_angle': np.array([0], dtype=np.float32),

                    # Translate the observed block position
                    'translate' : np.array([1], dtype=np.float32),
                    'translation_angle': np.array([-180], dtype=np.float32), # note that you have to add 180 to get downwards
                    'translation_distance': np.array([.2], dtype=np.float32),

                    
                }

                if np_obs_dict['translate'][0] == 1:
                    global global_translate, global_translation_angle, global_translation_distance
                    global_translate = True
                    global_translation_angle = np_obs_dict['translation_angle'][0]
                    global_translation_distance = np_obs_dict['translation_distance'][0]
       
                if self.past_action and (past_action is not None):
                    np_obs_dict['past_action'] = past_action[:, -(self.n_obs_steps-1):].astype(np.float32)

                # Device transfer
                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))

                # Run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # Device transfer
                if 'last_lie_step' in action_dict:
                    last_lie_step = action_dict['last_lie_step'].detach().to('cpu').numpy()

                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))

                np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']

                # Step env
                obs, reward, done, info = env.step(action)
                for batch in range(num_batches):

                    # NOTE: Manually replicating the succseful condition here ourselves since after 44 steps (determined by self.max_episode_steps) the env will auto mark them as successful
                    is_done, closest_b1_dist, closest_b1_target, closest_b2_dist, closest_b2_target = self.is_successful(obs=obs, batch=batch)
                    re_done_per_batch[batch] = re_done_per_batch[batch] or is_done # TODO: do a union to avoid overriding in the latter steps
                    closest_dist_per_batch[batch] = (closest_b1_dist, closest_b2_dist)
                    closest_target_per_batch[batch] = (closest_b1_target, closest_b2_target)

                done = np.all(done)
                past_action = action

                # Update pbar
                pbar.update(action.shape[1])

                # Compute distance traveled by blocks and update rolling lists
                for batch in range(num_batches):
                    block_distance_traveled = self.get_total_block_distance_traveled(obs_before=obs_before, obs_after=obs, batch=batch)
                    blocks_dist[batch].append(block_distance_traveled)

                    effector_distance = self.get_effector_distance_traveled(obs_before=obs_before, obs_after=obs, batch=batch)
                    effector_dist[batch].append(effector_distance)

                # Plot trajectories for specific batches of focus 
                for batch in [7, 8, 10, 21, 37, 43]:  # Focus on specific batches for plotting
                    self.plot_env_after_step(step=step, desired_trajectory=action, obs_before=obs_before, obs_after=obs, batch=batch, last_lie_step=last_lie_step)

                obs_before = obs
                step += 1

            done_batches = []
            not_done_batches = []
            for i in range(len(re_done_per_batch)):
                if (6 <= i and i <= 55): 
                    if re_done_per_batch[i]:
                        done_batches.append(i)
                    else:
                        not_done_batches.append(i)

            success_larger_threshold = 0 
            ss_path = pathlib.Path(self.output_dir).joinpath('summary.txt')
            with open(ss_path, 'w') as file:
                file.write(f'Successful Batches: {done_batches}\n')
                file.write(f'Unsuccessful Batches: {not_done_batches}\n')
                file.write(f"Success Rate: {(len(done_batches) / (len(re_done_per_batch) - 6)) * 100}\n")

                # success is actually only measured from batch 6-55

                file.write(f"Distance of Block 1, Block 2 to Closest Targest\n")
                for i in range(len(closest_dist_per_batch)):
                    (closest_dist_b1, closest_dist_b2) = closest_dist_per_batch[i]
                    (closest_target_b1, closest_target_b2) = closest_target_per_batch[i]
                    
                    # Renaming targets
                    target_rename = {"target": "t1", "target2": "t2"}
                    closest_target_b1 = target_rename[closest_target_b1]
                    closest_target_b2 = target_rename[closest_target_b2]
                    
                    last_lie = last_lie_step[i]
                    
                    # Formatting with stars based on the distances
                    if closest_dist_b1 < 0.06 and closest_dist_b2 < 0.06:
                        file.write(f"*+ Batch {i}: {closest_dist_b1:7.3f} to {closest_target_b1}, {closest_dist_b2:7.3f} to {closest_target_b2}  ")
                        success_larger_threshold += 1
                    elif closest_dist_b1 < 0.06:
                        file.write(f"*  Batch {i}: {closest_dist_b1:7.3f} to {closest_target_b1}, {closest_dist_b2:7.3f} to {closest_target_b2}  ")
                    elif closest_dist_b2 < 0.06:
                        file.write(f"+  Batch {i}: {closest_dist_b1:7.3f} to {closest_target_b1}, {closest_dist_b2:7.3f} to {closest_target_b2}  ")
                    else:
                        file.write(f"   Batch {i}: {closest_dist_b1:7.3f} to {closest_target_b1}, {closest_dist_b2:7.3f} to {closest_target_b2}  ")
                    
                    # Aligning the last lie step information
                    file.write(f"Last Lie Step: {last_lie}\n")



                file.write(f"Success Rate (Larger Threshold): {(success_larger_threshold / len(re_done_per_batch)) * 100}\n")
                file.write(f"Key: * Block 1 Close to Target, + Block 2 Close to Target\n")
            pbar.close()

            self.plot_total_distances()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            last_info[this_global_slice] = [dict((k,v[-1]) for k, v in x.items()) for x in info][this_local_slice]

        # log
        total_rewards = collections.defaultdict(list)
        total_p1 = collections.defaultdict(list)
        total_p2 = collections.defaultdict(list)
        prefix_event_counts = collections.defaultdict(lambda :collections.defaultdict(lambda : 0))
        prefix_counts = collections.defaultdict(lambda : 0)

        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            this_rewards = all_rewards[i]
            total_reward = np.unique(this_rewards).sum() # (0, 0.49, 0.51)
            p1 = total_reward > 0.4
            p2 = total_reward > 0.9

            total_rewards[prefix].append(total_reward)
            total_p1[prefix].append(p1)
            total_p2[prefix].append(p2)
            log_data[prefix+f'sim_max_reward_{seed}'] = total_reward

            # aggregate event counts
            prefix_counts[prefix] += 1
            for key, value in last_info[i].items():
                delta_count = 1 if value > 0 else 0
                prefix_event_counts[prefix][key] += delta_count

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in total_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value
        for prefix, value in total_p1.items():
            name = prefix+'p1'
            value = np.mean(value)
            log_data[name] = value
        for prefix, value in total_p2.items():
            name = prefix+'p2'
            value = np.mean(value)
            log_data[name] = value
        
        # summarize probabilities
        for prefix, events in prefix_event_counts.items():
            prefix_count = prefix_counts[prefix]
            for event, count in events.items():
                prob = count / prefix_count
                key = prefix + event
                log_data[key] = prob

        return log_data
    

