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
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
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
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

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


        text_x, text_y_start = 0.05, 0.95
        vertical_offset = 0.05  # Vertical space between lines
        
        plt.text(text_x, text_y_start - (vertical_offset*2), f'Desired Trajectory End Point: ({x_coords[-1]:.4f}, {y_coords[-1]:.4f})', color='red', fontsize=9, transform=plt.gca().transAxes)


    def plot_rectangles(self, x, y, orientation, color, label, goal_dist_tolerance=0.05, opacity=1.0):
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

    def get_block_distance(self, block_before, block_after):
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
        self.plot_rectangles(block_before['x'], block_before['y'], block_before['orientation'], 'blue', 'Block Before', opacity=0.5)
        self.plot_rectangles(block2_before['x'], block2_before['y'], block2_before['orientation'], 'orange', 'Block2 Before', opacity=0.5)
        
        # Plot blocks after
        self.plot_rectangles(block_after['x'], block_after['y'], block_after['orientation'], 'blue', 'Block After', opacity=1.0)
        self.plot_rectangles(block2_after['x'], block2_after['y'], block2_after['orientation'], 'orange', 'Block2 After', opacity=1.0)

        # Print values
        text_x, text_y_start = 0.05, 0.95
        vertical_offset = 0.05  # Vertical space between lines
        
        plt.text(text_x, text_y_start - (vertical_offset*3), f'Block 1 Position (After): ({block_before["x"]:.4f}, {block_after["y"]:.4f})', color='black', fontsize=9, transform=plt.gca().transAxes)
        plt.text(text_x, text_y_start - (vertical_offset*4), f'Block 2 Position (After): ({block2_before["x"]:.4f}, {block2_after["y"]:.4f})', color='black', fontsize=9, transform=plt.gca().transAxes)

        block_distance = self.get_block_distance(block_before, block_after)
        block2_distance = self.get_block_distance(block2_before, block2_after)

        # Print the distance traveled
        plt.text(text_x, text_y_start - (vertical_offset*9), f'Block 1 Distance Traveled: {block_distance:.4f}', color='black', fontsize=9, transform=plt.gca().transAxes)
        plt.text(text_x, text_y_start - (vertical_offset*10), f'Block 2 Distance Traveled: {block2_distance:.4f}', color='black', fontsize=9, transform=plt.gca().transAxes)

        # Print the distance traveled
        

   
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

        text_x, text_y_start = 0.05, 0.95
        vertical_offset = 0.05  # Vertical space between lines

        # Print final effector values 
        plt.text(text_x, text_y_start, f'Effector Value (Actual): ({effector_actual["x"]:.4f}, {effector_actual["y"]:.4f})', color='green', fontsize=9, transform=plt.gca().transAxes)

        plt.text(text_x, text_y_start - vertical_offset, f'Effector Value (Target): ({effector_target["x"]:.4f}, {effector_target["y"]:.4f})', color='red', fontsize=9, transform=plt.gca().transAxes)

        # Print starting effector values
        plt.text(text_x, text_y_start - (vertical_offset*6), f'Effector Value (Actual, At Start): ({starting_effector_actual["x"]:.4f}, {starting_effector_actual["y"]:.4f})', color='teal', fontsize=9, transform=plt.gca().transAxes)
        plt.text(text_x, text_y_start - (vertical_offset*7), f'Effector Value (Target, At Start): ({starting_effector_target["x"]:.4f}, {starting_effector_target["y"]:.4f})', color='purple', fontsize=9, transform=plt.gca().transAxes)

        plt.text(text_x, text_y_start - (vertical_offset*8), f'Effector Distance Traveled: {effector_distance:.4f}', color='black', fontsize=9, transform=plt.gca().transAxes)



        

    def plot_env_after_step(self, step, desired_trajectory, obs_before, obs_after, batch):

        
        plt.figure(figsize=(12, 12))
        self.plot_targets(obs_after=obs_after, batch=batch)
        self.plot_blocks(obs_before=obs_before, obs_after=obs_after, batch=batch)
        self.plot_desired_trajectory(desired_trajectory=desired_trajectory, batch=batch)
        self.plot_effector(obs_before=obs_before, obs_after=obs_after, batch=batch)

        # # Adjust legend and labels as needed
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        title = f"Batch {batch}; Step: {step}" 
        plt.title(title)
        plt.xlim(-0.5, 1.0)
        plt.ylim(-1.0, 0.5)
        plt.savefig(f"plots/batch_{batch}/step_{step}.png", bbox_inches='tight')
    


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
            while not done:
                # create obs dict
                if not self.obs_eef_target:
                    obs[...,8:10] = 0
                np_obs_dict = {
                    'obs': obs.astype(np.float32)
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))
            
            
                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'] # this is the

                # step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])

                self.plot_env_after_step(step=step, desired_trajectory=action, obs_before=obs_before, obs_after=obs, batch=0)

                self.plot_env_after_step(step=step, desired_trajectory=action, obs_before=obs_before, obs_after=obs, batch=5)

                self.plot_env_after_step(step=step, desired_trajectory=action, obs_before=obs_before, obs_after=obs, batch=10)
                obs_before = obs
                step += 1
            pbar.close()
       


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
