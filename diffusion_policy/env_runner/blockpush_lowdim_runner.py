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

import diffusion_policy.policy.utils.plotting_utils as pu



global global_translate, global_translation_angle, global_translation_distance

global_translate = False

global global_rotate, global_rotation_angle, global_rotation_distance
global_rotate = False

global global_ideal_trajectory, ideal_trajectories
global_ideal_trajectory = False

global custom_lies
ROTATIONS_PER_BATCH = [[], [], [], [], [], [], [], [], [], []]   # per batch, list out all the custom (step, deg, distance) lies that we want to do. if we keep it consistent here and in policy then any time we get that step = last_lie_step, we can just increment and know which deg and distance to rotate by.  


global DISPLAY_BATCHES
DISPLAY_BATCHES = [6, 7, 8, 9]

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

        #

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
            if i in DISPLAY_BATCHES:
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

            if (i+n_train) in DISPLAY_BATCHES:
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

    def get_distance_between_blocks(self, block_before, block_after):
        x1, y1 = block_before['x'], block_before['y']
        x2, y2 = block_after['x'], block_after['y']

        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

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

            # self.total_effector_distance = dict()
            # self.total_block_distance_traveled = dict()
            # self.total_block2_distance_traveled = dict()
            # self.total_block_to_target_distance = dict()
            # self.total_block_to_target2_distance = dict()
            # self.total_block2_to_target_distance = dict()
            # self.total_block2_to_target2_distance = dict()

            # for batch in DISPLAY_BATCHES:
            #     self.total_effector_distance[batch] = []
            #     self.total_block_distance_traveled[batch] = []
            #     self.total_block2_distance_traveled[batch] = []
                
            #     self.total_block_to_target_distance[batch] = []
            #     self.total_block_to_target2_distance[batch] = []
            #     self.total_block2_to_target_distance[batch] = []
            #     self.total_block2_to_target2_distance[batch] = []

            self.text_x, self.text_y_start = 0.05, 0.95
            self.vertical_offset = 0.025  # Vertical space between lines
            
            re_done_per_batch = [False] * num_batches
            closest_dist_per_batch = [(0, 0)] * num_batches
            closest_target_per_batch = [(0,0)] * num_batches
            last_lie_step = [0] * num_batches

            lie_conds = {'blocks_dist_5': 0, 'effector_dist_5': 1, 'first_step': 2, 'custom':3}
            

            while not done:

                pu.init_plots(step)



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
                    'lie_cond': np.array(int(lie_conds['custom']), dtype=np.float32),
                    'last_lie_step': np.array(last_lie_step), 
                    
                    # Rotate the observed effector positions
                    'rotate' : np.array([0], dtype=np.float32),
                    'rotation_angle': np.array([0], dtype=np.float32),
                    'rotation_distance': np.array([0], dtype=np.float32),

                    # Translate the observed block position
                    'translate' : np.array([0], dtype=np.float32),
                    'translation_angle': np.array([0], dtype=np.float32), # note that you have to add 180 to get downwards
                    'translation_distance': np.array([0], dtype=np.float32),

                    'custom': np.array([0], dtype=np.float32), # rotate custom amounts at custom steps

                    'vector_to_target': np.array([0], dtype=np.float32),
                    
                }

                # Setting global variables so I can use them in plot_lie_step(). Would be much better to just have a dictionary of my lie configuration. 

                if np_obs_dict['translate'][0] == 1:
                    global global_translate, global_translation_angle, global_translation_distance
                    global_translate = True
                    global_translation_angle = np_obs_dict['translation_angle'][0]
                    global_translation_distance = np_obs_dict['translation_distance'][0]

                if np_obs_dict['rotate'][0] == 1:
                    global global_rotate, global_rotation_angle, global_rotation_distance
                    global_rotate = True
                    global_rotation_angle = np_obs_dict['rotation_angle'][0]
                    global_rotation_distance = np_obs_dict['rotation_distance'][0]

                if np_obs_dict['custom'][0] == 1:
                    global global_custom
                    global_custom = True

                if np_obs_dict['vector_to_target'][0] == 1:
                    global global_ideal_trajectory, ideal_trajectories
                    global_ideal_trajectory = True
                    

       
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

                # if 'ideal_vectors' in action_dict:
                #     ideal_trajectories = action_dict['ideal_vectors']

                

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

                    pu.plot_env_after_step(step=step, desired_trajectory=action, obs_before=obs_before, obs_after=obs, batch=batch, last_lie_step=last_lie_step)

                    blocks_distance_traveled = pu.TOTAL_BLOCK_DISTANCE_TRAVELED[batch][-1] + pu.TOTAL_BLOCK2_DISTANCE_TRAVELED[batch][-1]
                    blocks_dist[batch].append(blocks_distance_traveled)
                    
                    # block_distance_traveled = self.get_total_block_distance_traveled(obs_before=obs_before, obs_after=obs, batch=batch)
                    # blocks_dist[batch].append(block_distance_traveled)

                    effector_distance = pu.TOTAL_EFFECTOR_DISTANCE_TRAVELED[batch][-1]
                    effector_dist[batch].append(effector_distance)

                    # effector_distance = self.get_effector_distance_traveled(obs_before=obs_before, obs_after=obs, batch=batch)
                    # effector_dist[batch].append(effector_distance)

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

            first_seen_block = "block_A (0: right)"
            first_seen_step = last_info[i]['REACH_0']

            # aggregate event counts
            prefix_counts[prefix] += 1
            for key, value in last_info[i].items():
                delta_count = 1 if value > 0 else 0
                prefix_event_counts[prefix][key] += delta_count

                # track intent
                log_data[prefix+f'batch_{i}_'+key] = log_data[prefix + f'batch_{i}_' + key] = float(int((value + 7) // 8)) if value != -1 else -1
                
                if (key == "REACH_1") and value != -1 and value < first_seen_step:
                    first_seen_block = "block_B (1: left)"
                    first_seen_step = value

            log_data[prefix+f'batch_{i}_first_seen_block'] = str(first_seen_block)

            # I want to compare REACH_0 and REACH_1 values and log the first block that we touched


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
    

