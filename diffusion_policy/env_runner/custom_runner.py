"""
Rolls out a custom demo. The actions are loaded from a Zarr file and the environment is rendered as a video.
"""

import numpy as np
import zarr
import os
import pathlib
from diffusion_policy.env.block_pushing.block_pushing_multimodal import BlockPushMultimodal
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from gym.wrappers import FlattenObservation
import json
import math

from data.block_pushing.block_pushing import plot_training_demos as pu

# print pwd
print(os.getcwd())

# File to store initial observations
INIT_OBS_FILE = os.path.join(os.path.dirname(__file__), "../env/block_pushing/init_obs.json")

# Set up environment parameters (default, copied from block_push_multimodal_runner.py)
task_fps = 10
fps = 5
crf = 22
steps_per_render = max(10 // fps, 1)  # Control rendering rate
seed = 42
abs_action = True
output_dir = "sim_videos"


# def env_fn(video_name, max_steps):
#     """Creates the environment with video recording"""
#     return MultiStepWrapper(
#         VideoRecordingWrapper(
#             FlattenObservation(
#                 BlockPushMultimodal(
#                     control_frequency=task_fps,
#                     shared_memory=False,
#                     seed=seed,
#                     abs_action=abs_action
#                 )
#             ),
#             video_recoder=VideoRecorder.create_h264(
#                 fps=fps,
#                 codec='h264',
#                 input_pix_fmt='rgb24',
#                 crf=crf,
#                 thread_type='FRAME',
#                 thread_count=1
#             ),
#             file_path=f"{output_dir}/{video_name}.mp4", 
#             steps_per_render=steps_per_render
#         ),
#         n_obs_steps=1,
#         n_action_steps=1,
#         max_episode_steps=max_steps
#     )

def env_fn(video_name, max_steps, record_video=False):
    """Creates the environment, optionally with video recording."""
    base_env = FlattenObservation(
        BlockPushMultimodal(
            control_frequency=task_fps,
            shared_memory=False,
            seed=seed,
            abs_action=abs_action
        )
    )

    if record_video:
        base_env = VideoRecordingWrapper(
            base_env,
            video_recoder=VideoRecorder.create_h264(
                fps=fps,
                codec='h264',
                input_pix_fmt='rgb24',
                crf=crf,
                thread_type='FRAME',
                thread_count=1
            ),
            file_path=f"{output_dir}/{video_name}.mp4",
            steps_per_render=steps_per_render
        )

    return MultiStepWrapper(
        base_env,
        n_obs_steps=1,
        n_action_steps=1,
        max_episode_steps=max_steps
    )


def save_init_obs(init_obs):
    """Saves initial observation to a JSON file."""

    # TODO: stop deleting the old videos and 
    if os.path.exists(INIT_OBS_FILE):
        os.remove(INIT_OBS_FILE)

    with open(INIT_OBS_FILE, "w") as f:
        json.dump(init_obs.tolist(), f)

def rollout_demo(init_obs, num_steps, action_dict, output_dir="sim_videos",video_name="zarr_action_sim", plot_steps=False, record_video=False):
    """
    Runs the block pushing environment using given observations and actions.

    Parameters:
        init_obs (np.array): Preloaded observation at step=0 shaped as (obs_dim).
        action_dict (np.array): Preloaded actions, shaped as (steps, batches, 1, action_dim).
        demo_num (int): Demonstration identifier.
        output_dir (str): Directory to save output.
    """
    # Prepare demonstration data
    
    current_demo = {
        'init_obs': init_obs,  # Copy to avoid modifying original data
        'action': action_dict.copy().reshape(num_steps, 1, 1, 2),  # Reshape to (steps, batch, 1, 2)
    }

    rollout_obs = []

    # Initialize environment
    env = env_fn(video_name, num_steps, record_video=record_video)

    save_init_obs(init_obs)
    demo_num = hash(video_name)
    # pu.setup_full_trajectory_plot(init_obs, demo_num) # Just hashed because I don't want to keep making a new number. 

    # Reset environment
    obs = env.reset()

    final_status = None

    # Run the environment using preloaded action
    done = False
    step = 0
    # for step in range(num_steps):

    log_path = f"{output_dir}/{video_name}.txt"
    with open(log_path, 'w') as f:
        f.write(f"Rollout for {video_name}\n")

    prior_obs = init_obs

    with open(log_path, 'a') as f:
        f.write(f"Initalization:\n")
        f.write(f" Block 1: {prior_obs[0]:.4f}, {prior_obs[1]:.4f}, {prior_obs[2]:.4f}\n")
        f.write(f" Block 2: {prior_obs[3]:.4f}, {prior_obs[4]:.4f}, {prior_obs[5]:.4f}\n")
        f.write(f" Effector: {prior_obs[6]:.4f}, {prior_obs[7]:.4f}\n")
        f.write(f" Target Effector: {prior_obs[8]:.4f}, {prior_obs[9]:.4f}\n")
        f.write(f" Target 1: {prior_obs[10]:.4f}, {prior_obs[11]:.4f}, {prior_obs[12]:.4f}\n")
        f.write(f" Target 2: {prior_obs[13]:.4f}, {prior_obs[14]:.4f}, {prior_obs[15]:.4f}\n\n")

    if plot_steps:
        pu.setup_full_trajectory_plot(obs_before=init_obs, demo_num=demo_num)
        

    # This should be the same length as the action_dict
    # while not done:

    for i in range(num_steps):
       
        
        batch = 0
        curr_action = current_demo['action'][step][batch]  # Shape: (1, 1, 2)
        # curr_obs = current_demo['obs'][step][batch]  # Shape: (1, obs_dim) Never needed since we're doing rollout.

        # Take a step in the environment
        obs, reward, done, info = env.step(curr_action)
        
        rollout_obs.append(obs[0])  # Append the first batch of observations
        

        if plot_steps:
            if step == 0:
                curr_obs = init_obs
            else:
                curr_obs = obs[0]

            pu.setup_full_trajectory_plot(obs_before=curr_obs, demo_num=demo_num)

            info_text = [f"REACH_0: {info['REACH_0']}, REACH_1: {info['REACH_1']}",
                         f"TARGET_0_0: {info['TARGET_0_0']}, TARGET_0_1: {info['TARGET_0_1']}",
                         f"TARGET_1_0: {info['TARGET_1_0']}, TARGET_1_1: {info['TARGET_1_1']}"]
            for text in info_text:
                pu.custom_label(demo_num=demo_num, custom_text=text, color="black")

            pu.plot_effector_actions(action=curr_action[0], run_step=step, demo_num=demo_num, color='gradient', start_timestep=0)
            pu.finalize_full_trajectory_plot(obs=obs[0], demo_num=demo_num, coloring='gradient', custom_file_name=f"step_plot_{step}")

        # Calculate speed from obs[x 0, y 1, orientation 2] and the prior obs 
        distance = math.sqrt((obs[0][0] - prior_obs[0])**2 + (obs[0][1] - prior_obs[1])**2)

        with open(log_path, 'a') as f:
            f.write(f"Step {step+1}:\n")
            f.write(f"  Action: {curr_action[0][0]:.4f}, {curr_action[0][1]:.4f}\n")
            f.write(f"  Block 1: {obs[0][0]:.4f}, {obs[0][1]:.4f}, {obs[0][2]:.4f}\n")
            f.write(f"  Block 2: {obs[0][3]:.4f}, {obs[0][4]:.4f}, {obs[0][5]:.4f}\n")
            f.write(f"  Effector: {obs[0][6]:.4f}, {obs[0][7]:.4f}\n")
            f.write(f"  Target Effector: {obs[0][8]:.4f}, {obs[0][9]:.4f}\n")
            f.write(f"  Target 1: {obs[0][10]:.4f}, {obs[0][11]:.4f}, {obs[0][12]:.4f}\n")
            f.write(f"  Target 2: {obs[0][13]:.4f}, {obs[0][14]:.4f}, {obs[0][15]:.4f}\n")
            f.write(f"  Distance: {distance:.4f}\n")
            f.write(f"  REACH_0: {info['REACH_0']}, REACH_1: {info['REACH_1']}, TARGET_0_0: {info['TARGET_0_0']}, TARGET_0_1: {info['TARGET_0_1']}, TARGET_1_0: {info['TARGET_1_0']}, TARGET_1_1: {info['TARGET_1_1']}\n\n")

        # Log observation and actions per step in the json

        # pu.plot_effector_actions(action=curr_action[0], run_step=step, demo_num=demo_num, color='gradient', start_timestep=0)
        
        # Lowkey we should crop the action dictionary to be the length of the rollout_obs...
        if done:
            final_status = (obs, reward, done, info)
            if reward <= 0.0:
                print(f"Failed to reach goal at step {step} with reward {reward}")
            else:
                print(f"Reached goal at step {step} with reward {reward}")
            break

        step += 1
        prior_obs = obs[0]

    
    # pu.finalize_full_trajectory_plot(obs=obs[0], demo_num=demo_num, coloring='gradient', custom_file_name="actual_artificial")
    # Stop recording and save video

    if record_video:
        env.env.video_recoder.stop()
    print(f"Video saved at: {output_dir}/{video_name}.mp4")

    # Make rollout_obs same type as action_dict
    rollout_obs = np.array(rollout_obs)

    return final_status, rollout_obs

# def rollout_single_demo(init_obs, action_dict, num_jumps, output_dir="sim_videos", video_name="zarr_jump_rollout"):

#     current_demo = {

#         'init_obs': init_obs,  # Copy to avoid modifying original data
#         'action': action_dict.copy().reshape(num_jumps, 1, 1, 2),  # Reshape to (steps, batch, 1, 2)

#     }

#     env = env_fn(video_name, num_jumps)
#     save_init_obs(init_obs)
#     obs = env.reset()

#     final_status = None

