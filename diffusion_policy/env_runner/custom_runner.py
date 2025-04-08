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

# Define paths
# create path relative to this file

# Note: I made a copy of the zarr file in the env_runner so I could make changes without modifiying the original. When it is working, I should make this relative to data/. 
# zarr_path = os.path.join(os.path.dirname(__file__), "multimodal_push_seed_abs.zarr")
# output_dir = "output_videos"

# # Load Zarr dataset
# action_zarr = zarr.open(zarr_path, mode='a')

# episode_ends = action_zarr['meta']['episode_ends']

# # Extract action data (each entry is (x, y))
# actions = action_zarr['data']['action']
# obs = action_zarr['data']['obs'] # NOTE: Observation values actually do not really matter other than extracting starting positions because we are rolling out and trying to figure out if the actions result in a successful rollout.
# batch_size = 1 # Number of batches from Zarr (since )

# # TODO: Change this to take in a demo num (lowkey I can just write code that says if demo num is 0 then start at )
# # Set fixed number of steps
# demo_num = 0 # This is the same as demo_num

# if demo_num == 0: 
#     start_timestep = 0 
# else:
#     start_timestep = episode_ends[demo_num - 1] 

# end_timestep = episode_ends[demo_num] - 1
# num_steps = end_timestep - start_timestep

# Extract the current demonstrations
# # NOTE: For now, I am copying the entire obs and actions arrays. This is not ideal, but I am doing it to avoid modifying the original zarr inputs for now. 
# current_demo ={
#     # Reshape to be per step, per batch, per obs, two observations (x,y)
#     'obs': (obs[start_timestep:end_timestep].copy()),
#     # Reshape to for this one demo, per step, per batch, per action, 16 fields (104, 1, 1, 2)
#     'actions': (actions[start_timestep:end_timestep].copy()).reshape(num_steps, 1, 1, 2),
#     'demo_num': demo_num
# } 



# # Set up environment parameters (default, copied from block_push_multimodal_runner.py)
# task_fps = 10
# fps = 5
# crf = 22
# steps_per_render = max(10 // fps, 1)  # Control rendering rate
# seed = 42
# abs_action = True
# max_steps = 104  # Set max episode steps to 104

# # Create output directory
# pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

# def env_fn():
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
#             file_path=f"{output_dir}/zarr_action_sim.mp4",  # Save video here
#             steps_per_render=steps_per_render
#         ),
#         n_obs_steps=1,  # Single observation step
#         n_action_steps=1,  # One action per step
#         max_episode_steps=max_steps
#     )

# # Delete the existing videos in output_videos
# # os.remove(f"{output_dir}")
# os.system(f"rm -r {output_dir}")
# os.mkdir(output_dir)

# # Initialize environment
# env = env_fn()

# init_obs = current_demo['obs'][0]

# # NOTE: Temporary solution to pass the initial observation. Too many function signatures to change. 

# INIT_OBS_FILE = os.path.join(os.path.dirname(__file__), "../env/block_pushing/init_obs.json")

# # We should start by deleting the file
# if os.path.exists(INIT_OBS_FILE):
#     os.remove(INIT_OBS_FILE)

# def save_init_obs(init_obs):
#     # Convert to a list so it can be saved in JSON format
#     with open(INIT_OBS_FILE, "w") as f:
#         json.dump(init_obs.tolist(), f)

#     print(f"Saved initial observation to {INIT_OBS_FILE}")

# save_init_obs(init_obs)


# # Want to hard wire the environmnet 
# obs = env.reset()
# # TODO: pass init obs all the way down. 

# # Run the environment using reshaped actions
# for step in range(num_steps):
#     batch = 0 
#     curr_action = current_demo['actions'][step][batch]  # Shape: (batch_size, 1, 2)
#     curr_obs = current_demo['obs'][step][batch]  # Shape: (batch_size, 1, 16)

#     print(f"Step {step}: Action: {curr_action}")

#     # Just checking that they I spliced the zarr correctly

#     assert actions[step][0] == curr_action[0][0]
#     assert actions[step][1] == curr_action[0][1]

#     obs, reward, done, info = env.step(curr_action)  # Pass action to environment
    
#     if done:
#         # Right now its ending because this is the end of the number of steps in the demo
#         print(f"Done at step {step}")

#         if done and step == num_steps - 1 and reward != 1:
#             print("Failed to reach goal")

#         break

# # Stop recording and save video
# env.env.video_recoder.stop()
# print(f"Video saved at: {output_dir}/zarr_action_sim.mp4")



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


def env_fn(video_name, max_steps):
    """Creates the environment with video recording"""
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
            file_path=f"{output_dir}/{video_name}.mp4", 
            steps_per_render=steps_per_render
        ),
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

    print(f"Saved initial observation to {INIT_OBS_FILE}")


def rollout_demo(init_obs, num_steps, action_dict, output_dir="sim_videos",video_name="zarr_action_sim"):
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

    # Initialize environment
    env = env_fn(video_name, num_steps)

    save_init_obs(init_obs)
    demo_num = hash(video_name)
    # pu.setup_full_trajectory_plot(init_obs, demo_num) # Just hashed because I don't want to keep making a new number. 

    # Reset environment
    obs = env.reset()

    final_status = None

    # TODO: new

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

            

    while not done:
        
        batch = 0
        curr_action = current_demo['action'][step][batch]  # Shape: (1, 1, 2)
        # curr_obs = current_demo['obs'][step][batch]  # Shape: (1, obs_dim) Never needed since we're doing rollout.

        # Sanity check that the action aligns with original input
        assert action_dict[step][0] == curr_action[0][0]
        assert action_dict[step][1] == curr_action[0][1]

        if step == 3 and video_name== "artificial_trajectory_3+659_f": 
            print("break!")

        # Take a step in the environment
        obs, reward, done, info = env.step(curr_action)

        # Calculate speed from obs[x 0, y 1, orientation 2] and the prior obs 
        distance = math.sqrt((obs[0][0] - prior_obs[0])**2 + (obs[0][1] - prior_obs[1])**2)
        

        with open(log_path, 'a') as f:
            f.write(f"Step {step}:\n")
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
    env.env.video_recoder.stop()
    print(f"Video saved at: {output_dir}/{video_name}.mp4")


    return final_status