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

# Define paths
# create path relative to this file

# Note: I made a copy of the zarr file in the env_runner so I could make changes without modifiying the original. When it is working, I should make this relative to data/. 
zarr_path = os.path.join(os.path.dirname(__file__), "multimodal_push_seed_abs.zarr")
output_dir = "output_videos"

# Load Zarr dataset
action_zarr = zarr.open(zarr_path, mode='a')

episode_ends = action_zarr['meta']['episode_ends']

# Extract action data (each entry is (x, y))
actions = action_zarr['data']['action']
obs = action_zarr['data']['obs'] # NOTE: Observation values actually do not really matter other than extracting starting positions because we are rolling out and trying to figure out if the actions result in a successful rollout.
batch_size = 1 # Number of batches from Zarr (since )

# TODO: Change this to take in a demo num (lowkey I can just write code that says if demo num is 0 then start at )
# Set fixed number of steps
demo_num = 0 # This is the same as demo_num

if demo_num == 0: 
    start_timestep = 0 
else:
    start_timestep = episode_ends[demo_num - 1] 

end_timestep = episode_ends[demo_num] - 1
num_steps = end_timestep - start_timestep

# Extract the current demonstrations
# NOTE: For now, I am copying the entire obs and actions arrays. This is not ideal, but I am doing it to avoid modifying the original zarr inputs for now. 
current_demo ={
    # Reshape to be per step, per batch, per obs, two observations (x,y)
    'obs': (obs[start_timestep:end_timestep].copy()),
    # Reshape to for this one demo, per step, per batch, per action, 16 fields (104, 1, 1, 2)
    'actions': (actions[start_timestep:end_timestep].copy()).reshape(num_steps, 1, 1, 2),
    'demo_num': demo_num
} 



# Set up environment parameters (default, copied from block_push_multimodal_runner.py)
task_fps = 10
fps = 5
crf = 22
steps_per_render = max(10 // fps, 1)  # Control rendering rate
seed = 42
abs_action = True
max_steps = 104  # Set max episode steps to 104

# Create output directory
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

def env_fn():
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
            file_path=f"{output_dir}/zarr_action_sim.mp4",  # Save video here
            steps_per_render=steps_per_render
        ),
        n_obs_steps=1,  # Single observation step
        n_action_steps=1,  # One action per step
        max_episode_steps=max_steps
    )

# Delete the existing videos in output_videos
# os.remove(f"{output_dir}")
os.system(f"rm -r {output_dir}")
os.mkdir(output_dir)

# Initialize environment
env = env_fn()

init_obs = current_demo['obs'][0]

# NOTE: Temporary solution to pass the initial observation. Too many function signatures to change. 

INIT_OBS_FILE = os.path.join(os.path.dirname(__file__), "../env/block_pushing/init_obs.json")

# We should start by deleting the file
if os.path.exists(INIT_OBS_FILE):
    os.remove(INIT_OBS_FILE)

def save_init_obs(init_obs):
    # Convert to a list so it can be saved in JSON format
    with open(INIT_OBS_FILE, "w") as f:
        json.dump(init_obs.tolist(), f)

    print(f"Saved initial observation to {INIT_OBS_FILE}")

save_init_obs(init_obs)


# Want to hard wire the environmnet 
obs = env.reset()
# TODO: pass init obs all the way down. 

# Run the environment using reshaped actions
for step in range(num_steps):
    batch = 0 
    curr_action = current_demo['actions'][step][batch]  # Shape: (batch_size, 1, 2)
    curr_obs = current_demo['obs'][step][batch]  # Shape: (batch_size, 1, 16)

    print(f"Step {step}: Action: {curr_action}")

    # Just checking that they I spliced the zarr correctly

    assert actions[step][0] == curr_action[0][0]
    assert actions[step][1] == curr_action[0][1]

    obs, reward, done, info = env.step(curr_action)  # Pass action to environment
    
    if done:
        # Right now its ending because this is the end of the number of steps in the demo
        print(f"Done at step {step}")

        if done and step == num_steps - 1 and reward != 1:
            print("Failed to reach goal")

        break

# Stop recording and save video
env.env.video_recoder.stop()
print(f"Video saved at: {output_dir}/zarr_action_sim.mp4")

# TODO: Show the observations (the blocks aren't moving??) and also get the stats of whether this was succesful
# lol i didn't actually need the observations because the point is im supposed to roll them out. ad use info to figure out 

# THE PROBLEM WAS THAT MY BLOCKS ARE NOT IN THE SAME STARTING POSITION... I need to extrac this from the first observation :)