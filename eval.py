"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')

def main(checkpoint, output_dir, device):


    def clear_directory(directory):
        import os
        import shutil

        if os.path.exists(directory):
            shutil.rmtree(directory)  # Remove the directory and all its contents
        os.makedirs(directory)  # Recreate the directory

    def convert_step_images_to_gif():

        # if output.mp4 exists, delete
        if os.path.exists('diff_batches.mp4'):
            os.remove('diff_batches.mp4')

        if os.path.exists('diff_traj.mp4'):
            os.remove('diff_traj.mp4')

        # turn the image steps into a video
        os.system("ffmpeg -framerate 3 -i plots/diff_batches/step%d.png -c:v libx264 -r 30 diff_batches.mp4")

        os.system("ffmpeg -framerate 3 -i plots/diff_traj/step%d.png -c:v libx264 -r 30 diff_traj.mp4")

    clear_directory('plots/diff_batches')
    clear_directory('plots/diff_traj')
    clear_directory('plots/rotated')


    # Commented out -- always overrides. easier to debug
    # if os.path.exists(output_dir):
    #     click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

    convert_step_images_to_gif()
    
if __name__ == '__main__':
    main()
