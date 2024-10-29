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

global DISPLAY_BATCHES
DISPLAY_BATCHES = [6, 7, 8, 9]

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')

@click.option('--n_test', default=None, type=int, help='Override number of test environments')

def main(checkpoint, output_dir, device, n_test=None):

    def clear_directory(batches):
        print("\nRemember to sync your DISPLAY_BATCHES values")
        import os
        import shutil

        if os.path.exists('plots'):
            shutil.rmtree('plots')

        for batch_dir in batches:
            directory = f'plots/batch_{batch_dir}'
            os.makedirs(directory)  # Recreate the directory

        # delete data/block_pushing_multimodal/eval and then make new
        if os.path.exists('data/blockpush_eval_output/media'):
            shutil.rmtree('data/blockpush_eval_output/media')
        os.makedirs('data/blockpush_eval_output/media')

    def convert_step_images_to_gif(batches):
        
        for batch_dir in batches:
            media_dir = pathlib.Path(output_dir).joinpath('media')
            mp4_filename = media_dir.joinpath(f'batch_{batch_dir}/2d_batch_{batch_dir}.mp4')
            images_pattern = f'plots/batch_{batch_dir}/step_%d.png'
            
            # Remove the existing MP4 file if it exists
            if os.path.exists(mp4_filename):
                os.remove(mp4_filename)
            
            # Convert images to video using ffmpeg
            os.system(f"ffmpeg -framerate 2 -i {images_pattern} -c:v libx264 -r 30 {mp4_filename}")

    
    clear_directory(DISPLAY_BATCHES)


    # Commented out -- always overrides. easier to debug
    # if os.path.exists(output_dir):
    #     click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    # workspace = cls(cfg, output_dir=output_dir)
    workspace = cls(cfg)
    workspace._output_dir = output_dir
    
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    if n_test is not None:
        cfg.task.env_runner['n_test'] = n_test
    
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


    convert_step_images_to_gif(DISPLAY_BATCHES)
    
if __name__ == '__main__':
    main()
