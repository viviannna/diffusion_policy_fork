from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

import numpy as np

class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory
    
    def rotate_point(self, x, y, deg):
        '''
        Rotates a given point (x, y) by deg degrees around the origin (0, 0)
        '''
        # Convert degrees to radians
        theta = np.radians(deg)

        # Compute cosine and sine of the angle
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Apply rotation matrix
        x_new = x * cos_theta - y * sin_theta
        y_new = x * sin_theta + y * cos_theta

        return x_new, y_new
    

    def rotate_observed_effector(self, batch, nobs, rotation_angle):
        '''
        Rotates the value of the effector in the t-1 observation. Lying about where we were two observations ago. We are NOT rotating the trajectory. 
        '''

        nobs_copy = nobs.clone()
        OBS_STEP = 0  # Which of the observations we're at (0 = t-1, first observation, 1 = t the observation right before this one)

        effector_actual = {
            'x': nobs_copy[batch][OBS_STEP][6], 
            'y': nobs_copy[batch][OBS_STEP][7],
        }

        rotated_effector = {
            'x': nobs_copy[batch][OBS_STEP][6], 
            'y': nobs_copy[batch][OBS_STEP][7],
        }

        rotated_effector['x'], rotated_effector['y'] = self.rotate_point(x=effector_actual['x'], y=effector_actual['y'], deg=rotation_angle)

        nobs[batch][OBS_STEP][6] = rotated_effector['x']
        nobs[batch][OBS_STEP][7] = rotated_effector['y']

        return nobs
    
    def rotate_if_blocks_dist_5(self, obs_dict, nobs, B):
        '''
        Condition: Both blocks have traveled less than DISTANCE_THRESHOLD in the past 5 steps combined.

        Lie: Rotate the observed effector in t-1 observation by rotation_angle degrees.

        DISTANCE_TRESHOLD = 0.001

        '''

        DISTANCE_TRESHOLD = 0.001 

        rotation_angle = int(obs_dict['rotation_angle'])
        for batch in range(B):
            blocks_dist_5 = obs_dict['blocks_dist'][batch].sum().item()
            curr_step = obs_dict['step']
            last_lie_step = obs_dict['last_lie_step'][batch]
            
            if blocks_dist_5 < DISTANCE_TRESHOLD and curr_step >= 5 and (curr_step - last_lie_step) > 5:

                nobs = self.rotate_observed_effector(batch, nobs, rotation_angle)
                obs_dict['last_lie_step'][batch] = curr_step
  

        return nobs, obs_dict
    
    def rotate_if_effector_dist_5(self, obs_dict, nobs, B):
        """
        Condition: The effector has traveled less than DISTANCE_TRESHOLD in the past 5 steps.

        Lie: Rotate the observed effector in t-1 observation by rotation_angle degrees.

        DISTANCE_TRESHOLD = 0.05
        """

        DISTANCE_TRESHOLD = 0.05
        rotation_angle = int(obs_dict['rotation_angle'])
        for batch in range(B):  
            effector_dist_5 = obs_dict['effector_dist'][batch].sum().item()
            curr_step = obs_dict['step']
            last_lie_step = obs_dict['last_lie_step'][batch]

            if effector_dist_5 < DISTANCE_TRESHOLD and curr_step >= 5 and (curr_step - last_lie_step) > 5:
                nobs = self.rotate_observed_effector(batch, nobs, rotation_angle)
                obs_dict['last_lie_step'][batch] = curr_step

        return nobs, obs_dict
    

    def translate_at_angle(self, x, y, deg, distance):
        angle_radians = np.radians(deg)

        # Calculate the new coordinates using numpy's cos and sin functions
        new_x = x + distance * np.cos(angle_radians)
        new_y = y + distance * np.sin(angle_radians)
        
        return new_x, new_y
    
    # NOTE: Do I want to lie about all of the blocks or just claculate which one is closest to the targe -- technically privledged info

    # NOTE: Currently lying about both block positions 
    def translate_obs_blocks(self, batch, nobs, deg, distance):
        
        """
        
        Helper function for translate_if_blocks_dist_5
        """

        nobs_copy = nobs.clone()
        OBS_STEP = 0 

        block_actual = {
            'x': nobs_copy[batch][OBS_STEP][0], 
            'y': nobs_copy[batch][OBS_STEP][1]
        }

        nobs[batch][OBS_STEP][0], nobs[batch][OBS_STEP][1]= self.translate_at_angle(x=block_actual['x'], y=block_actual['y'], deg=deg, distance=distance)

        block2_actual = {
            'x': nobs_copy[batch][OBS_STEP][3], 
            'y': nobs_copy[batch][OBS_STEP][4]
        }

        nobs[batch][OBS_STEP][3], nobs[batch][OBS_STEP][4] = self.translate_at_angle(x=block2_actual['x'], y=block2_actual['y'], deg=deg, distance=distance)

        return nobs
        
    def translate_if_blocks_dist_5(self, obs_dict, nobs, B):
        """
        
        Condition: Both blocks have traveled less than DISTANCE_THRESHOLD in the past 5 steps combined.

        Lie: Translate the observed block positions in t-1 observation by translation_angle degrees and distance distance.
        """

        DISTANCE_THRESHOLD = 0.05

        translation_angle = int(obs_dict['translation_angle'])
        distance = (obs_dict['translation_distance']).item()

        for batch in range(B):

            blocks_dist_5 = obs_dict['blocks_dist'][batch].sum().item()
            curr_step = obs_dict['step']
            last_lie_step = obs_dict['last_lie_step'][batch]

            if blocks_dist_5 < DISTANCE_THRESHOLD and curr_step >= 5 and (curr_step - last_lie_step) > 5:

                nobs = self.translate_obs_blocks(batch, nobs, translation_angle, distance)
                obs_dict['last_lie_step'][batch] = curr_step

        return nobs, obs_dict

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            
        elif self.obs_as_global_cond:
            # condition throught global feature

            # nobs[:,:To] is the past observation of shape (56, 2, 16) -- 56 batches, 2 timesteps, 16 features

            lie_cond = int(obs_dict['lie_cond'])

            # Rotate observed effector
            rotate = int(obs_dict['rotate'])
            if rotate != 0: 
                lie_cond = int(obs_dict['lie_cond'])
                if lie_cond == 0:
                    nobs, obs_dict = self.rotate_if_blocks_dist_5(obs_dict, nobs, B)    
                elif lie_cond == 1: 
                    nobs, obs_dict = self.rotate_if_effector_dist_5(obs_dict, nobs, B)

            translate = int(obs_dict['translate'])
            if translate != 0:
                lie_cond = int(obs_dict['lie_cond'])
                if lie_cond == 0:
                    nobs, obs_dict = self.translate_if_blocks_dist_5(obs_dict, nobs, B)
        

                # elif lie_cond == 1:
                #     nobs, obs_dict = self.translate_if_blocks_dist_5(self, obs_dict=obs_dict, nobs=nobs, B=B)

            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred,
            'last_lie_step': obs_dict['last_lie_step']
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
