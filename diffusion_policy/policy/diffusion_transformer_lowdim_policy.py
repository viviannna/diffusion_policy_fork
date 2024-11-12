from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import copy 
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

import numpy as np  # Added import for numpy

import diffusion_policy.policy.utils.plotting_utils as pu

# Define rotations per batch if needed
ROTATIONS_PER_BATCH = [[], [], [], [], [], [], [], [], [], []] 


class DiffusionTransformerLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: TransformerForDiffusion,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_cond=False,
            pred_action_steps_only=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        if pred_action_steps_only:
            assert obs_as_cond

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_dim,
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
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
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
            model_output = model(trajectory, t, cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    # ========= Lies Functions ============
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

    def rotate_observed_effector(self, batch, nobs, rotation_angle, rotation_distance, obs_step=0):
        '''
        Rotates the value of the effector in the t-1 observation. Lying about where we were two observations ago. We are NOT rotating the trajectory. 
        '''
        nobs_copy = copy.deepcopy(nobs)

        effector_actual = {
            'x': nobs_copy[batch][obs_step][6], 
            'y': nobs_copy[batch][obs_step][7],
        }

        rotated_effector = {
            'x': nobs_copy[batch][obs_step][6], 
            'y': nobs_copy[batch][obs_step][7],
        }

        rotated_effector['x'], rotated_effector['y'] = self.translate_at_angle(
            x=effector_actual['x'], y=effector_actual['y'], 
            deg=rotation_angle, distance=rotation_distance
        )

        nobs_copy[batch][obs_step][6] = rotated_effector['x']
        nobs_copy[batch][obs_step][7] = rotated_effector['y']

        return nobs_copy

    def rotate_if_blocks_dist_5(self, obs_dict, nobs, B):
        '''
        Condition: Both blocks have traveled less than DISTANCE_THRESHOLD in the past 5 steps combined.
        Lie: Rotate the observed effector in t-1 observation by rotation_angle degrees.
        '''
        DISTANCE_THRESHOLD = 0.001 

        rotation_angle = int(obs_dict['rotation_angle'])
        rotation_distance = (obs_dict['rotation_distance']).item()
        
        for batch in range(B):
            blocks_dist_5 = obs_dict['blocks_dist'][batch].sum().item()
            curr_step = obs_dict['step']
            last_lie_step = obs_dict['last_lie_step'][batch]
            
            if blocks_dist_5 < DISTANCE_THRESHOLD and curr_step >= 5 and (curr_step - last_lie_step) > 5:
                nobs = self.rotate_observed_effector(batch, nobs, rotation_angle, rotation_distance)
                obs_dict['last_lie_step'][batch] = curr_step

        return nobs, obs_dict

    def rotate_if_effector_dist_5(self, obs_dict, nobs, B):
        """
        Condition: The effector has traveled less than DISTANCE_THRESHOLD in the past 5 steps.
        Lie: Rotate the observed effector in t-1 observation by rotation_angle degrees.
        """
        DISTANCE_THRESHOLD = 0.05
        rotation_angle = int(obs_dict['rotation_angle'])
        rotation_distance = (obs_dict['rotation_distance']).item()

        for batch in range(B):  
            effector_dist_5 = obs_dict['effector_dist'][batch].sum().item()
            curr_step = obs_dict['step']
            last_lie_step = obs_dict['last_lie_step'][batch]

            if effector_dist_5 < DISTANCE_THRESHOLD and curr_step >= 5 and (curr_step - last_lie_step) > 5:
                nobs = self.rotate_observed_effector(batch, nobs, rotation_angle, rotation_distance)
                obs_dict['last_lie_step'][batch] = curr_step

        return nobs, obs_dict

    def rotate_if_first_step(self, obs_dict, nobs, B):
        """
        Condition: First step
        Lie: Rotate the observed effector in the t-1 observation by rotation_angle and rotation_distance.
        """
        rotation_angle = int(obs_dict['rotation_angle'])
        rotation_distance = (obs_dict['rotation_distance']).item()

        for batch in range(B):
            curr_step = obs_dict['step']
            last_lie_step = obs_dict['last_lie_step'][batch]

            if curr_step == 1:
                nobs = self.rotate_observed_effector(batch, nobs, rotation_angle, rotation_distance)
                obs_dict['last_lie_step'][batch] = curr_step

        return nobs, obs_dict

    def rotate_if_custom(self, obs_dict, nobs, B):
        for batch in range(B):
            curr_step = obs_dict['step']
            batch_rotations = ROTATIONS_PER_BATCH[batch]
            for step, rotation_angle, rotation_distance in batch_rotations:
                if curr_step == step:
                    nobs = self.rotate_observed_effector(batch, nobs, rotation_angle, rotation_distance)
                    obs_dict['last_lie_step'][batch] = curr_step

        return nobs, obs_dict

    def translate_at_angle(self, x, y, deg, distance):
        angle_radians = np.radians(deg)
        new_x = x + distance * np.cos(angle_radians)
        new_y = y + distance * np.sin(angle_radians)
        return new_x, new_y

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

        nobs[batch][OBS_STEP][0], nobs[batch][OBS_STEP][1] = self.translate_at_angle(
            x=block_actual['x'], y=block_actual['y'], deg=deg, distance=distance
        )

        block2_actual = {
            'x': nobs_copy[batch][OBS_STEP][3], 
            'y': nobs_copy[batch][OBS_STEP][4]
        }

        nobs[batch][OBS_STEP][3], nobs[batch][OBS_STEP][4] = self.translate_at_angle(
            x=block2_actual['x'], y=block2_actual['y'], deg=deg, distance=distance
        )

        return nobs

    def translate_if_blocks_dist_5(self, obs_dict, nobs, B):
        """
        Condition: Both blocks have traveled less than DISTANCE_THRESHOLD in the past 5 steps combined.
        Lie: Translate the observed block positions in t-1 observation by translation_angle degrees and distance.
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

        # Lie on the actual values THEN normalize
       
        return nobs, obs_dict
    
    def is_touching_block(self, batch, nobs): 
        """
        Helper function that returns whether the effector is touching a block.
        """

        OBS_STEP = 0

        effector = {
            'x': nobs[batch][OBS_STEP][6],
            'y': nobs[batch][OBS_STEP][7]
        }

        block = {
            'x': nobs[batch][OBS_STEP][0],
            'y': nobs[batch][OBS_STEP][1],
            'orientation': nobs[batch][OBS_STEP][2],
        }

        block2 = {
            'x': nobs[batch][OBS_STEP][3],
            'y': nobs[batch][OBS_STEP][4],
            'orientation': nobs[batch][OBS_STEP][5],
        }

        def is_touching(effector, block):
            # Define the distance threshold for "touching"
            touch_threshold = 0.1  # Adjust this value as needed

            # Calculate the Euclidean distance between the effector and the block
            distance = ((effector['x'] - block['x'])**2 + (effector['y'] - block['y'])**2)**0.5
            
            return distance <= touch_threshold

        # Check if the effector is touching either block
        if (is_touching(effector, block)):
            return "block_0"
        elif (is_touching(effector, block2)):
            return "block_1"
        else:
            return None

    def add_vector_to_target(self, obs_dict, nobs, batch, step, touching_block):
        
        nobsc = nobs.cpu().detach().numpy()
        OBS_STEP = 0
        if touching_block == "block_0":
            block = {

                'x': nobsc[batch][OBS_STEP][0],
                'y': nobsc[batch][OBS_STEP][1],
                'orientation': nobs[batch][OBS_STEP][2],
            }
        else:
            block = {
                'x': nobsc[batch][OBS_STEP][3],
                'y': nobsc[batch][OBS_STEP][4],
                'orientation': nobsc[batch][OBS_STEP][5],
            }

        preset_target = PRESET_GOALS[batch][touching_block]

        if preset_target == "target_0":
            target = {
                'x': nobsc[batch][OBS_STEP][10],
                'y': nobsc[batch][OBS_STEP][11],
                'orientation': nobsc[batch][OBS_STEP][12],
            }
        else: 
            target = {
                'x': nobsc[batch][OBS_STEP][13],
                'y': nobsc[batch][OBS_STEP][14],
                'orientation': nobsc[batch][OBS_STEP][15],
            }

        # Calculate the vector pointing from the effector to the block

        ideal_x = block['x'] - target['x']
        ideal_y = block['y'] - target['y']

        return {'ideal_x': ideal_x, 'ideal_y': ideal_y}

    def align_to_target(self, obs_dict, nobs, B):

        # for each batch, check that the effector is touching a block. 
        # if it is, call a function to create a vector that points from the effector to the (**closest??**) block -- perhaps not closest but rather the block that I predetermined per batch (lowkey what if i pass in a starting tuple of the (block, target) and the order matters. once its in the goal pop it off? .. or lets just do one at a time and do the first one first. need to make sure to update the last_lie_step so that 

        ideal_vectors = []

        curr_step = int(obs_dict['step'])
        # NOTE: This is doing it for all batches but I think we could reduce this to only the batches in DISPLAY_BATCHES
        for batch in range(B):

        
            # see which block it is touching
            block = self.is_touching_block(batch, nobs)

            # TODO: eventually fix this to the actual value instead of batch >= 5
            if block is not None and batch >= 5 :
                ideal_vector = self.add_vector_to_target(obs_dict, nobs, batch, curr_step, block)
                ideal_vectors.append(ideal_vector)

        return ideal_vectors

            # follow PRESET_GOALS to check which target we want this block to go towards


            # based off that, create a vector that points from the effector to the target

            # update the effector position (LIE) to be the effector position + vector




    def add_four_directions(self, obs_dict, B):

        '''
        Condition: At each step, user can input {up, down, left, right} to lie about the effector's position. 
        Lie: Rotate the observed effector in t-1 observation by preset (angle, distance) values based on user input.
        '''

        obs = obs_dict['obs']
        curr_step = int(obs_dict['step'].item())
        four_directions = {
            'up': (90, 0.1),
            'down': (270, 0.1),
            'left': (180, 0.1),
            'right': (0, 0.1)
        }

        directions_per_step = {batch: [(0, 0.2) for _ in range(200)] for batch in range(B)}
  
        # I think the problem is that I'm then refeeeding obs into the function instead of rotated_0_1

        rotated_0_1 = obs
        for batch in range(B):
            # (angle, distance) = directions_per_step[batch][curr_step]

            (angle, distance) = (0, 0.1)

            rotated_0 = self.rotate_observed_effector(batch, rotated_0_1, angle, distance)

            rotated_0_1 = self.rotate_observed_effector(batch, rotated_0, angle, distance, obs_step=1)

            obs_dict['last_lie_step'][batch] = curr_step
            if batch in pu.DISPLAY_BATCHES: 
                pu.plot_direction_vector(batch, curr_step, obs, rotated_0_1, obs_step=0)
                pu.plot_direction_vector(batch, curr_step, obs, rotated_0_1, obs_step=1)

                # print("Step 0 and 1 rotated (after print): \n")
                # print(f"{rotated_0_1[batch][0][6].item()}, {rotated_0_1[batch][0][7].item()}")
                # print(f"{rotated_0_1[batch][1][6].item()}, {rotated_0_1[batch][1][7].item()}")
                # print(f"{rotated_0_1[batch][2][6].item()}, {rotated_0_1[batch][2][7].item()}")

                # plot the side
                self.get_push_side(obs, batch, curr_step)
     
        # Lie on regular obs and then normalize
        nobs = self.normalizer['obs'].normalize(rotated_0_1)

        return nobs, obs_dict
    


    def determine_opposite_side_coordinate(self, block_pos, target_pos, width=0.05, height=0.05):
         # Compute direction vector from block to target
        direction_vector = {
            'x': target_pos['x'] - block_pos['x'],
            'y': target_pos['y'] - block_pos['y']
        }

        orientation = block_pos['orientation'].item()
        
        # Determine the dominant movement direction
        if abs(direction_vector['x']) > abs(direction_vector['y']):
            # Horizontal movement is dominant
            if direction_vector['x'] > 0:
                # Desired trajectory is to the right, so opposite side is the left
                offset_x, offset_y = -width / 2, 0
            else:
                # Desired trajectory is to the left, so opposite side is the right
                offset_x, offset_y = width / 2, 0
        else:
            # Vertical movement is dominant
            if direction_vector['y'] > 0:
                # Desired trajectory is upward, so opposite side is the bottom
                offset_x, offset_y = 0, -height / 2
            else:
                # Desired trajectory is downward, so opposite side is the top
                offset_x, offset_y = 0, height / 2

        # Rotate the offset by the block's orientation
        cos_theta = np.cos(orientation)
        sin_theta = np.sin(orientation)
        rotated_offset_x = cos_theta * offset_x - sin_theta * offset_y
        rotated_offset_y = sin_theta * offset_x + cos_theta * offset_y

        # Calculate the opposite side coordinate
        opposite_side_coordinate = {
            'x': block_pos['x'] + rotated_offset_x,
            'y': block_pos['y'] + rotated_offset_y
        }

        return opposite_side_coordinate



    def get_push_side(self, obs, batch, curr_step):
        
        def get_distance_between(x1, y1, x2, y2):
            """
            Calculate the distance between two points. 
            """

            return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        

        # hard code which block should go into one target. 
        # TODO: make this for each step. going to start by only focusing on batch 6

    
        preset_goals = {'block_1': 'target_0', 'block_0': 'target_1'}

            
        # 1. Identify the closest block to the effector
        closest_block = None
        min_distance = float('inf')

        OBS_STEP = 2 # Most recent step (current) 

        # effector = {
        #     'x': obs[batch][OBS_STEP][6],
        #     'y': obs[batch][OBS_STEP][7]
        # }

        # block = {
        #     'x': obs[batch][OBS_STEP][0],
        #     'y': obs[batch][OBS_STEP][1], 
        #     'orientation': obs[batch][OBS_STEP][2]
        # }


        # block_target = preset_goals['block_0']
        # block_to_effector = get_distance_between(effector['x'], effector['y'], block['x'], block['y'])

       

        # # block2_target = preset_goals['block_1']
        # # block2_to_effector = get_distance_between(effector['x'], effector['y'], block2['x'], block2['y'])


        # # # 2. For this closest block and its target, determine which side we should go towards. 

        # # NOTE: for now going to hard code to (block 1, target 0), I'll fix this later

        block = {
            'x': obs[batch][OBS_STEP][0],
            'y': obs[batch][OBS_STEP][1],
            'orientation': obs[batch][OBS_STEP][2]
        }

        target = {
            'x': obs[batch][OBS_STEP][10],
            'y': obs[batch][OBS_STEP][11]
        }

        opposite_side_coordinate = self.determine_opposite_side_coordinate(block, target)
        side_x, side_y = opposite_side_coordinate['x'].item(), opposite_side_coordinate['y'].item()

        pu.plot_coordinate(side_x, side_y, batch, curr_step)
        # so now we have our block and our target position, lets determine which side of the block we should go on

        

    # ========= Predict Action Function ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict  # Not implemented yet
        
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # Build input
        device = self.device
        dtype = self.dtype

        # Handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            # New code to apply lies
            lie_cond = int(obs_dict.get('lie_cond', 0))

            # Rotate observed effector
            rotate = int(obs_dict.get('rotate', 0))
            if rotate != 0: 
                if lie_cond == 0:
                    nobs, obs_dict = self.rotate_if_blocks_dist_5(obs_dict, nobs, B)    
                elif lie_cond == 1: 
                    nobs, obs_dict = self.rotate_if_effector_dist_5(obs_dict, nobs, B)
                elif lie_cond == 2: 
                    nobs, obs_dict = self.rotate_if_first_step(obs_dict, nobs, B)
                elif lie_cond == 3: 
                    nobs, obs_dict = self.rotate_if_custom(obs_dict, nobs, B)

            translate = int(obs_dict.get('translate', 0))
            if translate != 0:
                if lie_cond == 0:
                    nobs, obs_dict = self.translate_if_blocks_dist_5(obs_dict, nobs, B)
            
            vector_to_target = int(obs_dict.get('vector_to_target', 0))
            if vector_to_target != 0:
                ideal_vectors = self.align_to_target(obs_dict, nobs, B)

            four_directions = int(obs_dict.get('four_directions', 0))
            if four_directions != 0:
                nobs, obs_dict = self.add_four_directions(obs_dict, B)

            cond = nobs[:,:To]
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # Condition through inpainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # Run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            **self.kwargs)
        
        # Unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # Get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred,
            'last_lie_step': obs_dict.get('last_lie_step')  # Include if available
            
        }

        # if 'ideal_vectors' in locals() or 'ideal_vectors' in globals():
        #     result['ideal_vectors'] = ideal_vectors
        
        if not self.obs_as_cond:
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        
        # pu.initiate_plot(step, obs_before)
        # pu.close_plot(step)
        return result

    # ========= Training Functions ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        return self.model.configure_optimizers(
                weight_decay=weight_decay, 
                learning_rate=learning_rate, 
                betas=tuple(betas))

    def compute_loss(self, batch):
        # Normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # Handle different ways of passing observation
        cond = None
        trajectory = action
        if self.obs_as_cond:
            cond = obs[:,:self.n_obs_steps,:]
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)
        
        # Generate inpainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise to add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # Compute loss mask
        loss_mask = ~condition_mask

        # Apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, cond)

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
