import zarr
import os
import shutil
import plot_training_demos as pu
import numpy as np
from scipy.spatial.distance import euclidean
import subprocess
import glob
import re

from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment
import numpy as np

global EPISODE_STARTS 

# 1,001 demonstrations (root['meta']['EPISODE_STARTS'])
# This is actually really a list of inclusive starts.
# First demonstration ranges from [0, 103]
# Second demonstration ranges from [104, 226]
# Note that the last value (114962) is actually out of bounds. 
# So we range from [EPISODE_STARTS[i], EPISODE_STARTS[i+1]-1] 
# Note I manually added the 0 to the beginning after copy pasting from a breakpoint. 

EPISODE_STARTS = [0,   104,    227,    352,    461,    587,    695,    816,    917,
         1023,   1131,   1253,   1379,   1479,   1599,   1702,   1819,
         1940,   2056,   2164,   2262,   2394,   2515,   2633,   2735,
         2841,   2950,   3063,   3176,   3286,   3399,   3524,   3647,
         3769,   3885,   4009,   4139,   4243,   4360,   4447,   4551,
         4654,   4763,   4865,   4967,   5090,   5207,   5321,   5418,
         5532,   5640,   5762,   5891,   6017,   6135,   6241,   6358,
         6483,   6601,   6703,   6820,   6950,   7076,   7188,   7306,
         7419,   7543,   7672,   7783,   7893,   8008,   8111,   8226,
         8335,   8445,   8558,   8685,   8795,   8916,   9016,   9136,
         9242,   9358,   9477,   9584,   9686,   9798,   9928,  10035,
        10140,  10276,  10393,  10509,  10622,  10725,  10828,  10944,
        11041,  11137,  11252,  11365,  11497,  11611,  11715,  11865,
        11983,  12101,  12196,  12299,  12423,  12533,  12636,  12755,
        12872,  12988,  13096,  13207,  13319,  13436,  13542,  13670,
        13781,  13905,  14014,  14133,  14239,  14353,  14474,  14595,
        14716,  14816,  14934,  15055,  15178,  15304,  15418,  15539,
        15653,  15766,  15872,  16010,  16128,  16248,  16352,  16476,
        16588,  16714,  16826,  16934,  17049,  17149,  17272,  17395,
        17501,  17622,  17741,  17866,  17969,  18080,  18199,  18330,
        18449,  18573,  18679,  18789,  18913,  19022,  19132,  19240,
        19361,  19471,  19589,  19685,  19799,  19915,  20021,  20129,
        20253,  20361,  20484,  20614,  20727,  20856,  21001,  21091,
        21196,  21318,  21421,  21539,  21647,  21759,  21869,  21989,
        22104,  22215,  22330,  22449,  22560,  22677,  22780,  22885,
        23004,  23117,  23231,  23356,  23477,  23593,  23714,  23840,
        23948,  24058,  24184,  24290,  24408,  24520,  24648,  24758,
        24878,  24978,  25088,  25203,  25321,  25429,  25524,  25635,
        25752,  25871,  25985,  26096,  26218,  26331,  26429,  26543,
        26656,  26767,  26888,  27001,  27120,  27235,  27364,  27469,
        27592,  27710,  27840,  27936,  28056,  28177,  28287,  28404,
        28515,  28625,  28741,  28865,  28972,  29086,  29186,  29306,
        29407,  29525,  29651,  29778,  29893,  30014,  30107,  30213,
        30332,  30424,  30527,  30621,  30746,  30884,  30996,  31110,
        31217,  31328,  31433,  31549,  31648,  31773,  31883,  31981,
        32085,  32200,  32313,  32435,  32554,  32680,  32804,  32926,
        33038,  33152,  33281,  33395,  33516,  33635,  33755,  33879,
        33977,  34096,  34205,  34315,  34442,  34551,  34662,  34788,
        34889,  35006,  35100,  35203,  35307,  35413,  35539,  35652,
        35775,  35889,  35995,  36113,  36225,  36338,  36444,  36557,
        36680,  36795,  36899,  37029,  37158,  37255,  37350,  37459,
        37570,  37689,  37817,  37939,  38051,  38158,  38274,  38407,
        38514,  38636,  38740,  38865,  38983,  39082,  39185,  39296,
        39421,  39547,  39676,  39787,  39903,  40003,  40108,  40226,
        40327,  40435,  40557,  40662,  40800,  40916,  41041,  41150,
        41263,  41374,  41489,  41595,  41705,  41804,  41905,  42028,
        42156,  42274,  42373,  42490,  42594,  42697,  42800,  42912,
        43031,  43149,  43280,  43403,  43513,  43623,  43747,  43868,
        43980,  44107,  44225,  44344,  44457,  44590,  44708,  44827,
        44938,  45068,  45173,  45294,  45408,  45527,  45655,  45774,
        45884,  45999,  46120,  46237,  46357,  46465,  46586,  46704,
        46826,  46947,  47062,  47168,  47275,  47388,  47490,  47597,
        47725,  47826,  47932,  48052,  48157,  48267,  48376,  48485,
        48602,  48712,  48812,  48921,  49022,  49147,  49262,  49382,
        49488,  49620,  49733,  49835,  49947,  50064,  50176,  50281,
        50394,  50514,  50629,  50739,  50850,  50972,  51079,  51206,
        51298,  51400,  51504,  51601,  51721,  51828,  51948,  52052,
        52169,  52296,  52388,  52496,  52612,  52963,  53056,  53165,
        53288,  53401,  53527,  53646,  53753,  53887,  54003,  54107,
        54202,  54313,  54419,  54536,  54650,  54760,  54867,  54992,
        55114,  55222,  55338,  55450,  55557,  55676,  55801,  55929,
        56042,  56144,  56269,  56373,  56482,  56577,  56692,  56793,
        56894,  57025,  57150,  57267,  57388,  57508,  57611,  57726,
        57844,  57965,  58090,  58205,  58307,  58418,  58539,  58646,
        58760,  58885,  59021,  59142,  59255,  59379,  59499,  59599,
        59726,  59830,  59949,  60077,  60193,  60298,  60422,  60529,
        60645,  60754,  60868,  60978,  61089,  61207,  61310,  61412,
        61536,  61649,  61748,  61864,  61973,  62068,  62175,  62292,
        62423,  62533,  62643,  62766,  62876,  62995,  63095,  63222,
        63344,  63467,  63578,  63700,  63823,  63930,  64033,  64152,
        64276,  64384,  64502,  64628,  64744,  64861,  64978,  65070,
        65176,  65284,  65397,  65535,  65644,  65750,  65849,  65971,
        66072,  66182,  66294,  66399,  66515,  66642,  66760,  66870,
        66989,  67093,  67220,  67323,  67428,  67535,  67646,  67765,
        67890,  68022,  68145,  68267,  68385,  68486,  68588,  68715,
        68835,  68950,  69301,  69411,  69536,  69666,  69786,  69901,
        70024,  70149,  70274,  70381,  70509,  70621,  70722,  70850,
        70953,  71076,  71196,  71302,  71420,  71530,  71646,  71757,
        71850,  71977,  72089,  72213,  72321,  72448,  72568,  72690,
        72800,  72922,  73027,  73147,  73264,  73388,  73497,  73602,
        73723,  73838,  73947,  74068,  74194,  74298,  74412,  74529,
        74648,  74765,  74896,  74997,  75106,  75223,  75328,  75431,
        75548,  75675,  75784,  75890,  76001,  76105,  76225,  76344,
        76461,  76577,  76675,  76794,  76912,  77020,  77134,  77247,
        77354,  77478,  77589,  77697,  77818,  77951,  78069,  78175,
        78285,  78394,  78516,  78635,  78765,  78868,  79008,  79130,
        79247,  79364,  79470,  79580,  79696,  79802,  79909,  80005,
        80112,  80225,  80336,  80445,  80557,  80663,  80788,  80914,
        81024,  81133,  81258,  81378,  81497,  81616,  81731,  81846,
        81975,  82071,  82193,  82304,  82433,  82546,  82659,  82803,
        82915,  83040,  83161,  83283,  83379,  83472,  83593,  83707,
        83826,  83921,  84045,  84156,  84273,  84400,  84514,  84611,
        84719,  84842,  84963,  85163,  85278,  85376,  85491,  85603,
        85727,  85847,  85952,  86067,  86192,  86301,  86409,  86536,
        86655,  86763,  86872,  86974,  87091,  87193,  87317,  87428,
        87553,  87684,  87785,  87896,  88015,  88131,  88249,  88373,
        88475,  88572,  88691,  88810,  88928,  89050,  89158,  89260,
        89368,  89481,  89591,  89710,  89812,  89923,  90039,  90161,
        90271,  90390,  90512,  90614,  90708,  90828,  90919,  91029,
        91144,  91251,  91359,  91472,  91581,  91776,  91893,  92009,
        92124,  92234,  92338,  92433,  92563,  92677,  92801,  92920,
        93033,  93148,  93279,  93387,  93485,  93597,  93691,  93796,
        93916,  94042,  94138,  94252,  94362,  94468,  94584,  94695,
        94830,  94930,  95052,  95166,  95281,  95382,  95494,  95617,
        95720,  95832,  95961,  96078,  96211,  96330,  96681,  96781,
        96893,  96997,  97104,  97237,  97332,  97443,  97560,  97665,
        97798,  97899,  98020,  98127,  98226,  98352,  98460,  98575,
        98670,  98794,  98901,  98998,  99107,  99217,  99333,  99453,
        99551,  99659,  99769,  99886,  99998, 100133, 100241, 100367,
       100480, 100596, 100705, 100825, 100949, 101072, 101190, 101293,
       101418, 101540, 101654, 101758, 101872, 101988, 102098, 102217,
       102330, 102439, 102556, 102679, 102797, 102907, 103008, 103119,
       103220, 103320, 103427, 103546, 103661, 103782, 103894, 104003,
       104122, 104245, 104353, 104475, 104576, 104705, 104811, 104927,
       105044, 105162, 105261, 105367, 105475, 105590, 105677, 105805,
       105911, 106031, 106140, 106249, 106343, 106460, 106564, 106678,
       106779, 106904, 107006, 107110, 107243, 107359, 107480, 107576,
       107693, 107812, 107936, 108053, 108170, 108298, 108409, 108519,
       108629, 108737, 108854, 108963, 109093, 109196, 109322, 109433,
       109536, 109654, 109775, 109872, 109991, 110116, 110219, 110329,
       110445, 110565, 110696, 110829, 110952, 111044, 111147, 111255,
       111364, 111483, 111610, 111715, 111824, 111940, 112076, 112186,
       112314, 112416, 112512, 112633, 112745, 112871, 112980, 113113,
       113223, 113358, 113463, 113567, 113681, 113794, 113918, 114039,
       114155, 114278, 114383, 114499, 114611, 114739, 114851, 114962]



import numpy as np
import os
import shutil

def reorder_observations_first_n(labels, obs, actions, n=470):
    """
    Utility function to reorder the first n timesteps according to a label map
    and then concatenate the remainder. 
    This is intentionally separate from the PathSegmenter class.
    """
    assert False

    # I don't think any of this function gets used anymore. 

    # 1) Slice out the first n items
    labels_sub  = labels[:n]
    obs_sub     = obs[:n]
    actions_sub = actions[:n]

    # The remainder that stays untouched
    labels_rest  = labels[n:]
    obs_rest     = obs[n:]
    actions_rest = actions[n:]

    # 2) Define label priorities
    order_map = {
        "path1_before_k":  0,
        "path0_after_k":   1,
        "path0_before_k":  2,
        "path1_after_k":   3,
    }
    LOWEST_PRIORITY = 999

    def get_sort_key(label):
        # Unknown labels => 999, so they end up at the back (but preserve relative order).
        return order_map.get(label, LOWEST_PRIORITY)

    # 3) Zip the sub-block so we can sort them together.
    zipped_sub = list(zip(labels_sub, obs_sub, actions_sub))
    # Sort by label priority
    zipped_sub_sorted = sorted(zipped_sub, key=lambda x: get_sort_key(x[0]))
    # Unzip back into separate sequences
    labels_sub_sorted, obs_sub_sorted, actions_sub_sorted = zip(*zipped_sub_sorted)

    # 4) Convert obs_sub_sorted / actions_sub_sorted to NumPy arrays if needed
    obs_sub_sorted     = np.array(obs_sub_sorted)
    actions_sub_sorted = np.array(actions_sub_sorted)

    # 5) Concatenate sorted sub-block with remainder along axis=0
    obs_concat     = np.concatenate([obs_sub_sorted, obs_rest], axis=0)
    actions_concat = np.concatenate([actions_sub_sorted, actions_rest], axis=0)

    # 6) Convert them back to lists or keep them as arrays
    new_labels  = list(labels_sub_sorted) + labels_rest
    new_obs     = obs_concat.tolist()
    new_actions = actions_concat.tolist()

    return new_labels, new_obs, new_actions

# ------------------------------------------------------------------
# PathSegmenter: Analyzing and Modifying Effector Trajectories
# ------------------------------------------------------------------
# The PathSegmenter class processes the trajectory of a demonstration 
# (from start_timestep to end_timestep) using observation and action 
# dictionaries. It is designed to:
#
# - Identify key transition points in the trajectory, such as when 
#   blocks enter target areas.
# - Segment the trajectory into meaningful phases for analysis and 
#   modification.
# - Construct new trajectory representations based on these key points.
# - Assign labels to different trajectory segments to track progress.
# - Provide visualization methods to highlight trajectory structure 
#   and decision points.
#
# This class is integral for studying effector behavior in block-pushing 
# tasks, allowing users to analyze how paths evolve over time, compare 
# different movement strategies, and explore alternative trajectories.
# ------------------------------------------------------------------


class PathSegmenter:
    def __init__(self, obs, action, start_timestep, end_timestep, demo_num):
        """
        Initialize with required data and parameters.
        """
        self.obs = obs
        self.action = action
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.demo_num = demo_num

        # These values will be set by calculate_key_points()
        self.no_blocks = None
        self.one_block = None
        self.pivot_point = None
        self.both_blocks = None

        # If you have a special "switch step" concept
        self.switch_step_k = None

        # This will be populated by label_segments_from_k()
        self.labels = None

        # (Optional) Store the initial observation for convenience
        self.start_obs = None



    def calculate_key_points(self, pivot="midpoint", type_k="midpoint"):
        """
        First pass through the demonstration to identify key timesteps.
        
        - no_blocks: first time we see neither block in the target
        - one_block: first time we see exactly one block in target
        - both_blocks: first time we see both in target
        - pivot_point: midpoint between one_block and both_blocks

        Plotting and labeling should occur in color_code_ functions, NOT here.  This just populates self values. 
        """
        # You may want to store this for plotting
        self.start_obs = self.obs[self.start_timestep]

        touched_0 = False 
        touched_1 = False


        for step in range(self.start_timestep, self.end_timestep + 1):
            curr_obs = self.obs[step]
            curr_action = self.action[step]

            done, b0_in_target, b1_in_target = pu.is_successful(curr_obs)
            touching_0, touching_1 = pu.is_touching_block(curr_obs)

            if not touched_0 and touching_0:
                self.first_touch_0 = step
                touched_0 = True
            elif not touched_1 and touching_1:
                self.first_touch_1 = step
                touched_1 = True
            
            if not b0_in_target and not b1_in_target:
                # We found the first time no blocks are in the target
                if self.no_blocks is None:
                    self.no_blocks = step
            elif b0_in_target != b1_in_target:
                # Exactly one block in target
                if self.one_block is None:
                    self.one_block = step
            else:
                # Both blocks in target
                if self.both_blocks is None:
                    self.both_blocks = step

        if any(x is None for x in [self.no_blocks, self.one_block, self.both_blocks]):
            raise ValueError("Could not determine pivot points correctly.")

        self.midpoint = self.one_block + ((self.both_blocks - self.one_block) // 2)

        # Pivot splits the two different points
        if pivot == "midpoint":
            self.pivot_point = self.midpoint
        
        elif pivot == "closest_to_base":
            base_position = np.array([0.3, -0.4]) # (Assumes starting effector position)
            min_distance = float("inf")
            best_pivot = None

            # Iterate over candidate pivot points
            for step in range(self.one_block, self.both_blocks + 1):

                curr_position = self.action[step]

                # Calculate distance to base position
                distance = np.linalg.norm(curr_position - base_position)

                if distance < min_distance:
                    min_distance = distance
                    best_pivot = step

            self.pivot_point = best_pivot

        # K is the point to jump between paths
        if type_k == "fixed":
            print("NOTE: Our k values are fixed for this run. ")
            pass # Should be assigned when label_segments_from_k() is called

        # Between start and touching the first block... for now I'm just going to make it the same for both pathways
        elif type_k == "midpoint":
            earliest = min(self.first_touch_0, self.first_touch_1)

            # Compute cumulative distance at each step
            cumulative_distance = 0
            half_distance = None
            total_distance = 0
            
            for step in range(self.start_timestep, earliest + 1):
                # Compute distance traveled at this step
                step_distance = pu.get_step_distance(self.obs[step], self.obs[step - 1])
                total_distance += step_distance
            
            half_distance = total_distance / 2  # Midpoint in terms of distance
            cumulative_distance = 0

            for step in range(self.start_timestep, earliest + 1):
                step_distance = pu.get_step_distance(self.obs[step], self.obs[step - 1])
                cumulative_distance += step_distance
                
                if cumulative_distance >= half_distance:

                    # Remember that switch step always be relative to zero 
                    self.switch_step_k = (step - self.start_timestep) + 1
                    break
            
            
            
            
            # This calculates it by picking the middle time step 
            # earliest = min(self.first_touch_0, self.first_touch_1)
            # distance_between_first_touch = earliest - self.start_timestep 
            # self.switch_step_k = (distance_between_first_touch // 2)

            # I think instead we should just pick the point closest to the middle distance travelled -- I don't think that I really need to track the actual trajectory taken (I hope..?)




        elif type_k == "heuristic_middle":
            print("TODO: Implement heuristic_middle for k.")
            raise NotImplementedError
            pass
            # calculate the vector from the self.start_timestep to 


        print(f"[Demo {self.demo_num}] no_blocks={self.no_blocks}, "
              f"one_block={self.one_block}, pivot={self.pivot_point}, "
              f"both_blocks={self.both_blocks}", f"switch_step_k={self.switch_step_k}")
        
        # Set the value of k to be the midpoint between one_block and both_blocks


    def label_segments_from_k(self):
        """
        Label each timestep between start_timestep and end_timestep as belonging to
        one of the 'path0' or 'path1' segments, specifically marking whether
        it comes before/at/after the switch_step_k.

        This stores all labels in self.labels.
        """

        # if any(x is None for x in [self.no_blocks, self.one_block, self.both_blocks, self.pivot_point]):
        #     raise RuntimeError("Pivot points have not been calculated yet. "
        #                        "Call calculate_key_points() first.")

        # Create a label array for the entire dataset length. We'll fill only the relevant range.

        assert self.switch_step_k is not None

        self.labels = [''] * len(self.obs)

        for step in range(self.start_timestep, self.end_timestep + 1):
            if self.no_blocks <= step <= (self.switch_step_k + self.start_timestep):
                self.labels[step] = 'pathA_before_k'
            elif (self.switch_step_k + self.start_timestep) < step <= self.pivot_point:
                self.labels[step] = 'pathA_after_k'
            
            if self.pivot_point < step <= (self.pivot_point + self.switch_step_k):
                self.labels[step] = 'pathB_before_k'
            elif (self.pivot_point + self.switch_step_k) < step <= self.end_timestep:
                self.labels[step] = 'pathB_after_k'
            else:
                pass  # If there's some gap or off-by-one, handle or assert as needed.

        return self.labels

    # ------------------------------------------------------------------
    # Color-coding methods
    # ------------------------------------------------------------------

    def color_by_time(self):
        """
        Simple time-based color gradient, marking pivot point in black.
        """
        if self.pivot_point is None:
            raise RuntimeError("Call calculate_key_points() before calling color methods.")

        for step in range(self.start_timestep, self.end_timestep + 1):
            curr_action = self.action[step]

            if step == self.pivot_point:
                color = 'black'
            else:
                color = 'gradient'

            pu.plot_denoising_trajectories(action=curr_action, run_step=(step - self.start_timestep), demo_num=self.demo_num, color=color)

    def color_code_3_segments(self):
        """
        Colors the path in 3 segments:
        1) [no_blocks, one_block]
        2) [one_block, pivot_point]
        3) [pivot_point, both_blocks]
        pivot_point is explicitly indicated (e.g., green).
        """
        if any(x is None for x in [self.no_blocks, self.one_block, self.both_blocks, self.pivot_point]):
            raise RuntimeError("Call calculate_key_points() first.")

        
        for step in range(self.start_timestep, self.end_timestep + 1):
            curr_action = self.action[step]

            alpha = 0.5

            if self.no_blocks <= step <= self.one_block:
                color = 'red'
            elif step == self.pivot_point:
                color = 'green'
                alpha = 1.0
            elif self.one_block < step <= self.pivot_point:
                color = 'yellow'
            elif self.pivot_point < step <= self.both_blocks:
                color = 'blue'
            else:
                # If there's a gap after both_blocks or before no_blocks, handle or raise an error
                raise ValueError(f"Step {step} out of expected range.")

            pu.plot_effector_actions(action=curr_action, run_step=step, demo_num=self.demo_num, color=color, alpha=alpha)

            if step == self.pivot_point:
                pu.custom_label(demo_num=self.demo_num, custom_text=f"Pivot Point at ({curr_action[0]:.2f}, {curr_action[1]:.2f})", color='green')
                pu.arrow_to_point(demo_num=self.demo_num, x_target=curr_action[0], y_target=curr_action[1])

            if step == self.first_touch_0:
                print(f"First touch 0 at {step}")
                pu.arrow_to_point(demo_num=self.demo_num, x_target=curr_action[0], y_target=curr_action[1], color='blue')
            elif step == self.first_touch_1:
                print(f"First touch 1 at {step}")
                pu.arrow_to_point(demo_num=self.demo_num, x_target=curr_action[0], y_target=curr_action[1], color='orange') 

    def color_code_at_k(self):
        """
        Colors the path with respect to `switch_step_k`, using colors based on labels.

        This function assumes `self.labels` is already populated by `label_segments_from_k()`.
        It uses a dictionary `label_color_dict` where keys are labels and values are their corresponding colors.

        Args:
            label_color_dict (dict): Mapping of labels (from `label_segments_from_k`) to colors.
        """
        assert self.switch_step_k is not None
        assert self.pivot_point is not None
        assert self.labels is not None

        label_color_dict = {
            'pathA_before_k': 'red',
            'pathA_after_k': 'orange',
            'pathB_before_k': 'blue',
            'pathB_after_k': 'purple',
            'switch_step_k': 'cyan',
        }

        for step in range(self.start_timestep, self.end_timestep + 1):
            curr_action = self.action[step]

            # Handle special cases for first touches
            if step == self.first_touch_0:
                print(f"First touch 0 at {step}")
                pu.arrow_to_point(demo_num=self.demo_num, x_target=curr_action[0], y_target=curr_action[1], color='blue')
            elif step == self.first_touch_1:
                print(f"First touch 1 at {step}")
                pu.arrow_to_point(demo_num=self.demo_num, x_target=curr_action[0], y_target=curr_action[1], color='orange')

            # Assign colors based on labels
            label = self.labels[step]

            # Handle special cases for colors
            if step == self.switch_step_k + self.start_timestep or step == self.pivot_point + self.switch_step_k:
                color = label_color_dict.get('switch_step_k', 'gray')
            else:
                color = label_color_dict.get(label, 'gray')  # Default to gray if label is missing

            pu.plot_effector_actions(action=curr_action, run_step=step, demo_num=self.demo_num, color=color)

            if step == self.pivot_point:
                pu.custom_label(demo_num=self.demo_num, custom_text=f"Pivot Point at ({curr_action[0]:.2f}, {curr_action[1]:.2f})", color='green')
                pu.arrow_to_point(demo_num=self.demo_num, x_target=curr_action[0], y_target=curr_action[1])

            

    # ------------------------------------------------------------------
    # One-shot convenience  to calculate the key points of a single demo
    # and set up and finalize the full trajectory plot. (Individual steps 
    # are plotted in the color-coding methods.)
    # ------------------------------------------------------------------

    def chunk_path(self, switch_step_k=None, type_k="midpoint", pivot="closest_to_base", dist=None, target_num=None):
        """
        Convenience method that for a single demonstration:
        - Initializes the global plot
        - Calculates pivot points
        - Labels the segments
        - (Optionally) calls a specific color-coding method
        - Plots the final state of the environment

        Parameters:
            switch_step_k: If not none, fixes the switch step to a specific value.
            type_k: How k should be calculated. Options are {"fixed", "midpoint", "heuristic_middle"}
            pivot: How the pivot point should be calculated. Options are {"midpoint", "closest_to_base"}
            dist: If not None, labels the distance from this environment to the target environment.
            target_num: If not None, corresponds to the target environment number you are comparing to. 

        Returns:
        --------
        self.labels : The final list of labels for each timestep.
        """
        pu.setup_full_trajectory_plot(self.obs[self.start_timestep], self.demo_num)

        self.calculate_key_points(pivot=pivot, type_k=type_k)

        if type_k == "fixed":
            self.switch_step_k = switch_step_k

        # if switch_step_k is not None:
        #     switch_step_k = self.midpoint
        #     self.label_segments_from_k(switch_step_k)
        self.label_segments_from_k()
        self.color_code_at_k()
        # self.color_code_3_segments()

        if dist is not None and target_num is not None:
            pu.label_environment_distance(current_num=self.demo_num, target_num=target_num, dist=dist)

        pu.finalize_full_trajectory_plot(obs=self.obs[self.end_timestep], demo_num=self.demo_num, coloring="at_k")
        return self.labels

    def plot_artificial_path(self, custom_file_name=None):

        pu.setup_full_trajectory_plot(self.obs[self.start_timestep], self.demo_num)

        for step in range(self.start_timestep, self.end_timestep +1 ):
            curr_action = self.action[step]
            pu.plot_effector_actions(action=curr_action, run_step=step, demo_num=self.demo_num, color='gradient', start_timestep=self.start_timestep)

        pu.finalize_full_trajectory_plot(obs=self.obs[self.end_timestep], demo_num=self.demo_num, coloring="gradient", custom_file_name=custom_file_name)

        # We don't really care about segments anymore since we merged them together into one artificial demo

         
    def plot_each_step(self, custom_file_name=None):
        """
            Instead of plotting all the steps over time, plot each step and then create a video of all them together. Useful for seeing the environment at each step. 
        """

        for step in range(self.start_timestep, self.end_timestep+1):

            file_name = f"{custom_file_name}_{step}.png"
            pu.setup_full_trajectory_plot(self.obs[step], self.demo_num)
            curr_action = self.action[step]
            pu.plot_effector_actions(action=curr_action, run_step=step, demo_num=self.demo_num, color='gradient', start_timestep=self.start_timestep, label_step=True)
            pu.finalize_full_trajectory_plot(obs=self.obs[step], demo_num=self.demo_num, coloring="gradient", custom_file_name=file_name)

        subprocess.run([
        "ffmpeg", "-framerate", "2", "-start_number", str(self.start_timestep),
        "-i", f"global_plots/{custom_file_name}_%d.png",
        "-vf", "scale=1000:-1:flags=lanczos",
        "-loop", "0", "-y", "demo0_everystep.mp4"
        ], capture_output=True, text=True)





        






# ------------------------------------------------------------------
# ModifyDemos: Global Analysis and Comparison of Demonstrations
# ------------------------------------------------------------------
# The ModifyDemos class provides a global perspective on multiple 
# demonstrations, allowing for the comparison and inspection of 
# environments relative to a chosen target demonstration.
#
# Key Functionalities:
# - Loads and stores observation/action data from a Zarr dataset 
#   in either absolute or relative coordinate mode.
# - Extracts block positions from demonstrations for further analysis.
# - Computes similarity between environments using the Hungarian 
#   algorithm to match blocks and minimize positional differences.
# - Identifies and retrieves all unique environments across 
#   demonstrations, tracking key timestamps and block positions.
# - Finds the closest environments to a target demo based on spatial 
#   similarity, allowing for structured comparisons.
# - Facilitates visualization and segmentation of trajectories using 
#   the PathSegmenter class, enabling deeper insights into trajectory 
#   differences.
#
# This class is essential for understanding variations between 
# different demonstrations, analyzing movement consistency, and 
# identifying key differences in effector behavior across tasks.
# ------------------------------------------------------------------

class ModifyDemos:
    def __init__(self, mode='abs'):
        """
        Open the desired zarr dataset. Store observations/actions for further use.
        """
        if mode == 'abs':
            self.zarr_abs = zarr.open("multimodal_push_seed_abs.zarr", mode='r')
            self.obs = self.zarr_abs['data']['obs']
            self.action = self.zarr_abs['data']['action']
        elif mode == 'rel':
            self.zarr_rel = zarr.open("multimodal_push_seed.zarr", mode='r')
            self.obs = self.zarr_rel['data']['obs']
            self.action = self.zarr_rel['data']['action']
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _get_blocks_as_dicts(self, obs):
        """
        Given a single observation with 6 block entries,
        return a list of two dicts (one for each block).
        """
        block1 = {
            'x': obs[0],
            'y': obs[1],
            'orientation': obs[2]
        }
        block2 = {
            'x': obs[3],
            'y': obs[4],
            'orientation': obs[5]
        }
        return [block1, block2]

    def calculate_environment_similarity(self, blocks_env1, blocks_env2, method="rms"):
        """
        Computes the similarity between two lists of block dictionaries, 
        each containing [{'x':..., 'y':..., 'orientation':...}, ...]. 
        We match blocks via the Hungarian algorithm to minimize 
        distance in (x, y) space.

        Arguments:
        -----------
        blocks_env1, blocks_env2 : list of 2 dicts, 
            each dict with keys 'x','y','orientation'.
        method : str, 'rms' or 'mean'

        Returns:
        --------
        A single float indicating the distance between the two envs.
        """
        # Unpack each environment (2 blocks each).
        block1_env1, block2_env1 = blocks_env1
        block1_env2, block2_env2 = blocks_env2

        # Prepare for cost matrix in (x,y) only
        blocks1 = [
            {'pos': [block1_env1['x'], block1_env1['y']], 
             'orientation': block1_env1['orientation']},
            {'pos': [block2_env1['x'], block2_env1['y']], 
             'orientation': block2_env1['orientation']}
        ]
        blocks2 = [
            {'pos': [block1_env2['x'], block1_env2['y']], 
             'orientation': block1_env2['orientation']},
            {'pos': [block2_env2['x'], block2_env2['y']], 
             'orientation': block2_env2['orientation']}
        ]

        cost_matrix = np.array([
            [euclidean(b1['pos'], b2['pos']) for b2 in blocks2]
            for b1 in blocks1
        ])

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_distances = [cost_matrix[i, j] for i, j in zip(row_ind, col_ind)]

        if method == "rms":
            return np.sqrt(np.mean(np.square(matched_distances)))
        elif method == 'mean':
            return np.mean(matched_distances)
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_all_environments(self):
        """
        Return a list of dictionaries. Each dict holds:
            - demo_idx (int): Index of this demonstration
            - start_timestep (int)
            - end_timestep (int)
            - blocks (list of 2 dicts): block positions for that environment
        """
        envs = []
        for i in range(len(EPISODE_STARTS) - 1):
            start_ts = EPISODE_STARTS[i]
            end_ts = EPISODE_STARTS[i + 1] - 1
            blocks = self._get_blocks_as_dicts(self.obs[start_ts])

            env_data = {
                'demo_idx': i,
                'start_timestep': start_ts,
                'end_timestep': end_ts,
                'blocks': blocks
            }
            envs.append(env_data)
        return envs

    def find_closest_envs(self, target_env_data, all_envs, method='rms'):
        """
        Given a target environment dict, compute distances to all_envs 
        (each a dict with 'blocks'), return a list sorted by ascending distance.
        """
        target_blocks = target_env_data['blocks']
        distances = []
        for env_data in all_envs:
            dist = self.calculate_environment_similarity(
                target_blocks, env_data['blocks'], method=method
            )
            # Attach the computed distance
            distances.append({
                **env_data,   # copy over everything (demo_idx, start_timestep, etc.)
                'distance': dist
            })

        distances.sort(key=lambda d: d['distance'])
        return distances

    def get_closest_envs(self, target_demo=0, num_demos=None, method='rms'):
        """
        Return:
            - The target environment data (dict)
            - A list of environment dicts sorted by ascending distance 
              (each dict includes 'distance')
        """
        all_envs = self.get_all_environments()

        # Extract the one we want as target
        target_env_data = all_envs.pop(target_demo)

        # Now find the closest ones
        sorted_by_dist = self.find_closest_envs(target_env_data, all_envs, method=method)

        if num_demos is not None:
            sorted_by_dist = sorted_by_dist[:num_demos]

        return target_env_data, sorted_by_dist



    def print_closest_envs(self, target_demo=0, num_demos=3):
        """
        Demonstrates how to:
          1) Grab the target env and its closest demos.
          2) Instantiate PathSegmenter for each one.
          3) Plot/label the path with chunk_path or any other method.
        """

        

        # Get the target environment data and the closest few
        target_env_data, closest_envs = self.get_closest_envs(target_demo, num_demos)

        print(f"\n--- Target Environment: Demo {target_env_data['demo_idx']} ---")
        print(f"Start={target_env_data['start_timestep']}, End={target_env_data['end_timestep']}")

        # ========================================
        # # 1) Plot the target demonstration
        # seg_target = PathSegmenter(
        #     obs=self.obs,
        #     action=self.action,
        #     start_timestep=target_env_data['start_timestep'],
        #     end_timestep=target_env_data['end_timestep'],
        #     demo_num=target_env_data['demo_idx']
        # )
        # # You can pick any method or color coding you like here:
        # labels = seg_target.chunk_path(switch_step_k=None, type_k="midpoint")

        # # 2) Now, do the same for each closest environment
        # print(f"\n--- Closest Environments to Demo {target_demo} ---")
        # for env_data in closest_envs:
        #     print(f"Demo {env_data['demo_idx']} "
        #           f"(start={env_data['start_timestep']}, end={env_data['end_timestep']}) "
        #           f"is {env_data['distance']:.3f} away.")

        #     seg_closest = PathSegmenter(
        #         obs=self.obs,
        #         action=self.action,
        #         start_timestep=env_data['start_timestep'],
        #         end_timestep=env_data['end_timestep'],
        #         demo_num=env_data['demo_idx']
        #     )
        #     # Provide distance & target_num so it can annotate if needed
        #     seg_closest.chunk_path(switch_step_k=None, type_k="midpoint", dist=env_data['distance'], target_num=target_demo)
        
        # ========================================

    def create_artificial_demo(self, start_0, start_1, ordering, custom_file_name=None):
        """
        Creates a new artificial trajectory by extracting segments from two demos and arranging them based on `ordering`.

        Arguments:
        - start_0: int, start timestep of first demonstration
        - start_1: int, start timestep of second demonstration
        - ordering: list of strings defining how to arrange segments (e.g., ['demo0_before_k', 'demo1_after_k'])
        """

        # Find endpoints using EPISODE_STARTS
        end_0 = EPISODE_STARTS[EPISODE_STARTS.index(start_0) + 1] - 1
        end_1 = EPISODE_STARTS[EPISODE_STARTS.index(start_1) + 1] - 1

        # Find demo numbers by getting the index
        demo_num_0 = EPISODE_STARTS.index(start_0)
        demo_num_1 = EPISODE_STARTS.index(start_1)

        # Extract first demonstration
        demo_0 = PathSegmenter(
            obs=self.obs,                  # (114962, 16)
            action=self.action,            # (114962, 2)
            start_timestep=start_0,
            end_timestep=end_0,
            demo_num=demo_num_0
        )

        demo_0.chunk_path(switch_step_k=None, type_k="midpoint", pivot="closest_to_base")

        # demo_0.calculate_key_points(pivot="closest_to_base", type_k="midpoint")
        # demo_0.label_segments_from_k()

        # Extract second demonstration
        demo_1 = PathSegmenter(
            obs=self.obs,
            action=self.action,
            start_timestep=start_1,
            end_timestep=end_1,
            demo_num=demo_num_1
        )

        # TODO: Instead of this call chunk_path (Want to plot the og demos anyways)
        # demo_1.calculate_key_points(pivot="closest_to_base", type_k="midpoint")
        # demo_1.label_segments_from_k()
        demo_1.chunk_path(switch_step_k=None, type_k="midpoint", pivot="closest_to_base")

        # Store extracted segments in a dictionary with 4 segments per demo
        segment_dict = {
            "demo0_pathA_before_k": {
                "obs": np.array([self.obs[i] for i, label in enumerate(demo_0.labels) if label == "pathA_before_k"]),
                "action": np.array([self.action[i] for i, label in enumerate(demo_0.labels) if label == "pathA_before_k"])
            },
            "demo0_pathA_after_k": {
                "obs": np.array([self.obs[i] for i, label in enumerate(demo_0.labels) if label == "pathA_after_k"]),
                "action": np.array([self.action[i] for i, label in enumerate(demo_0.labels) if label == "pathA_after_k"])
            },
            "demo0_pathB_before_k": {
                "obs": np.array([self.obs[i] for i, label in enumerate(demo_0.labels) if label == "pathB_before_k"]),
                "action": np.array([self.action[i] for i, label in enumerate(demo_0.labels) if label == "pathB_before_k"])
            },
            "demo0_pathB_after_k": {
                "obs": np.array([self.obs[i] for i, label in enumerate(demo_0.labels) if label == "pathB_after_k"]),
                "action": np.array([self.action[i] for i, label in enumerate(demo_0.labels) if label == "pathB_after_k"])
            },
            "demo1_pathA_before_k": {
                "obs": np.array([self.obs[i] for i, label in enumerate(demo_1.labels) if label == "pathA_before_k"]),
                "action": np.array([self.action[i] for i, label in enumerate(demo_1.labels) if label == "pathA_before_k"])
            },
            "demo1_pathA_after_k": {
                "obs": np.array([self.obs[i] for i, label in enumerate(demo_1.labels) if label == "pathA_after_k"]),
                "action": np.array([self.action[i] for i, label in enumerate(demo_1.labels) if label == "pathA_after_k"])
            },
            "demo1_pathB_before_k": {
                "obs": np.array([self.obs[i] for i, label in enumerate(demo_1.labels) if label == "pathB_before_k"]),
                "action": np.array([self.action[i] for i, label in enumerate(demo_1.labels) if label == "pathB_before_k"])
            },
            "demo1_pathB_after_k": {
                "obs": np.array([self.obs[i] for i, label in enumerate(demo_1.labels) if label == "pathB_after_k"]),
                "action": np.array([self.action[i] for i, label in enumerate(demo_1.labels) if label == "pathB_after_k"])
            }
        }

        # Dynamically construct the new trajectory
        new_obs = np.concatenate([segment_dict[segment]["obs"] for segment in ordering], axis=0)
        new_action = np.concatenate([segment_dict[segment]["action"] for segment in ordering], axis=0)


        # Dynamically construct the new trajectory
        new_obs = np.concatenate([segment_dict[segment]["obs"] for segment in ordering], axis=0)
        new_action = np.concatenate([segment_dict[segment]["action"] for segment in ordering], axis=0)

        # Append the new demo to obs and action datasets
        self.obs = np.concatenate([self.obs, new_obs], axis=0)
        self.action = np.concatenate([self.action, new_action], axis=0)

        # Update EPISODE_STARTS with the new demo start
        new_demo_start = EPISODE_STARTS[-1] 
        new_demo_end = new_demo_start + (len(new_obs) - 1) # The -1 is an artifact of the way we use EPISODE_STARTS. Episode ends should be the next value - 1 (but then our episode ends are inclusive so we bump the range in our loops by one)
        EPISODE_STARTS.append(new_demo_end + 1)

        # len(self.action) = 115113. so we can only access up to 115112

        # Process the new artificial demo
        new_demo_num = len(EPISODE_STARTS)
        artificial_demo = PathSegmenter(
            obs=self.obs,              # (115113, 16)
            action=self.action,        # (115113, 2)
            start_timestep=new_demo_start,
            end_timestep=new_demo_end,      # Needs to be one less than the actual end 
            demo_num=new_demo_num
        )

        artificial_demo.plot_artificial_path(custom_file_name=custom_file_name)

        # artificial_segmenter.artificial_label_segments_from_k()
        # artificial_segmenter.color_code_at_k()

        # # Finalize visualization
        # pu.finalize_full_trajectory_plot(
        #     obs=new_obs[-1], demo_num=new_demo_num, coloring="artificial_labels"
        # )

    # def loop_through_ordering(self, ordering, group_name="artificial"):
    #     """
    #     Creates an animation "chunk-by-chunk" for each ordering in the list. Allows you to see the generated trajectory in sections. 
        
    #     Requires:
    #         - demos object must already have labels (i.e label_segments_from_k() has been called)

    #     """

    #     for i in range(1, len(ordering)+1, 1):
    #         sub_ordering = ordering[:i]
    #         name = f"{sub_ordering[-1]}"
    #         print("Subordering: ", sub_ordering)
            
    #         self.create_artificial_demo(start_0=0, start_1=76912, ordering=sub_ordering, custom_file_name=f"{group_name}_{i}_{name}")

    #     # delete ouput.gif if it exists
    #     if os.path.exists("output.mp4"):
    #         os.remove("output.mp4")

    #     subprocess.run([
    #     "ffmpeg", "-framerate", "2", "-i", f"global_plots/{group_name}_%d.png",
    #     "-vf", "scale=1000:-1:flags=lanczos,palettegen", "-y", "palette.png"
    #     ])

    #     # Apply palette and slow down frames
    #     subprocess.run([
    #         "ffmpeg", "-framerate", "1.5", "-i", f"global_plots/{group_name}_%d.png",
    #         "-i", "palette.png", "-lavfi", "scale=1000:-1:flags=lanczos [x]; [x][1:v] paletteuse",
    #         "-loop", "0", "output.mp4"
    #     ])

    def loop_through_ordering(self, ordering, group_name="artificial"):
        """
        Creates an animation "chunk-by-chunk" for each ordering in the list. Allows you to see the generated trajectory in sections. 

        Requires:
            - demos object must already have labels (i.e label_segments_from_k() has been called)
        """

        for i in range(1, len(ordering) + 1):
            sub_ordering = ordering[:i]
            name = f"{sub_ordering[-1]}"
            print("Subordering: ", sub_ordering)
            
            self.create_artificial_demo(start_0=0, start_1=76912, ordering=sub_ordering, custom_file_name=f"{group_name}_{i}_{name}")

        # Delete existing output file if it exists
        if os.path.exists("output.mp4"):
            os.remove("output.mp4")

        # Find all matching image files
        files = glob.glob(f"global_plots/{group_name}_*_*.png")

        # Extract frame numbers and sort files accordingly
        def extract_frame_num(filename):
            match = re.search(rf"{group_name}_(\d+)_.*\.png", filename)
            return int(match.group(1)) if match else float("inf")

        sorted_files = sorted(files, key=extract_frame_num)

        # Create a temporary text file listing all the sorted images
        list_file = "image_list.txt"
        with open(list_file, "w") as f:
            for file in sorted_files:
                f.write(f"file '{file}'\n")

        # Generate the palette
        subprocess.run([
            "ffmpeg", "-r", "2", "-f", "concat", "-safe", "0", "-i", list_file,
            "-vf", "scale=1000:-1:flags=lanczos,palettegen", "-y", "palette.png"
        ], check=True)

        # Apply palette and slow down frames
        subprocess.run([
            "ffmpeg", "-r", "1.5", "-f", "concat", "-safe", "0", "-i", list_file,
            "-i", "palette.png", "-lavfi", "scale=1000:-1:flags=lanczos [x]; [x][1:v] paletteuse",
            "-loop", "0", "output.mp4"
        ], check=True)

    


def main():

    
    # Clear old plots
    if os.path.exists("global_plots"):
        shutil.rmtree("global_plots")
    os.makedirs("global_plots", exist_ok=True)

    demos = ModifyDemos()
    

    # NOTE: Also lots of assumptions here about starting on the same path. Should probably enable the ability to filter similarity not just by the same starting direction/which block they go to first. 

    # Dictionary of all types of orderings:
    order = {
        'forward': ['demo0_pathA_before_k', "demo1_pathA_after_k", "demo1_pathB_before_k", "demo1_pathB_after_k"],
        'reverse': ['demo1_pathA_before_k', "demo0_pathA_after_k", "demo0_pathB_before_k", "demo0_pathB_after_k"],
        'demo0': ['demo0_pathA_before_k', "demo0_pathA_after_k", "demo0_pathB_before_k", "demo0_pathB_after_k"],
        'demo1': ['demo1_pathA_before_k', "demo1_pathA_after_k", "demo1_pathB_before_k", "demo1_pathB_after_k"]
    }
    
    
    # Forward
    ordering = order['demo0']

    demos.create_artificial_demo(start_0=0, start_1=76912, ordering=ordering, custom_file_name=f"artificial")
    demos.loop_through_ordering(ordering, group_name="artificial")

    





    # demos.print_closest_envs(target_demo=0, num_demos=3)


    # print(os.getcwd())
    # zarr_abs = zarr.open("multimodal_push_seed_abs.zarr", mode='r')
    # zarr_rel = zarr.open("multimodal_push_seed.zarr", mode='r')

    # obs = zarr_rel['data']['obs']
    # action = zarr_abs['data']['action']

    # if os.path.exists('global_plots'):
    #     shutil.rmtree('global_plots')
    #     os.makedirs('global_plots')


    # for demo_num in range(1):
    #     start_timestep = EPISODE_STARTS[demo_num]
    #     end_timestep = EPISODE_STARTS[demo_num+1] -1

    #     print(f"Demo {demo_num}: {start_timestep} to {end_timestep}")

    #     segmenter = PathSegmenter(obs, action, start_timestep, end_timestep, demo_num)
    #     segmenter.chunk_path(switch_step_k=10)

    

if __name__ == "__main__":
    main()



