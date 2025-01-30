import zarr
import os
import shutil
import copy_plotting_utils as pu
import numpy as np
from scipy.spatial.distance import euclidean


from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment
import numpy as np

global EPISODE_ENDS 
EPISODE_ENDS = [0,   104,    227,    352,    461,    587,    695,    816,    917,
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



class PathSegmenter:
    def __init__(self, zarr_obs, zarr_action, start_timestep, end_timestep, demo_num):
        """
        Initialize the PathSegmenter with required data and parameters.
        """
        self.zarr_obs = zarr_obs
        self.zarr_action = zarr_action
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.demo_num = demo_num

        self.no_blocks = None
        self.one_block = None
        self.pivot_point = None
        self.both_blocks = None

        self.labels = None

    def get_pivot_points(self):
        """
        Pass through the entire demonstration and get key timesteps -- pivot point for the two paths, the two paths at step k. 
        """

        for step in range(self.start_timestep, self.end_timestep + 1):
            curr_obs = self.zarr_obs[step]
            curr_action = self.zarr_action[step]

            done, b0_in_target, b1_in_target = pu.is_successful(curr_obs)

            if not b0_in_target and not b1_in_target:
                assert not done

                if self.no_blocks is None:
                    self.no_blocks = step

            elif b0_in_target != b1_in_target:
                if self.one_block is None:
                    self.one_block = step

            else:
                if self.both_blocks is None:
                    self.both_blocks = step

        assert self.no_blocks is not None and self.one_block is not None and self.both_blocks is not None

        self.pivot_point = self.one_block + ((self.both_blocks - self.one_block) // 2)
        print(f"no_blocks: {self.no_blocks}, one_block: {self.one_block}, "
              f"pivot_point: {self.pivot_point}, both_blocks: {self.both_blocks}")
        

    def label_segments_from_k(self):
        """
        Given a step, return the label (path_pivot, path0_before_k, path0_at_k path0_after_k, path1_before_k, path1_at_k, path1_after_k)

        Asusmes that the pivot point and the switch step k have been found. (get_pivot_points)

        """

        assert self.no_blocks is not None and self.one_block is not None and self.both_blocks is not None and self.pivot_point is not None

        assert self.switch_step_k is not None


        self.labels = [''] * len(self.zarr_obs)

        for step in range(self.start_timestep, self.end_timestep + 1):

            curr_obs = self.zarr_obs[step]
            curr_action = self.zarr_action[step]


            if self.no_blocks <= step and step <= (self.switch_step_k + self.start_timestep):
                self.labels[step] = 'path0_before_k'
            elif (self.switch_step_k + self.start_timestep) < step and step <= self.pivot_point:
                self.labels[step] = 'path0_after_k'
            elif (self.pivot_point < step) and (step <= (self.pivot_point + self.switch_step_k)):
                self.labels[step] = 'path1_before_k'
            elif (self.pivot_point + self.switch_step_k) < step and step <= self.end_timestep:
                self.labels[step] = 'path1_after_k'
            else:
                assert False, "Step is out of documented range."

        print("End!")

    def reorder_around_k(self):

        order_map = {
            "path1_before_k": 0,
            "path0_after_k":  1,
            "path0_before_k": 2,
            "path1_after_k":  3
        }

        # # Define a function that returns the sort priority for a given index
        # def get_sort_key(i):
        #     label = self.labels[i]
        #     #  if label not in order map, just push to the end but keep same relative order
        #     return order_map.get(label, 999)
    
        # sorted_indices = sorted(range(len(self.labels)), key=get_sort_key)

        # # Reindex all your lists
        # self.labels = [self.labels[i] for i in sorted_indices]
        # self.zarr_obs = [self.zarr_obs[i] for i in sorted_indices]
        # self.zarr_action = [self.zarr_action[i] for i in sorted_indices]


        # CHEAT -- at least for the first 4 demos, don't need to reorder everything just the first 470 steps. Should save time. 

        import numpy as np

    def reorder_first_470(self):

        # Number of steps to reorder
        N = 470

        # 1) Slice out the first N items
        labels_sub  = self.labels[:N]         # these are the labels to reorder
        obs_sub     = self.zarr_obs[:N]       # shape (N, dim_obs)
        actions_sub = self.zarr_action[:N]    # shape (N, dim_action)

        # The remainder that stays untouched
        labels_rest  = self.labels[N:]
        obs_rest     = self.zarr_obs[N:]      # shape (total_steps - N, dim_obs)
        actions_rest = self.zarr_action[N:]   # shape (total_steps - N, dim_action)

        # 2) Define label priorities
        order_map = {
            "path1_before_k":  0,
            "path0_after_k":   1,
            "path0_before_k":  2,
            "path1_after_k":   3,
        }
        LOWEST_PRIORITY = 999

        def get_sort_key(label):
            # Unknown labels => 999, so they end up at the back (but preserve relative order)
            return order_map.get(label, LOWEST_PRIORITY)

        # 3) Zip the sub-block so we can sort them all together
        zipped_sub = list(zip(labels_sub, obs_sub, actions_sub))
        # Sort by label priority
        zipped_sub_sorted = sorted(zipped_sub, key=lambda x: get_sort_key(x[0]))
        # Unzip back into separate sequences
        labels_sub_sorted, obs_sub_sorted, actions_sub_sorted = zip(*zipped_sub_sorted)

        # 4) Convert obs_sub_sorted/actions_sub_sorted to NumPy arrays
        #    (each is currently a tuple of shape (N, dim_obs) or (N, dim_action))
        obs_sub_sorted     = np.array(obs_sub_sorted)
        actions_sub_sorted = np.array(actions_sub_sorted)

        # 5) Concatenate sorted sub-block with remainder
        #    along axis=0 (row-wise)
        obs_concat     = np.concatenate([obs_sub_sorted, obs_rest], axis=0)
        actions_concat = np.concatenate([actions_sub_sorted, actions_rest], axis=0)

        # 6) Update your class attributes
        #    Labels can remain as a Python list, so we just do a list concatenation
        self.labels      = list(labels_sub_sorted) + labels_rest
        # If you *need* obs and actions as lists of lists, convert to .tolist()
        self.zarr_obs    = obs_concat.tolist()
        self.zarr_action = actions_concat.tolist()

        # If you prefer keeping them as NumPy arrays:
        #   self.zarr_obs    = obs_concat
        #   self.zarr_action = actions_concat


    def color_by_time(self):
        """
        Just uses the regular color gradient and marks the pivot point in black. 
        """

        for step in range(self.start_timestep, self.end_timestep + 1):
            curr_obs = self.zarr_obs[step]
            curr_action = self.zarr_action[step]

            color = None

            if step == self.pivot_point:
                color = 'black'


            # If you don't pass in a color, it uses the gradient
            pu.plot_denoising_trajectories(curr_action, (step-self.start_timestep), self.demo_num, color)


    def color_code_3_segments(self):
        """
        Colors the pathway into 3: (start, one_block), (one_block, pivot_point), (pivot_point, both_blocks),
        and marks the pivot point.
        """
        for step in range(self.start_timestep, self.end_timestep + 1):
            curr_obs = self.zarr_obs[step]
            curr_action = self.zarr_action[step]
            if self.no_blocks <= step <= self.one_block:
                color = 'red'
            elif step == self.pivot_point:
                color = 'green'
            elif self.one_block <= step <= self.pivot_point:
                color = 'yellow'
            elif self.pivot_point <= step <= self.both_blocks:
                color = 'blue'
            else:
                assert False, "Ranges don't make sense."



            pu.plot_denoising_trajectories(curr_action, step, self.demo_num, color)

            


    
    def color_code_at_k(self):
        '''
        Colors the pathway into 4: 0_before, 0_after, 1_before, 1_after, relative to swtich_step_k.
        '''

        # path_0 = (self.start_timestep, self.pivot_point)
        # path_1 = (self.pivot_point, self.end_timestep + 1)



        for step in range(self.start_timestep, self.end_timestep + 1):
            curr_obs = self.zarr_obs[step]
            curr_action = self.zarr_action[step]

            # print(f"step: {step}, in range ({self.start_timestep}, {self.end_timestep})")
            # print(f"self.no_blocks: {self.no_blocks}")
            # print(f"self.one_block: {self.one_block}")
            # print(f"self.both_blocks: {self.both_blocks}")
            # print(f"self.pivot_point: {self.pivot_point}")
            # print(f"self.switch_step_k: {self.switch_step_k}")
            # print(f"self.pivot_point+self.switch_step_k: {self.pivot_point+self.switch_step_k}")

            # print("\n")

            # Pathway 0
            if self.no_blocks <= step and step < (self.switch_step_k + self.start_timestep):
                color = 'red'
            elif step == (self.switch_step_k + self.start_timestep):
                color = 'cyan'
            elif (self.switch_step_k + self.start_timestep) <= step and step < self.pivot_point:
                color = 'orange'
     
            # Pathway 1
            elif step == self.pivot_point:
                color = 'black'
            elif step == (self.pivot_point + self.switch_step_k):
                color = 'cyan'
            elif self.pivot_point <= step and step <= (self.pivot_point + self.switch_step_k):
                color = 'blue'
            elif (self.pivot_point + self.switch_step_k) <= step and step <= self.both_blocks:
                color = 'purple'
            else:
                assert False, "Step is out of documented range."

            pu.plot_denoising_trajectories(curr_action, step, self.demo_num, color)
    
    def chunk_path(self, switch_step_k=None, dist=None, target_num=None):
        """
        Chunk the pathway based on the pivot point and color-code the segments.

        Plots global_plots
        """

        if switch_step_k is not None:
            self.switch_step_k = switch_step_k

        self.start_obs = self.zarr_obs[self.start_timestep]
        pu.init_global_plots(self.start_obs, self.demo_num)

        self.get_pivot_points()
        self.label_segments_from_k()
        # # self.reorder_around_k()
        # self.reorder_first_470()

        # self.color_by_time()
        self.color_code_3_segments()
        # self.color_code_at_k()

        if dist is not None and target_num is not None:
            pu.plot_dist_to_target_demo(target_num, self.demo_num, dist)

        
        pu.close_global_plots(self.zarr_obs[self.end_timestep], self.demo_num, "3_segments")

        return self.labels




# 1,000 demonstrations (root['meta']['episode_ends'])


class ModifyDemos:

    def __init__(self, mode='abs'):
        # case on value of mode
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


    def get_block_positions(self, obs):
        
        block = {
            'x': obs[0],
            'y': obs[1], 
            'orientation': obs[2]
        }

        block2 = {
            'x': obs[3],
            'y': obs[4],
            'orientation': obs[5]
        }

        return (block, block2)

    def calculate_environment_similarity(self, env1, env2, method="rms"):
        """
        Computes the similarity between two environments by optimally matching the blocks
        in the two environments using the Hungarian algorithm.

        Arguments:
        env1, env2 -- Each environment consists of two blocks with (x, y, orientation).
        method -- Either 'rms' (root mean square distance) or 'mean' (average distance).

        Returns:
        A similarity score (lower is more similar).
        """
        (dist1, (block1_env1, block2_env1)) = env1

        if isinstance(env2, tuple) and len(env2) == 2:
            (block1_env2, block2_env2) = env2
        else:
            # Case 2: env2 is a tuple like (dist, (block1_env2, block2_env2))
            (dist, (block1_env2, block2_env2)) = env2

        
        # Extract positions (x, y) and orientations
        blocks_env1 = [
            {'pos': [block1_env1['x'], block1_env1['y']], 'orientation': block1_env1['orientation']},
            {'pos': [block2_env1['x'], block2_env1['y']], 'orientation': block2_env1['orientation']}
        ]
        blocks_env2 = [
            {'pos': [block1_env2['x'], block1_env2['y']], 'orientation': block1_env2['orientation']},
            {'pos': [block2_env2['x'], block2_env2['y']], 'orientation': block2_env2['orientation']}
        ]

        # Compute pairwise distances (position only)
        cost_matrix = np.array([
            [euclidean(block1['pos'], block2['pos']) for block2 in blocks_env2]
            for block1 in blocks_env1
        ])

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_distances = [cost_matrix[i, j] for i, j in zip(row_ind, col_ind)]

        if method == "rms":
            return np.sqrt(np.mean(np.square(matched_distances)))
        elif method == 'mean':
            return np.mean(matched_distances)

        return None


    def find_closest_env(self, target_env, all_envs, method='rms'):
        """

        (Helper) Returns a list of the closest environments to a single target environment.
        
        Args:
            target_env: Tuple of dictionaries representing the target environment (block1, block2). 
            all_envs: List of tuples of dictionaries representing all environments
            method: Method to calculate distance between environments. Options: 'weighted', 'rms'
        
        """

        distances = []
        for (start_timestep, env) in all_envs:

            demo_num = EPISODE_ENDS.index(start_timestep)
            
            dist = self.calculate_environment_similarity(target_env, env, method=method)
            
           
            if demo_num == 10: 
                print(f"Distance between target and {demo_num} is {dist}")

            distances.append((start_timestep, env, dist))
        
        # Sort environments by distance to the target
        distances.sort(key=lambda x: x[2])
        return distances  # Returns a list of environments sorted by proximity

    def get_closest_env(self, target_demo=0, num_demos=None):

        """
        Returns a list of all the closest environments to the target demo.

        Args
            target_demo: Index of the target demo to compare to. (default: 0)

        Returns
            target_env: Tuple of (start_timestep, env), the target environment to compare to.
            closest_env: List of (start_timestep, env, dist), the closest environments to the target demo. 

        """

        all_envs = []

        # Genearate all environments (just tracking starting block positions). Should do this outside the loop to prevent redundant calculations for all_envs.


        for start_timestep in EPISODE_ENDS[:-1]:
            obs = self.obs[start_timestep]
            env = (start_timestep, self.get_block_positions(obs))
            all_envs.append(env)

        target_env = all_envs.pop(target_demo)
        closest_env = self.find_closest_env(target_env, all_envs)

        if num_demos is not None:
            closest_env = closest_env[:num_demos]
        return (target_env, closest_env)
    
        
    def print_closest_env(self, target_demo=0, num_demos=5):

        if os.path.exists(f"global_plots"):
            shutil.rmtree(f"global_plots")
            os.makedirs(f"global_plots")


        # Plot the target demonstration
        (target_env, closest_env) = self.get_closest_env(target_demo, num_demos)

    
        (start_timestep, (target_block1, target_block2)) = target_env
        end_timestep = EPISODE_ENDS[target_demo+1]-1

        segment_path = PathSegmenter(self.obs, self.action, start_timestep, end_timestep, target_demo)
        labels = segment_path.chunk_path(switch_step_k=10, dist=0, target_num=target_demo)

        # Basically I only label between the start and end timestep.

        # pu.init_global_plots(self.obs[start_timestep], target_demo)
        # pu.plot_denoising_trajectories(self.action[start_timestep], 0, target_demo)
        # pu.close_global_plots(self.obs[end_timestep], target_demo)

        for (start_timestep, env, dist) in closest_env:

            (block1, block2) = env

            # NOTE: I should just make them all relative indexes but this is a brute force solution

            

            demo_num = EPISODE_ENDS.index(start_timestep)
            end_timestep = EPISODE_ENDS[demo_num+1]-1
            print(f"Demo {demo_num} is {dist} away from target demo {target_demo}.")

            segment_path = PathSegmenter(self.obs, self.action, start_timestep, end_timestep, demo_num)
            labels = segment_path.chunk_path(switch_step_k=10, dist=dist, target_num=target_demo)
            print("Labels")
        

            # Let's create a new path based off target and closest envs

            

    

            
    
            
   


def main():

    demos = ModifyDemos()
    demos.print_closest_env(target_demo=0, num_demos=5)


    # print(os.getcwd())
    # zarr_abs = zarr.open("multimodal_push_seed_abs.zarr", mode='r')
    # zarr_rel = zarr.open("multimodal_push_seed.zarr", mode='r')

    # zarr_obs = zarr_rel['data']['obs']
    # zarr_action = zarr_abs['data']['action']

    # if os.path.exists('global_plots'):
    #     shutil.rmtree('global_plots')
    #     os.makedirs('global_plots')


    # for demo_num in range(1):
    #     start_timestep = EPISODE_ENDS[demo_num]
    #     end_timestep = EPISODE_ENDS[demo_num+1] -1

    #     print(f"Demo {demo_num}: {start_timestep} to {end_timestep}")

    #     segmenter = PathSegmenter(zarr_obs, zarr_action, start_timestep, end_timestep, demo_num)
    #     segmenter.chunk_path(switch_step_k=10)



   
    

if __name__ == "__main__":
    main()



