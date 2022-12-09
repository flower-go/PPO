ALL_LAYOUTS = ["cramped_room", "forced_coordination", "asymmetric_advantages", "coordination_ring", "counter_circuit"]

class ExperimentsParamsManager(object):
    def __init__(self,args):
        self.args = args
        self.args["layout_name"] = "cramped_room"
        self.args["num_workers"] = 8
        self.args["action_prob_diff_reward_coef"] = 0
        self.args["eval_interval"] = 5
        self.args["evals_num_to_threshold"] = 4
        self.args["training_percent_start_eval"] = 0.0
        self.args["device"] = "cpu"
        self.args["divergent_check_timestep"] = 10e6
        self.args["rnd_obj_prob_thresh"] = 0.0
        self.args["random_start"] = True

        self.init_base_args_for_layout(args["layout_name"])



    def init_base_args_for_layout(self, layout):
        if layout == "coordination_ring":
            # self.args["divergent_check_timestep"] = 1e6
            self.args["sparse_r_coef_horizon"] = 6e6 #TODO: check if not colliding with other experiments
            self.args["eval_stop_threshold"] = 183

        if layout == "cramped_room":
            # self.args["divergent_check_timestep"] = 3e5
            self.args["eval_stop_threshold"] = 223
            self.args["sparse_r_coef_horizon"] = 6e6

        if layout == "asymmetric_advantages":
            # self.args["divergent_check_timestep"] = 3e18
            self.args["eval_stop_threshold"] = 183
            self.args["sparse_r_coef_horizon"] = 7e6

        if layout == "forced_coordination":
            # self.args["divergent_check_timestep"] = 3e18
            self.args["eval_stop_threshold"] = 113
            self.args["sparse_r_coef_horizon"] = 6e6

        if layout == "counter_circuit":
            # self.args["divergent_check_timestep"] = 3e18
            self.args["eval_stop_threshold"] = 93
            self.args["sparse_r_coef_horizon"] = 6e6

    def init_exp_specific_args(self, exp):
        self.args["exp"] = exp
        fn_name = "set_" + self.args["layout_name"] + "_" + self.args["exp"]
        fn = getattr(self, fn_name)
        fn()

    def set_coordination_ring_SP_E0_1_Drop(self):
        self.args["exp"] = "SP_E0_1_Drop"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.00
        self.args["trained_models"] = 6
        self.args["ent_coef_horizon"] = 10e6
        self.args["total_timesteps"] = 10e6

    def set_asymmetric_advantages_SP_E0_1_Drop(self):
        self.args["exp"] = "SP_E0_1_Drop"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.00
        self.args["trained_models"] = 6
        self.args["ent_coef_horizon"] = 10e6
        self.args["total_timesteps"] = 10e6

    def set_forced_coordination_SP_E0_1_Drop_Arti(self):
        self.args["exp"] = "SP_E0_1_Drop_Arti"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.00
        self.args["trained_models"] = 6
        self.args["ent_coef_horizon"] = 10e6
        self.args["total_timesteps"] = 10e6

    def set_forced_coordination_SP_E0_1_Drop(self):
        self.args["exp"] = "SP_E0_1_Drop"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.00
        self.args["trained_models"] = 6
        self.args["ent_coef_horizon"] = 10e6
        self.args["total_timesteps"] = 10e6

    def set_counter_circuit_SP_E0_1_Drop(self):
        self.args["exp"] = "SP_E0_1_Drop"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.00
        self.args["trained_models"] = 6
        self.args["ent_coef_horizon"] = 10e6
        self.args["total_timesteps"] = 10e6

    def set_cramped_room_SP_E0_1_Drop(self):
        self.args["exp"] = "SP_E0_1_Drop"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.00
        self.args["trained_models"] = 6
        self.args["ent_coef_horizon"] = 10e6
        self.args["total_timesteps"] = 10e6

    def set_cramped_room_CNN_SP_E0_1_Drop(self):
        self.args["exp"] = "CNN_SP_E0_1_Drop"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.00
        self.args["trained_models"] = 6
        self.args["ent_coef_horizon"] = 10e6
        self.args["total_timesteps"] = 10e6

    def set_cramped_room_CNN_SP_E0_1_Artic_Params(self):
        self.args["exp"] = "CNN_SP_E0_1_Artic_Params"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.1
        self.args["trained_models"] = 6
        self.args["ent_coef_horizon"] = 10e6
        self.args["total_timesteps"] = 10e6

    def set_cramped_room_CNN_SP_E0_03(self):
        self.args["exp"] = "CNN_SP_E0_03"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.05
        self.args["ent_coef_end"] = 0.05
        self.args["trained_models"] = 6
        self.args["ent_coef_horizon"] = 10e6
        self.args["total_timesteps"] = 10e6

    def set_cramped_room_CNN_SP_E0_05(self):
        self.args["exp"] = "CNN_SP_E0_05"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.05
        self.args["ent_coef_end"] = 0.05
        self.args["trained_models"] = 6
        self.args["ent_coef_horizon"] = 10e6
        self.args["total_timesteps"] = 10e6

    def set_cramped_room_CNN_SP_E0_01(self):
        self.args["exp"] = "CNN_SP_E0_01"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.01
        self.args["ent_coef_end"] = 0.01
        self.args["trained_models"] = 6
        self.args["ent_coef_horizon"] = 10e6
        self.args["total_timesteps"] = 10e6


    def set_default_population_BU_RS(self):
        self.args["exp"] = "default_population_BU"
        self.args["action_prob_diff_reward_coef"] = 0.0001
        self.args["ent_coef_start"] = 0.05
        self.args["ent_coef_end"] = 0.00
        self.args["trained_models"] = 6
        self.args["ent_coef_horizon"] = 4.1e6
        self.args["total_timesteps"] = 8e6

    def set_default_SP(self):
        self.args["exp"] = "default_SP"
        self.args["ent_coef_start"] = 0.05
        self.args["ent_coef_end"] = 0.00
        self.args["trained_models"] = 6
        self.args["ent_coef_horizon"] = 4.1e6
        self.args["total_timesteps"] = 1e7
        self.args["sparse_r_coef_horizon"] = 2e6

    def set_default_SP_RS(self):
        self.args["exp"] = "default_SP_RS"
        self.args["ent_coef_start"] = 0.05
        self.args["ent_coef_end"] = 0.00
        self.args["trained_models"] = 6
        self.args["ent_coef_horizon"] = 4.1e6
        self.args["total_timesteps"] = 1e7
        self.args["sparse_r_coef_horizon"] = 2e6

    def set_default_SP_debug(self):
        self.args["exp"] = "default_SP_debug"
        self.args["ent_coef_start"] = 0.05
        self.args["ent_coef_end"] = 0.00
        self.args["trained_models"] = 6
        self.args["ent_coef_horizon"] = 4.1e6
        self.args["total_timesteps"] = 1e7
        self.args["sparse_r_coef_horizon"] = 2e6

    def set_pretrained_entropy_SP(self):
        self.args["exp"] = "pretrained_entropy_SP"
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.00
        self.args["trained_models"] = 10
        self.args["ent_coef_horizon"] = 2e6
        self.args["total_timesteps"] = 3e6


    def set_SP_RS_E0(self):
        self.args["exp"] = "SP_RS_E0"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.0
        self.args["ent_coef_end"] = 0.0
        self.args["trained_models"] = 7
        self.args["ent_coef_horizon"] = 3e6
        self.args["total_timesteps"] = 3e6
        self.args["sparse_r_coef_horizon"] = 1e6

    def set_SP_RS_E0_01(self):
        self.args["exp"] = "SP_RS_E0_01"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.01
        self.args["ent_coef_end"] = 0.01
        self.args["trained_models"] = 5
        self.args["ent_coef_horizon"] = 3e6
        self.args["total_timesteps"] = 3e6
        self.args["sparse_r_coef_horizon"] = 1e6

    def set_SP_RS_E0_01_Drop(self):
        self.args["exp"] = "SP_RS_E0_01_Drop"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.01
        self.args["ent_coef_end"] = 0.00
        self.args["trained_models"] = 8
        self.args["ent_coef_horizon"] = 2e6
        self.args["total_timesteps"] = 6e6
        self.args["sparse_r_coef_horizon"] = 1e6

    def set_SP_RS_E0_02(self):
        self.args["exp"] = "SP_RS_E0_02"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.02
        self.args["ent_coef_end"] = 0.02
        self.args["trained_models"] = 5
        self.args["ent_coef_horizon"] = 3e6
        self.args["total_timesteps"] = 3e6
        self.args["sparse_r_coef_horizon"] = 1e6

    def set_SP_RS_E0_03(self):
        self.args["exp"] = "SP_RS_E0_03"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.03
        self.args["ent_coef_end"] = 0.03
        self.args["trained_models"] = 5
        self.args["ent_coef_horizon"] = 3e6
        self.args["total_timesteps"] = 3e6
        self.args["sparse_r_coef_horizon"] = 1e5

    def set_SP_RS_E0_05(self):
        self.args["exp"] = "SP_RS_E0_05"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.05
        self.args["ent_coef_end"] = 0.05
        self.args["trained_models"] = 5
        self.args["ent_coef_horizon"] = 3e6
        self.args["total_timesteps"] = 3e6
        self.args["sparse_r_coef_horizon"] = 1e6

    def set_SP_RS_E0_05_Drop(self):
        self.args["exp"] = "SP_RS_E0_05_Drop"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.05
        self.args["ent_coef_end"] = 0.00
        self.args["trained_models"] = 10
        self.args["ent_coef_horizon"] = 4e6
        self.args["total_timesteps"] = 6e6
        self.args["sparse_r_coef_horizon"] = 1e6