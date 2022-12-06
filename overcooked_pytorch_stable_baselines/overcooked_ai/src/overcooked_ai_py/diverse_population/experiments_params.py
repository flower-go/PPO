class ExperimentsParamsManager(object):
    def __init__(self,args):
        self.args = args
        self.args["layout_name"] = "cramped_room"
        self.args["num_workers"] = 8
        self.args["action_prob_diff_reward_coef"] = 0
        self.args["eval_interval"] = 30
        self.args["device"] = "cpu"
        self.args["divergent_check_timestep"] = 3e5
        self.args["rnd_obj_prob_thresh"] = 0.2

    def set_default_population_BU_RS(self):
        self.args["mode"] = "default_population_BU"
        self.args["action_prob_diff_reward_coef"] = 0.0001
        self.args["ent_coef_start"] = 0.05
        self.args["ent_coef_end"] = 0.00
        self.args["trained_models"] = 9
        self.args["ent_coef_horizon"] = 4.1e6
        self.args["total_timesteps"] = 8e6

    def set_default_SP(self):
        self.args["mode"] = "default_SP"
        self.args["ent_coef_start"] = 0.05
        self.args["ent_coef_end"] = 0.00
        self.args["trained_models"] = 6
        self.args["ent_coef_horizon"] = 4.1e6
        self.args["total_timesteps"] = 1e7
        self.args["sparse_r_coef_horizon"] = 2e6

    def set_default_SP_RS(self):
        self.args["mode"] = "default_SP_RS"
        self.args["ent_coef_start"] = 0.05
        self.args["ent_coef_end"] = 0.00
        self.args["trained_models"] = 6
        self.args["ent_coef_horizon"] = 4.1e6
        self.args["total_timesteps"] = 1e7
        self.args["sparse_r_coef_horizon"] = 2e6

    def set_default_SP_debug(self):
        self.args["mode"] = "default_SP_debug"
        self.args["ent_coef_start"] = 0.05
        self.args["ent_coef_end"] = 0.00
        self.args["trained_models"] = 6
        self.args["ent_coef_horizon"] = 4.1e6
        self.args["total_timesteps"] = 1e7
        self.args["sparse_r_coef_horizon"] = 2e6

    def set_pretrained_entropy_SP(self):
        self.args["mode"] = "pretrained_entropy_SP"
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.00
        self.args["trained_models"] = 10
        self.args["ent_coef_horizon"] = 2e6
        self.args["total_timesteps"] = 3e6


    def set_SP_RS_E0(self):
        self.args["mode"] = "SP_RS_E0"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.0
        self.args["ent_coef_end"] = 0.0
        self.args["trained_models"] = 7
        self.args["ent_coef_horizon"] = 3e6
        self.args["total_timesteps"] = 3e6
        self.args["sparse_r_coef_horizon"] = 1e6

    def set_SP_RS_E0_01(self):
        self.args["mode"] = "SP_RS_E0_01"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.01
        self.args["ent_coef_end"] = 0.01
        self.args["trained_models"] = 5
        self.args["ent_coef_horizon"] = 3e6
        self.args["total_timesteps"] = 3e6
        self.args["sparse_r_coef_horizon"] = 1e6

    def set_SP_RS_E0_01_Drop(self):
        self.args["mode"] = "SP_RS_E0_01_Drop"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.01
        self.args["ent_coef_end"] = 0.00
        self.args["trained_models"] = 6
        self.args["ent_coef_horizon"] = 2e6
        self.args["total_timesteps"] = 6e6
        self.args["sparse_r_coef_horizon"] = 1e6

    def set_SP_RS_E0_02(self):
        self.args["mode"] = "SP_RS_E0_02"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.02
        self.args["ent_coef_end"] = 0.02
        self.args["trained_models"] = 5
        self.args["ent_coef_horizon"] = 3e6
        self.args["total_timesteps"] = 3e6
        self.args["sparse_r_coef_horizon"] = 1e6

    def set_SP_RS_E0_03(self):
        self.args["mode"] = "SP_RS_E0_03"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.03
        self.args["ent_coef_end"] = 0.03
        self.args["trained_models"] = 5
        self.args["ent_coef_horizon"] = 3e6
        self.args["total_timesteps"] = 3e6
        self.args["sparse_r_coef_horizon"] = 1e5

    def set_SP_RS_E0_05(self):
        self.args["mode"] = "SP_RS_E0_05"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.05
        self.args["ent_coef_end"] = 0.05
        self.args["trained_models"] = 5
        self.args["ent_coef_horizon"] = 3e6
        self.args["total_timesteps"] = 3e6
        self.args["sparse_r_coef_horizon"] = 1e6

    def set_SP_RS_E0_05_Drop(self):
        self.args["mode"] = "SP_RS_E0_05_Drop"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.05
        self.args["ent_coef_end"] = 0.00
        self.args["trained_models"] = 5
        self.args["ent_coef_horizon"] = 1.5e6
        self.args["total_timesteps"] = 3e6
        self.args["sparse_r_coef_horizon"] = 1e6