class ExperimentsParamsHolder(object):
    def __init__(self,args):
        self.args = args
        self.args["layout_name"] = "cramped_room"
        self.args["num_workers"] = 8

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
