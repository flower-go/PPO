ALL_LAYOUTS = ["forced_coordination", "cramped_room", "coordination_ring", "asymmetric_advantages", "counter_circuit_o_1order","counter_circuit_o_1order"]

class ExperimentsParamsManager(object):
    def __init__(self,args):
        self.args = args
        self.args["layout_name"] = "cramped_room"
        self.args["num_workers"] = 30
        self.args["action_prob_diff_reward_coef"] = 0
        self.args["eval_interval"] = 10
        self.args["evals_num_to_threshold"] = 2
        self.args["training_percent_start_eval"] = 0.2
        self.args["device"] = "cuda"
        self.args["divergent_check_timestep"] = 3.5e6
        self.args["rnd_obj_prob_thresh"] = 0.0
        self.args["random_start"] = True

        self.init_base_args_for_layout(args["layout_name"])



    def init_base_args_for_layout(self, layout):
        if layout == "coordination_ring":
            # self.args["divergent_check_timestep"] = 1e6
            self.args["sparse_r_coef_horizon"] = 2.5e6 #TODO: check if not colliding with other experiments
            self.args["eval_stop_threshold"] = 185 #140  #160

        if layout == "cramped_room":
            # self.args["divergent_check_timestep"] = 3e5
            self.args["eval_stop_threshold"] = 210 #195
            self.args["sparse_r_coef_horizon"] = 5e6

        if layout == "asymmetric_advantages":
            # self.args["divergent_check_timestep"] = 3e18
            self.args["eval_stop_threshold"] = 205
            self.args["sparse_r_coef_horizon"] = 2.5e6

        if layout == "forced_coordination":
            # self.args["divergent_check_timestep"] = 3e18
            self.args["eval_stop_threshold"] = 165 #150
            self.args["sparse_r_coef_horizon"] = 5.5e6

        if layout == "counter_circuit_o_1order":
            # self.args["divergent_check_timestep"] = 3e18
            self.args["eval_stop_threshold"] = 123
            self.args["sparse_r_coef_horizon"] = 3.5e6
            self.args["divergent_check_timestep"] = 2.6e6

        if layout == "counter_circuit":
            # self.args["divergent_check_timestep"] = 3e18
            self.args["eval_stop_threshold"] = 93
            self.args["sparse_r_coef_horizon"] = 3.5e6


    def init_exp_specific_args(self, exp):
        self.args["exp"] = exp
        fn_name = "set_" + self.args["layout_name"] + "_" + self.args["exp"]
        fn = getattr(self, fn_name)
        fn()




    def set_cramped_room_MLP_CUDA_TEST(self):
        self.args["exp"] = "MLP_CUDA_TEST"
        self.args["random_start"] = False
        self.args["ent_coef_start"] = 0.2
        self.args["ent_coef_end"] = 0.005
        self.args["trained_models"] = 1
        self.args["ent_coef_horizon"] = 1e6
        self.args["total_timesteps"] = 4.6e6
        self.args["vf_coef"] = 0.01
        self.args["batch_size"] = 4000
        self.args["device"] = "cuda"
        self.args["max_grad_norm"] = 0.1
        self.args["clip_range"] = 0.05
        self.args["learning_rate"] = 0.001
        self.args["n_steps"] = 400
        self.args["n_epochs"] = 8
        self.args["sparse_r_coef_horizon"] = 2.5e6



    def set_cramped_room_MLP_CUDA_RS(self):
        self.args["exp"] = "MLP_CUDA_RS"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.1
        self.args["trained_models"] = 1
        self.args["ent_coef_horizon"] = 2e6
        self.args["total_timesteps"] = 4e6
        self.args["vf_coef"] = 0.01
        self.args["batch_size"] = 2000
        self.args["device"] = "cuda"
        self.args["max_grad_norm"] = 0.1
        self.args["clip_range"] = 0.05
        self.args["learning_rate"] = 0.001
        self.args["n_steps"] = 400
        self.args["n_epochs"] = 8
        self.args["sparse_r_coef_horizon"] = 2.5e6

    def set_cramped_room_CNN_CUDA_RS(self):
        self.args["exp"] = "CNN_CUDA_RS"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.03
        self.args["trained_models"] = 30
        self.args["ent_coef_horizon"] = 1.5e6
        self.args["total_timesteps"] = 5.5e6
        self.args["vf_coef"] = 0.1
        self.args["batch_size"] = 2000
        self.args["device"] = "cuda"
        self.args["max_grad_norm"] = 0.3
        self.args["clip_range"] = 0.1
        self.args["learning_rate"] = 0.0004
        self.args["n_steps"] = 400
        self.args["n_epochs"] = 8
        self.args["sparse_r_coef_horizon"] = 2.5e6

    def set_forced_coordination_MLP_CUDA_RS(self):
        self.args["exp"] = "MLP_CUDA_RS"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.2
        self.args["ent_coef_end"] = 0.005
        self.args["trained_models"] = 5
        self.args["ent_coef_horizon"] = 4e6
        self.args["total_timesteps"] = 10e6
        self.args["vf_coef"] = 0.1
        self.args["batch_size"] = 4000
        self.args["device"] = "cuda"
        self.args["max_grad_norm"] = 0.1
        self.args["clip_range"] = 0.05
        self.args["learning_rate"] = 0.001
        self.args["n_steps"] = 400
        self.args["n_epochs"] = 8
        self.args["sparse_r_coef_horizon"] = 10e6

    def set_forced_coordination_CNN_CUDA_RS(self):
        self.args["exp"] = "CNN_CUDA_RS"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.03
        self.args["trained_models"] = 30
        self.args["ent_coef_horizon"] = 1.5e6
        self.args["total_timesteps"] = 5.5e6
        self.args["vf_coef"] = 0.1
        self.args["batch_size"] = 2000
        self.args["device"] = "cuda"
        self.args["max_grad_norm"] = 0.3
        self.args["clip_range"] = 0.1
        self.args["learning_rate"] = 0.0004
        self.args["n_steps"] = 400
        self.args["n_epochs"] = 8
        self.args["sparse_r_coef_horizon"] = 2.5e6
        self.args["divergent_check_timestep"] = 3e6




    def set_coordination_ring_MLP_CUDA_RS(self):
        self.args["exp"] = "MLP_CUDA_RS"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.2
        self.args["ent_coef_end"] = 0.005
        self.args["trained_models"] = 15
        self.args["ent_coef_horizon"] = 4e6
        self.args["total_timesteps"] = 10e6
        self.args["vf_coef"] = 0.1
        self.args["batch_size"] = 4000
        self.args["device"] = "cuda"
        self.args["max_grad_norm"] = 0.1
        self.args["clip_range"] = 0.05
        self.args["learning_rate"] = 0.001
        self.args["n_steps"] = 400
        self.args["n_epochs"] = 8
        self.args["sparse_r_coef_horizon"] = 10e6

    def set_coordination_ring_CNN_CUDA_RS(self):
        self.args["exp"] = "CNN_CUDA_RS"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.03# 0.02
        self.args["trained_models"] = 30
        self.args["ent_coef_horizon"] = 1.5e6#0.95e6
        self.args["total_timesteps"] = 5.5e6
        self.args["vf_coef"] = 0.1#0.15
        self.args["batch_size"] = 2000
        self.args["device"] = "cuda"
        self.args["max_grad_norm"] = 0.3 #0.35
        self.args["clip_range"] = 0.1# 0.14
        self.args["learning_rate"] = 0.0004
        self.args["n_steps"] = 400
        self.args["n_epochs"] = 8
        self.args["sparse_r_coef_horizon"] = 2.5e6

    def set_coordination_ring_CNN_CUDA_RS_TEST(self):
        self.args["exp"] = "CNN_CUDA_RS_TEST"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.03# 0.02
        self.args["trained_models"] = 3
        self.args["ent_coef_horizon"] = 1.5e6#0.95e6
        self.args["total_timesteps"] = 4e6
        self.args["vf_coef"] = 0.1#0.15
        self.args["batch_size"] = 2000
        self.args["device"] = "cuda"
        self.args["max_grad_norm"] = 0.3 #0.35
        self.args["clip_range"] = 0.1# 0.14
        self.args["learning_rate"] = 0.0004
        self.args["n_steps"] = 400
        self.args["n_epochs"] = 8
        self.args["sparse_r_coef_horizon"] = 2.5e6

    def set_asymmetric_advantages_MLP_CUDA_RS(self):
        self.args["exp"] = "MLP_CUDA_RS"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.2
        self.args["ent_coef_end"] = 0.005
        self.args["trained_models"] = 15
        self.args["ent_coef_horizon"] = 4e6
        self.args["total_timesteps"] = 10e6
        self.args["vf_coef"] = 0.1
        self.args["batch_size"] = 4000
        self.args["device"] = "cuda"
        self.args["max_grad_norm"] = 0.1
        self.args["clip_range"] = 0.05
        self.args["learning_rate"] = 0.001
        self.args["n_steps"] = 400
        self.args["n_epochs"] = 8
        self.args["sparse_r_coef_horizon"] = 10e6

    def set_asymmetric_advantages_CNN_CUDA_RS(self):
        self.args["exp"] = "CNN_CUDA_RS"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.03
        self.args["trained_models"] = 30
        self.args["ent_coef_horizon"] = 1.5e6
        self.args["total_timesteps"] = 5.5e6
        self.args["vf_coef"] = 0.1
        self.args["batch_size"] = 2000
        self.args["device"] = "cuda"
        self.args["max_grad_norm"] = 0.3
        self.args["clip_range"] = 0.1
        self.args["learning_rate"] = 0.0004
        self.args["n_steps"] = 400
        self.args["n_epochs"] = 8
        self.args["sparse_r_coef_horizon"] = 2.5e6

    def set_counter_circuit_o_1order_CNN_CUDA_RS(self):
        self.args["exp"] = "CNN_CUDA_RS"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.03
        self.args["trained_models"] = 30
        self.args["ent_coef_horizon"] = 1.5e6
        self.args["total_timesteps"] = 5.5e6
        self.args["vf_coef"] = 0.1
        self.args["batch_size"] = 2000
        self.args["device"] = "cuda"
        self.args["max_grad_norm"] = 0.3
        self.args["clip_range"] = 0.1
        self.args["learning_rate"] = 0.0003
        self.args["n_steps"] = 400
        self.args["n_epochs"] = 8
        self.args["sparse_r_coef_horizon"] = 2.5e6


