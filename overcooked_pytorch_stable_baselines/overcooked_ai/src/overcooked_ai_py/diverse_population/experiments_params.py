ALL_LAYOUTS = ["forced_coordination", "cramped_room", "coordination_ring", "asymmetric_advantages", "counter_circuit_o_1order"]

class ExperimentsParamsManager(object):
    def __init__(self,args):
        self.args = args
        self.args["layout_name"] = "cramped_room"
        self.args["num_workers"] = 15
        self.args["action_prob_diff_reward_coef"] = 0
        self.args["eval_interval"] = 10
        self.args["evals_num_to_threshold"] = 4
        self.args["training_percent_start_eval"] = 0.2
        self.args["device"] = "cuda"
        self.args["divergent_check_timestep"] = 4.5e6
        self.args["rnd_obj_prob_thresh"] = 0.0
        self.args["random_start"] = True

        self.init_base_args_for_layout(args["layout_name"])



    def init_base_args_for_layout(self, layout):
        if layout == "coordination_ring":
            # self.args["divergent_check_timestep"] = 1e6
            self.args["sparse_r_coef_horizon"] = 8e6 #TODO: check if not colliding with other experiments
            self.args["eval_stop_threshold"] = 203

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
            self.args["eval_stop_threshold"] = 140
            self.args["sparse_r_coef_horizon"] = 8e6

        if layout == "counter_circuit_o_1order":
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
        self.args["trained_models"] = 1
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

    # This settings start kind of working
    # after cca 4M steps found solution:                         batch_size=200, n_steps=400, vf_coef=0.0001,         self.args["num_workers"] = 8
    # base settings for finding better args
    def set_cramped_room_CNN_SP_SMALL_NN(self):
        self.args["exp"] = "CNN_SP_SMALL_NN"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.0005
        self.args["trained_models"] = 1
        self.args["ent_coef_horizon"] = 10e6
        self.args["total_timesteps"] = 0.4e5
        self.args["vf_coef"] = 0.01 # First modification, originally 0.0001
        self.args["batch_size"] = 200
        self.args["max_grad_norm"] = 0.5
        self.args["clip_range"] = 0.2
        self.args["n_steps"] = 400
        self.args["learning_rate"] = 3e-4
        # vf_coef = 0.0001
        #no other PPO modifications
        # 5x5(2), 3x3(1), 3x3(0), 32D, 32D

    def set_cramped_room_CNN_SP_NORMAL_NN(self):
        self.args["exp"] = "CNN_SP_NORMAL_NN"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.0005
        self.args["trained_models"] = 1
        self.args["ent_coef_horizon"] = 10e6
        self.args["total_timesteps"] = 4e6
        self.args["vf_coef"] = 0.01 # First modification, originally 0.0001
        self.args["batch_size"] = 200
        self.args["max_grad_norm"] = 0.5
        self.args["clip_range"] = 0.2
        self.args["n_steps"] = 400
        self.args["learning_rate"] = 3e-4

    def set_cramped_room_CNN_SP_NORMAL_NN_64(self):
        self.args["exp"] = "CNN_SP_NORMAL_NN_64"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.0005
        self.args["trained_models"] = 1
        self.args["ent_coef_horizon"] = 10e6
        self.args["total_timesteps"] = 4e6
        self.args["vf_coef"] = 0.01 # First modification, originally 0.0001
        self.args["batch_size"] = 200
        self.args["max_grad_norm"] = 0.5
        self.args["clip_range"] = 0.2
        self.args["n_steps"] = 400
        self.args["learning_rate"] = 3e-4

    def set_coordination_ring_CNN_SP_NORMAL_NN_64(self):
        self.args["exp"] = "CNN_SP_NORMAL_NN_64"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.0005
        self.args["trained_models"] = 1
        self.args["ent_coef_horizon"] = 10e6
        self.args["total_timesteps"] = 2.5e6
        self.args["vf_coef"] = 0.01 # First modification, originally 0.0001
        self.args["batch_size"] = 200
        self.args["max_grad_norm"] = 0.5
        self.args["clip_range"] = 0.2
        self.args["n_steps"] = 400
        self.args["learning_rate"] = 3e-4

    def set_cramped_room_CNN_SP_NORMAL_NN_RAND(self):
        self.args["exp"] = "CNN_SP_NORMAL_NN_RAND"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.0005
        self.args["trained_models"] = 1
        self.args["ent_coef_horizon"] = 10e6
        self.args["total_timesteps"] = 3e6
        self.args["vf_coef"] = 0.01 # First modification, originally 0.0001
        self.args["batch_size"] = 200
        self.args["max_grad_norm"] = 0.5
        self.args["clip_range"] = 0.2
        self.args["n_steps"] = 400
        self.args["learning_rate"] = 3e-4


    def set_cramped_room_CNN_SP_SMALL_NN_WORKERS(self):
        self.args["exp"] = "CNN_SP_SMALL_NN_WORKERS"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.0005
        self.args["trained_models"] = 1
        self.args["ent_coef_horizon"] = 10e6
        self.args["total_timesteps"] = 4e6
        self.args["vf_coef"] = 0.01 # First modification, originally 0.0001
        self.args["batch_size"] = 200
        self.args["max_grad_norm"] = 0.5
        self.args["clip_range"] = 0.2
        self.args["n_steps"] = 400
        self.args["learning_rate"] = 3e-4

    # (20 workers) For Batch_sizes 2000 or 1000 significant speed up in terms of number of env steps
    # however it does not seem to be learning
    # for Batch_size 500 small speed up and still learning
    def set_cramped_room_CNN_SP_SMALL_NN_CUDA(self):
        self.args["exp"] = "CNN_SP_SMALL_NN_CUDA"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.0005
        self.args["trained_models"] = 1
        self.args["ent_coef_horizon"] = 10e6
        self.args["total_timesteps"] = 4e6
        self.args["vf_coef"] = 0.001 # First modification, originally 0.0001
        self.args["batch_size"] = 2400
        self.args["device"] = "cuda"
        self.args["max_grad_norm"] = 0.5
        self.args["clip_range"] = 0.2
        self.args["n_steps"] = 2400
        self.args["learning_rate"] = 3e-4

    def set_cramped_room_MLP_CPU_RS(self):
        self.args["exp"] = "MLP_CPU_RS"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.1
        self.args["trained_models"] = 1
        self.args["ent_coef_horizon"] = 2e6
        self.args["total_timesteps"] = 4e6
        self.args["vf_coef"] = 0.5
        self.args["batch_size"] = 2000
        self.args["device"] = "cuda"
        self.args["max_grad_norm"] = 0.5
        self.args["clip_range"] = 0.2
        self.args["learning_rate"] = 0.001
        self.args["n_steps"] = 400
        self.args["n_epochs"] = 8
        self.args["sparse_r_coef_horizon"] = 2.5e6

    def set_cramped_room_CNN_CUDA_RS(self):
        self.args["exp"] = "CNN_CUDA_RS"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.01
        self.args["trained_models"] = 1
        self.args["ent_coef_horizon"] = 2e6
        self.args["total_timesteps"] = 4e6
        self.args["vf_coef"] = 0.5
        self.args["batch_size"] = 600
        self.args["device"] = "cuda"
        self.args["max_grad_norm"] = 0.5
        self.args["clip_range"] = 0.2
        self.args["learning_rate"] = 0.001
        self.args["n_steps"] = 400
        self.args["n_epochs"] = 8
        self.args["sparse_r_coef_horizon"] = 2.5e6

    def set_forced_coordination_CNN_CUDA_RS(self):
        self.args["exp"] = "CNN_CUDA_RS"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.1
        self.args["trained_models"] = 6
        self.args["ent_coef_horizon"] = 0.8e6
        self.args["total_timesteps"] = 2.1e6
        self.args["vf_coef"] = 0.08
        self.args["batch_size"] = 2000
        self.args["device"] = "cuda"
        self.args["max_grad_norm"] = 0.1
        self.args["clip_range"] = 0.2
        self.args["learning_rate"] = 0.001
        self.args["n_steps"] = 400
        self.args["n_epochs"] = 12
        self.args["sparse_r_coef_horizon"] = 3.5e6

    def set_forced_coordination_CNN_CUDA_RS_EVAL(self):
        self.args["exp"] = "CNN_CUDA_RS_EVAL"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.2
        self.args["ent_coef_end"] = 0.0005
        self.args["trained_models"] = 10
        self.args["ent_coef_horizon"] = 0.3e6
        self.args["total_timesteps"] = 9e6
        self.args["vf_coef"] = 0.08
        self.args["batch_size"] = 2000
        self.args["device"] = "cuda"
        self.args["max_grad_norm"] = 0.1
        self.args["clip_range"] = 0.2
        self.args["learning_rate"] = 0.001
        self.args["n_steps"] = 1200
        self.args["n_epochs"] = 8
        self.args["sparse_r_coef_horizon"] = 2.5e6



    #4 out of 5
    # Works also with smaller net (32x32x32)
    def set_forced_coordination_CNN_CUDA_RS_WORKS(self):
        self.args["exp"] = "CNN_CUDA_RS_WORKS"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.2
        self.args["ent_coef_end"] = 0.0005
        self.args["trained_models"] = 9
        self.args["ent_coef_horizon"] = 0.8e6 #1.2e6
        self.args["total_timesteps"] = 8e6
        self.args["vf_coef"] = 0.08
        self.args["batch_size"] = 9600
        self.args["device"] = "cuda"
        self.args["max_grad_norm"] = 0.1
        self.args["clip_range"] = 0.2
        self.args["learning_rate"] = 0.001
        self.args["n_steps"] = 1200
        self.args["n_epochs"] = 12
        self.args["sparse_r_coef_horizon"] = 4e6


    # worked 4.5 out of 5 cases (5th attempt probably caught up at very last moment)
    def set_forced_coordination_CNN_CUDA_WORKS2(self):
        self.args["exp"] = "CNN_CUDA"
        self.args["random_start"] = False
        self.args["ent_coef_start"] = 0.2
        self.args["ent_coef_end"] = 0.0005
        self.args["trained_models"] = 5
        self.args["ent_coef_horizon"] = 1.6e6
        self.args["total_timesteps"] = 2.5e6
        self.args["vf_coef"] = 0.08
        self.args["batch_size"] = 9600
        self.args["device"] = "cuda"
        self.args["max_grad_norm"] = 0.1
        self.args["clip_range"] = 0.2
        self.args["learning_rate"] = 0.001
        self.args["n_steps"] = 1200
        self.args["n_epochs"] = 12
        self.args["sparse_r_coef_horizon"] = 3.5e6

    def set_forced_coordination_CNN_CUDA_WORKS(self):
        self.args["exp"] = "CNN_CUDA"
        self.args["random_start"] = False
        self.args["ent_coef_start"] = 0.2
        self.args["ent_coef_end"] = 0.0005
        self.args["trained_models"] = 5
        self.args["ent_coef_horizon"] = 1.1e6
        self.args["total_timesteps"] = 4e6
        self.args["vf_coef"] = 0.08
        self.args["batch_size"] = 6400
        self.args["device"] = "cuda"
        self.args["max_grad_norm"] = 0.1
        self.args["clip_range"] = 0.2
        self.args["learning_rate"] = 0.001
        self.args["n_steps"] = 800
        self.args["n_epochs"] = 12
        self.args["sparse_r_coef_horizon"] = 3.5e6
        self.args["num_workers"] = 32

    def set_coordination_ring_CNN_CUDA_RS(self):
        self.args["exp"] = "CNN_CUDA_RS"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.0005
        self.args["trained_models"] = 5
        self.args["ent_coef_horizon"] = 3e6
        self.args["total_timesteps"] = 3e6
        self.args["vf_coef"] = 0.05
        self.args["batch_size"] = 1600
        self.args["device"] = "cuda"
        self.args["max_grad_norm"] = 0.4
        self.args["clip_range"] = 0.15
        self.args["learning_rate"] = 0.0005
        self.args["n_steps"] = 400
        self.args["n_epochs"] = 8

    def set_counter_circuit_CNN_CUDA_RS(self):
        self.args["exp"] = "CNN_CUDA_RS"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.01
        self.args["ent_coef_end"] = 0.01
        self.args["trained_models"] = 5
        self.args["ent_coef_horizon"] = 1.6e6
        self.args["total_timesteps"] = 10e6
        self.args["vf_coef"] = 0.5
        self.args["batch_size"] = 2000
        self.args["device"] = "cuda"
        self.args["max_grad_norm"] = 0.1
        self.args["clip_range"] = 0.05
        self.args["learning_rate"] = 0.0008
        self.args["n_steps"] = 400
        self.args["n_epochs"] = 8
        self.args["sparse_r_coef_horizon"] = 5e6
        self.args["divergent_check_timestep"] = 10e6

    # 3 out of 3, n_steps = 800 is also OK, however divergent check should be set to cca 2.9e6
    def set_coordination_ring_CNN_CUDA_RS_WORKS(self):
        self.args["exp"] = "CNN_CUDA_RS_WORKS"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.0005
        self.args["trained_models"] = 10
        self.args["ent_coef_horizon"] = 3e6
        self.args["total_timesteps"] = 8e6
        self.args["vf_coef"] = 0.05
        self.args["batch_size"] = 1600
        self.args["device"] = "cuda"
        self.args["max_grad_norm"] = 0.4
        self.args["clip_range"] = 0.15
        self.args["learning_rate"] = 0.0005
        self.args["n_steps"] = 800
        self.args["n_epochs"] = 8


    def set_forced_coordination_CNN_CUDA_RS(self):
        self.args["exp"] = "CNN_CUDA_RS"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.05
        self.args["ent_coef_end"] = 0.05
        self.args["trained_models"] = 6
        self.args["ent_coef_horizon"] = 0.8e6
        self.args["total_timesteps"] = 10e6
        self.args["vf_coef"] = 0.5
        self.args["batch_size"] = 2000
        self.args["device"] = "cuda"
        self.args["max_grad_norm"] = 0.1
        self.args["clip_range"] = 0.2
        self.args["learning_rate"] = 0.001
        self.args["n_steps"] = 400
        self.args["n_epochs"] = 12
        self.args["sparse_r_coef_horizon"] = 3.5e6
        self.args["target_kl"] = None


    # this found solution also for coordination_ring
    def set_coordination_ring_CNN_SP_SMALL_NN(self):
        self.args["exp"] = "CNN_SP_SMALL_NN"
        self.args["random_start"] = True
        self.args["ent_coef_start"] = 0.1
        self.args["ent_coef_end"] = 0.0005
        self.args["trained_models"] = 1
        self.args["ent_coef_horizon"] = 10e6
        self.args["total_timesteps"] = 10e6
