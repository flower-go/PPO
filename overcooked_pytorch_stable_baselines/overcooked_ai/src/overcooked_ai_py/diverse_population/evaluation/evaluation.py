import numpy as np
import torch as th
import os
import random
import logging


np.set_printoptions(threshold=np.inf)
obs_to_store = []
class Evaluator(object):
    def __init__(self, venv, args, deterministic=True, device="cpu"):
        self.venv = venv
        self.deterministic=deterministic
        self.device = device
        self.args = args

        self.venv.reset_times([i for i in range(args.num_workers)])
        random.seed(args.seed)
        logger = logging.getLogger("obs_logger")
        logger.setLevel(logging.DEBUG)
        log_filename = f'./observations/{args.exp}_observations.log'
        handler = logging.FileHandler(log_filename)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s\n')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        self.logger = logger

    def evaluate(self, agent_set_0, agent_set_1, num_games_per_worker = 2, layout_name = None, group_name = None, deterministic=True, eval_env="", mode="POP"):
        """
        Pairwise cross-evaluation is performed.
        Result is saved for future straightforward loading.
        """
        file_dir = f"{os.environ['PROJDIR']}/diverse_population/evaluation/{layout_name}/"
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        file_full_name = f"{os.environ['PROJDIR']}/diverse_population/evaluation/{layout_name}/" + group_name + ('' if deterministic else '_STOCH')
        file_full_name += eval_env
        # try:
        #    result_matrix = np.loadtxt(file_full_name)
        #    print("result_matrix loaded")
        # except:
        total = len(agent_set_0) * len(agent_set_1)
        completed = 0
        result_matrix = np.zeros((len(agent_set_0), len(agent_set_1)))
        agent_idxs = [0 for x in range(self.venv.num_envs)]
        self.venv.remote_set_agent_idx(agent_idxs)

        for i in range(len(agent_set_0)):
            for j in range(len(agent_set_1)):
                result_matrix[i,j] = self.eval_episodes(agent_set_0[i], agent_set_1[j], num_games_per_worker, deterministic=deterministic)
                completed = completed + 1

            print(f"completed {completed} out of {total}")
        if self.args.execution_mode == "obs":
            np.save(f"./observations/{self.args.exp}_observations_all.log", obs_to_store)
        else:
            if file_full_name is not None:
                np.savetxt(file_full_name, np.round(np.array(result_matrix)))

        return np.array(result_matrix)

    def eval_episodes(self, self_agent_model, other_agent_model, num_games_per_worker, deterministic=True):
        """
        Several environment episodes are evaluated for given pair of models.
        Returns average outcome.
        """

        evaluation_rewards = []

        for _ in range(num_games_per_worker):
            self.venv._last_obs = self.venv.reset()
            obs = np.array([entry["both_agent_obs"][0] for entry in self.venv._last_obs])
            other_agent_obs = np.array([entry["both_agent_obs"][1] for entry in self.venv._last_obs])

            #Observations are modified according to currently used frame stacking method
            obs = self.initialize_obs(obs)
            other_agent_obs = self.initialize_obs(other_agent_obs)

            for _ in range(400):
                with th.no_grad():
                    obs = self.update_obs(obs, np.array([entry["both_agent_obs"][0] for entry in self.venv._last_obs]))
                    if self.args.save_states and random.randint(0,100) < 3:
                        print("loguju observatitons")
                        #self.logger.debug(f"{obs}")
                        obs_to_store.append(obs)
                    actions, _ = self_agent_model.policy.predict(obs, deterministic=deterministic)

                    other_agent_obs = self.update_obs(other_agent_obs, np.array([entry["both_agent_obs"][1] for entry in self.venv._last_obs]))
                    other_agent_actions, _ = other_agent_model.policy.predict(other_agent_obs, deterministic=deterministic)

                joint_action = [(actions[i], other_agent_actions[i]) for i in range(len(actions))]
                new_obs, rewards, dones, infos = self.venv.step(joint_action)

                evaluation_rewards.append(rewards)

                self.venv._last_obs = new_obs

            assert dones[0] == True, "after 400 steps env is not done"

        evaluation_rewards = np.concatenate(evaluation_rewards)
        evaluation_avg_rewards_per_episode = np.sum(evaluation_rewards) / num_games_per_worker / self.venv.num_envs
        print(evaluation_avg_rewards_per_episode)

        return evaluation_avg_rewards_per_episode


    def update_obs(self, last_obs, obs):
        """
        Current observations are appended to previous state representation using corresponding frame stacking method.
        """
        obs = self.tranpose(obs)

        if self.args.frame_stacking > 1:
            if self.args.frame_stacking_mode == "tuple":
                last_obs = np.roll(last_obs, 1, axis=1)
                last_obs[:, 0] = obs
            else:
                num_channels_per_state = 22
                player_info_part = obs[:, 0:10, :, :]
                last_obs = np.roll(last_obs, 10, axis=1)
                last_obs[:, num_channels_per_state:num_channels_per_state + 10, :, :] = player_info_part
                last_obs[:, 0:num_channels_per_state, :, :] = obs
            obs = last_obs

        return obs

    def initialize_obs(self, obs):
        """
        Initial observations are resctructured according to the applied frame stacking method.
        """
        obs = self.tranpose(obs)

        if self.args.frame_stacking > 1:
            if self.args.frame_stacking_mode == "tuple":
                # creates (B X Frames x C x W x H) observation
                obs = np.array([[single_obs for _ in range(self.args.frame_stacking)] for single_obs in obs])
            else:
                # creates (B x (Frames * 10 + C) x W x H) observation
                data = [obs]
                for _ in range(self.args.frame_stacking - 1):
                    data.append(obs[:, 0:10, :, :])
                obs = np.concatenate(data, axis=1)
        return obs

    def tranpose(self, obs):
        """
        observations are transposed to channels-first format
        """
        if len(obs.shape) == 5:
            obs = np.transpose(obs, (0, 1, 4, 2, 3))
        if len(obs.shape) == 4:
            obs = np.transpose(obs, (0, 3, 1, 2))

        return obs

