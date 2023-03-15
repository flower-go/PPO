import copy

import numpy as np
import torch as th
import os
from stable_baselines3.common.utils import obs_as_tensor


class Evaluator(object):
    def __init__(self, venv, args, deterministic=True, device="cpu"):
        self.venv = venv
        self.deterministic=deterministic
        self.device = device
        self.args = args

        self.venv.reset_times([i for i in range(args.num_workers)])

    def evaluate(self, agent_set_0, agent_set_1, num_games_per_worker = 2, layout_name = None, group_name = None, deterministic=True, eval_env="", mode="POP"):
        file_full_name = f"{os.environ['PROJDIR']}/diverse_population/evaluation/{layout_name}/" + group_name + ('' if deterministic else '_STOCH')
        file_full_name += eval_env
        try:
            result_matrix = np.loadtxt(file_full_name)
        except:
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

            if file_full_name is not None:
                np.savetxt(file_full_name, np.round(np.array(result_matrix)))

        return np.array(result_matrix)

    def analyze(self, table, mode="POP", verbose=0):
        best_agent = None
        best_agent_avg = None
        best_init_avg=None
        init_avg=None
        final_best_avg=None
        best_pop_avg=None

        if mode == "POP":
            row_avgs = np.sum(table, axis=1) / table.shape[1]
            best_init_avg = np.max(row_avgs[:self.args.init_SP_agents])
            init_avg = np.mean(row_avgs[:self.args.init_SP_agents])
            avg = np.mean(table[self.args.init_SP_agents:])
            non_zero_avg = np.sum(table[self.args.init_SP_agents:][table[self.args.init_SP_agents:] > 0]) / table[self.args.init_SP_agents:][table[self.args.init_SP_agents:] > 0].size

            final_best_avg = np.max(row_avgs[self.args.trained_models:])
            best_pop_avg = np.max(row_avgs[self.args.init_SP_agents:self.args.trained_models])

            above_threshold = (table[self.args.init_SP_agents:] > 40).sum(axis=1)
            max_above_threshold = np.max(above_threshold)
            avg_above_threshold = np.mean(above_threshold)


            if verbose:
                print(f"best init row average: {best_init_avg}")
                print(f"init average: {init_avg}")

                print(f"final best row average: {final_best_avg}")
                # print(f"pop best row average: {best_pop_avg}")

                print(f"max_above_threshold: {max_above_threshold}")

                # print(f"mean SP part: {np.mean(table[:self.args.init_SP_agents])}")
                # print(f"mean SP of non zeros: {np.sum(table[:self.args.init_SP_agents][table[:self.args.init_SP_agents] > 0]) / table[:self.args.init_SP_agents][table[:self.args.init_SP_agents] > 0].size}")
                # print(f"zero SP ratio: {np.sum(table[:self.args.init_SP_agents] == 0.0) / table[:self.args.init_SP_agents].size}%")
                # print(f"mean POP part: {avg}")
                # print(f"mean POP of non zeros: {non_zero_avg}")
                # print(f"zero POP ratio: {np.sum(table[self.args.init_SP_agents:] == 0.0) / table[self.args.init_SP_agents:].size}%")

        else:
            zero_diag = copy.deepcopy(table)
            np.fill_diagonal(zero_diag, 0)
            row_avgs = np.sum(table, axis=1) / table.shape[1]
            best_agent = np.argmax(row_avgs)
            best_agent_avg = np.max(row_avgs)

            row_avgs = np.sum(zero_diag, axis=1) / zero_diag.shape[1]
            best_agent = np.argmax(row_avgs)
            best_agent_avg = np.max(row_avgs)

            sorted = np.sort(row_avgs)
            print(sorted)

            above_threshold = (zero_diag > 40).sum(axis=1)
            max_above_threshold = np.max(above_threshold)
            avg_above_threshold = np.mean(above_threshold)

            avg = np.mean(table[~np.eye(table.shape[0], dtype=bool)])
            if verbose:
                print(f"avg of diagonal: {np.mean(np.diagonal(table))}")
                print(f"avg of NON-diagonal: {avg}")

            non_zero_avg = None
        return {
            "best_agent": best_agent,
            "best_agent_avg": best_agent_avg,
            "avg": avg,
            "non_zero_avg": non_zero_avg,
            "best_init_avg": best_init_avg,
            "init_avg": init_avg,
            "final_best_avg": final_best_avg,
            "best_pop_avg": best_pop_avg,
            "max_above_threshold": max_above_threshold,
            "avg_above_threshold": avg_above_threshold

        }

    def eval_episodes(self, self_agent_model, other_agent_model, num_games_per_worker, deterministic=True):
        evaluation_rewards = []

        for _ in range(num_games_per_worker):
            self.venv._last_obs = self.venv.reset()
            obs = np.array([entry["both_agent_obs"][0] for entry in self.venv._last_obs])
            other_agent_obs = np.array([entry["both_agent_obs"][1] for entry in self.venv._last_obs])

            obs = self.initialize_obs(obs)
            other_agent_obs = self.initialize_obs(other_agent_obs)



            for _ in range(400):
                with th.no_grad():
                    # Convert to pytorch tensor or to TensorDict
                    # obs = np.array([entry["both_agent_obs"][0] for entry in self.venv._last_obs])
                    obs = self.update_obs(obs, np.array([entry["both_agent_obs"][0] for entry in self.venv._last_obs]))
                    actions, _ = self_agent_model.policy.predict(obs, deterministic=deterministic)

                    # other_agent_obs = np.array([entry["both_agent_obs"][1] for entry in self.venv._last_obs])
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

    # def update_obs(self, last_obs, obs):
    #     if self.args.frame_stacking > 1:
    #         last_obs = np.roll(last_obs, 1, axis=1)
    #         last_obs[:,0] = obs
    #         obs = last_obs
    #
    #     return obs

    def update_obs(self, last_obs, obs):
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
        obs = self.tranpose(obs)

        if self.args.frame_stacking > 1:
            if self.args.frame_stacking_mode == "tuple":
                # creates (B X Frames x C x W x H) observation
                obs = np.array([[single_obs for _ in range(self.args.frame_stacking)] for single_obs in obs])
            else:
                # creates (B x (C x Frames) x W x H) observation
                data = [obs]
                for _ in range(self.args.frame_stacking - 1):
                    data.append(obs[:, 0:10, :, :])
                # arr = [obs[:,0:10,:,:] for _ in range(self.args.frame_stacking)]
                obs = np.concatenate(data, axis=1)



        return obs

    def tranpose(self, obs):
        if len(obs.shape) == 5:
            obs = np.transpose(obs, (0, 1, 4, 2, 3))
        if len(obs.shape) == 4:
            obs = np.transpose(obs, (0, 3, 1, 2))

        return obs

