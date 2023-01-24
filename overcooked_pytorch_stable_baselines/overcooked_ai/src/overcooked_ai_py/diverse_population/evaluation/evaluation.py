import numpy as np
import torch as th
from stable_baselines3.common.utils import obs_as_tensor


class Evaluator(object):
    def __init__(self, venv, args, deterministic=True, device="cpu"):
        self.venv = venv
        self.deterministic=deterministic
        self.device = device
        self.args = args

        self.venv.reset_times([i for i in range(args.num_workers)])

    def evaluate(self, agent_set_0, agent_set_1, num_games_per_worker = 2, file_name=None, deterministic=True):
        file_full_name = f"diverse_population/evaluation/{self.args.layout_name}/" + file_name + '' if deterministic else '_STOCH'
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

            if file_name is not None:
                np.savetxt(file_full_name, np.round(np.array(result_matrix)))

        return np.array(result_matrix)


    def eval_episodes(self, self_agent_model, other_agent_model, num_games_per_worker, deterministic=True):
        evaluation_rewards = []

        for _ in range(num_games_per_worker):
            self.venv._last_obs = self.venv.reset()
            for _ in range(400):
                with th.no_grad():
                    # Convert to pytorch tensor or to TensorDict
                    obs = np.array([entry["both_agent_obs"][0] for entry in self.venv._last_obs])
                    # obs_tensor = obs_as_tensor(obs, self.device)
                    actions, _= self_agent_model.policy.predict(obs, deterministic=deterministic)

                    other_agent_obs = np.array([entry["both_agent_obs"][1] for entry in self.venv._last_obs])
                    # other_agent_obs_tensor = obs_as_tensor(other_agent_obs, self.device)
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

