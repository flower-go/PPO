import copy
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gym
import numpy as np
import torch as th

import random

from overcooked_ai_py.diverse_population.DivergentSolutionException import DivergentSolutionException
import overcooked_ai_py.mdp.actions
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

OnPolicyAlgorithmSelf = TypeVar("OnPolicyAlgorithmSelf", bound="OnPolicyAlgorithm")



class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
    ):

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else RolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        pop_diff_reward = np.array([0 for _ in range(env.num_envs)])
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        kl_diff_reward_loss = th.nn.KLDivLoss(reduction="batchmean", log_target=True)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs = np.array([entry["both_agent_obs"][0] for entry in self._last_obs])
                obs_tensor = obs_as_tensor(obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
                # from pytorch_model_summary import summary
                # summary(self.policy, obs_tensor, print_summary=True, show_hierarchical=True, max_depth=10, )

                other_agent_obs = np.array([entry["both_agent_obs"][1] for entry in self._last_obs])
                if self.env.venv.population_mode:
                    other_agent_actions = []

                    ind_start = 0
                    for (pop_chunk, other_agent_model) in self.split_pop_indices():
                        other_obs = other_agent_obs[ind_start:ind_start+pop_chunk]
                        ind_start+=pop_chunk
                        other_agent_a, _ = other_agent_model.policy.predict(other_obs, deterministic=False) # TODO: should population agents play argmax during training?
                        other_agent_actions.append(other_agent_a)

                    other_agent_actions = np.concatenate(other_agent_actions)
                else:
                    other_agent_obs_tensor = obs_as_tensor(other_agent_obs, self.device)
                    other_agent_actions, other_agent_values, other_agent_log_probs = env.other_agent_model.policy(other_agent_obs_tensor)
                    other_agent_actions = other_agent_actions.cpu().numpy()

            actions = actions.cpu().numpy()

            joint_action = [(actions[i], other_agent_actions[i]) for i in range(len(actions))]

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(joint_action)

            agent_shaped_r = [info["shaped_r_by_agent"][info["policy_agent_idx"]] for info in infos]



            if self.env.population_mode:
                if self.args.delay_shared_reward:
                    rewards = (1 - self.shaped_r_coef_horizon) * rewards
                rewards = rewards + self.shaped_r_coef_horizon * np.array(agent_shaped_r)
            else:
                rewards = rewards + self.shaped_r_coef_horizon * np.array(agent_shaped_r)

            if self.env.population_mode and self.args.kl_diff_reward_coef > 0:
                with th.no_grad():
                    kl_divs = []
                    # actions_dist_logits = self.policy.get_distribution(obs_tensor).distribution.logits.cpu()
                    actions_dist_logits = self.policy.get_distribution(obs_tensor).distribution.logits

                    for ind in self.env.population:
                        # pop_ind_actions_dist_logits = ind.policy.get_distribution(obs_tensor).distribution.logits.cpu()
                        pop_ind_actions_dist_logits = ind.policy.get_distribution(obs_tensor).distribution.logits
                        diff = kl_diff_reward_loss(actions_dist_logits, pop_ind_actions_dist_logits)
                        kl_divs.append(diff.item())
                    pop_diff_reward = self.args.kl_diff_reward_coef * np.mean(kl_divs)
                    pop_diff_reward = np.clip(pop_diff_reward, 0, self.args.kl_diff_reward_clip)

                    rewards = rewards + pop_diff_reward

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            self._last_obs = np.array([entry["both_agent_obs"][0] for entry in self._last_obs])
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs, pop_diff_reward)

            self._last_obs = new_obs
            self._last_episode_starts = dones


        with th.no_grad():
            # Compute value for the last timestep

            new_obs = np.array([entry["both_agent_obs"][0] for entry in new_obs])
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self: OnPolicyAlgorithmSelf,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
            args = None
    ) -> OnPolicyAlgorithmSelf:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        self.args = args

        callback.on_training_start(locals(), globals())
        self.shaped_r_coef_horizon = 1
        self.kl_diff_reward_coef = args.kl_diff_reward_coef
        best_model = None
        best_model_eval_val = -1

        while self.num_timesteps < total_timesteps:

            if self.env.population_mode:
                random.shuffle(self.env.population)

            if self.args:
                self.anneal_learning_parameters()

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            # Divergent solution check
            if not self.env.population_mode and args.divergent_check_timestep is not None and self.num_timesteps > args.divergent_check_timestep and self.num_timesteps < args.divergent_check_timestep + 1e5:
                sparse_r = safe_mean([ep_info["ep_game_stats"]["cumulative_sparse_rewards_by_agent"][1 - ep_info["policy_agent_idx"]] for ep_info in self.ep_info_buffer])
                sparse_r_other_agent = safe_mean([ep_info["ep_game_stats"]["cumulative_sparse_rewards_by_agent"][ep_info["policy_agent_idx"]] for ep_info in self.ep_info_buffer])
                # Neither of agents have managed to serve single plate of soup within first args["divergent_check_timestep"], models have likely converged to some bad local optima
                if sparse_r < 3 or sparse_r_other_agent < 3:
                    print(f"Divergent solution with sparse reward values for agents ({sparse_r}, {sparse_r_other_agent})")
                    raise DivergentSolutionException(f"Divergent solution with sparse reward values for agents ({sparse_r}, {sparse_r_other_agent})")

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # self.rollout_buffer.on_rollout_start() # TODO: did i modify this?

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["ep_sparse_r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_shaped_rew_mean", safe_mean([ep_info["ep_sparse_r"] + ep_info["ep_shaped_r"]  for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/pop_diff_reward", np.mean(self.rollout_buffer.pop_diff_reward))
                    self.logger.record("rollout/cumulative_shaped_rewards_by_agent", safe_mean([ep_info["ep_game_stats"]["cumulative_shaped_rewards_by_agent"][ep_info["policy_agent_idx"]] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/cumulative_sparse_rewards_by_agent", safe_mean([ep_info["ep_game_stats"]["cumulative_sparse_rewards_by_agent"][ep_info["policy_agent_idx"]] for ep_info in self.ep_info_buffer]))

                    self.logger.record("rollout/cumulative_shaped_rewards_by_other_agent", safe_mean([ep_info["ep_game_stats"]["cumulative_shaped_rewards_by_agent"][1 - ep_info["policy_agent_idx"]] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/cumulative_sparse_rewards_by_other_agent", safe_mean([ep_info["ep_game_stats"]["cumulative_sparse_rewards_by_agent"][1 - ep_info["policy_agent_idx"]] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["ep_length"] for ep_info in self.ep_info_buffer]), exclude="tensorboard")
                self.logger.record("time/fps", fps, exclude="tensorboard")
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()


            evaluate = self.num_timesteps > args.training_percent_start_eval * args.total_timesteps

            if evaluate and args.eval_interval is not None and iteration % args.eval_interval == 0:
                evaluation_avg_rewards_per_episode = self.evaluate_env()
                eval_values = [evaluation_avg_rewards_per_episode]
                if evaluation_avg_rewards_per_episode > 0.98 * best_model_eval_val:
                    for _ in range(args.evals_num_to_threshold):
                        eval_values.append(self.evaluate_env())
                    if np.mean(eval_values) > best_model_eval_val:
                        print(f"found better model with value {np.mean(eval_values)}")
                        best_model_eval_val = np.mean(eval_values)
                        best_model = copy.deepcopy(self.policy)

                # if "eval_stop_threshold" in args and evaluation_avg_rewards_per_episode > args.eval_stop_threshold:
                if evaluation_avg_rewards_per_episode > args.eval_stop_threshold:
                    print(f"evaluation result over {args.eval_stop_threshold} detected, continuing with further evaluation")
                    eval_values = [evaluation_avg_rewards_per_episode]
                    for _ in range(args.evals_num_to_threshold):
                        eval_values.append(self.evaluate_env())

                    print(f"evaluation result after re-evaluation: {np.mean(eval_values)}")
                    if np.mean(eval_values) > args.eval_stop_threshold:
                        print("found good solution, terminating training")
                        self.logger.record("evaluation_rollout/avg_ep_rew_sum", np.mean(eval_values))
                        break
                else:
                    self.logger.record("evaluation_rollout/avg_ep_rew_sum", evaluation_avg_rewards_per_episode)

        callback.on_training_end()

        self.policy = best_model
        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

    def anneal_learning_parameters(self):
        if self.args:
            self.ent_coef = max(self.args.ent_coef_end,self.args.ent_coef_start - (self.num_timesteps / self.args.ent_coef_horizon) * (self.args.ent_coef_start - self.args.ent_coef_end))
            self.shaped_r_coef_horizon = max(0, 1 - (self.num_timesteps / self.args.shaped_r_coef_horizon))

    def evaluate_env(self):
        evaluation_rewards = []

        self._last_obs = self.env.reset()
        for _ in range(400):
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs = np.array([entry["both_agent_obs"][0] for entry in self._last_obs])
                # obs_tensor = obs_as_tensor(obs, self.device)
                actions, _= self.policy.predict(obs, deterministic=True)


                other_agent_obs = np.array([entry["both_agent_obs"][1] for entry in self._last_obs])
                if self.env.venv.population_mode:
                    other_agent_actions = []
                    ind_start = 0
                    for (pop_chunk, other_agent_model) in self.split_pop_indices():
                        other_obs = other_agent_obs[ind_start:ind_start+pop_chunk]
                        ind_start+=pop_chunk
                        # other_agent_obs_tensor = obs_as_tensor(other_obs, self.device)
                        other_agent_a, _ = other_agent_model.policy.predict(other_obs, deterministic=True)
                        other_agent_actions.append(other_agent_a)

                    other_agent_actions = np.concatenate(other_agent_actions)
                else:
                    other_agent_actions, _ = self.env.other_agent_model.policy.predict(other_agent_obs, deterministic=True)

            joint_action = [(actions[i], other_agent_actions[i]) for i in range(len(actions))]
            new_obs, rewards, dones, infos = self.env.step(joint_action)

            evaluation_rewards.append(rewards)

            self._last_obs = new_obs

        evaluation_rewards = np.array(evaluation_rewards)

        if self.env.venv.population_mode:
            ind_start = 0
            ind_sum_rewards = []

            for (pop_chunk, other_agent_model) in self.split_pop_indices():
                rews = evaluation_rewards[:, ind_start:ind_start+pop_chunk]
                sum_rewards = np.sum(rews) / pop_chunk
                ind_sum_rewards.append((sum_rewards, other_agent_model.custom_id))
                ind_start += pop_chunk

            print(ind_sum_rewards)

        evaluation_rewards = np.concatenate(evaluation_rewards)

        assert dones[0] == True, "after 400 steps env is not done"

        evaluation_avg_rewards_per_episode = np.sum(evaluation_rewards) / self.env.num_envs
        return evaluation_avg_rewards_per_episode


    def split_pop_indices(self):
        """
        Current population will be evenly distributed into available parallel environments
        This methods returns how many environments correspond to given individual from population
        """

        indices = []
        remaining_pop_size = len(self.env.population)
        remaining_pop_size = max(remaining_pop_size, 1)
        total = self.env.num_envs
        remaining = total
        while remaining > 0:
            chunk = int(np.ceil(remaining / remaining_pop_size))
            remaining_pop_size-=1
            remaining-=chunk
            indices.append(chunk)

        return zip(indices, self.env.population)


