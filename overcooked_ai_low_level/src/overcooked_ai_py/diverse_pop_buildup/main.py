import gym.spaces

from overcooked_ai_py.mdp.overcooked_env import *
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.planning.planners import MotionPlanner, NO_COUNTERS_PARAMS
from overcooked_ai_py.agents.agent import AgentGroup, NNPPOPolicy, RandomAgent, AgentFromPolicy, AgentPair
from overcooked_ai_py.diverse_pop_buildup.models.network import Network

import tensorflow as tf
import tensorflow_probability as tfp
import copy
from timeit import default_timer as timer


mdp = OvercookedGridworld.from_layout_name("cramped_room")
mlam = MediumLevelActionManager(mdp, NO_COUNTERS_PARAMS)
overcooked_env = OvercookedEnv.from_mdp(mdp, horizon=400)


env = Overcooked()
feature_fn = lambda _, state: overcooked_env.featurize_state_mdp(state)
env.custom_init(overcooked_env, feature_fn)




net = Network(mdp.featurize_state_shape, (Action.NUM_ACTIONS))

nn_ppo_policy = NNPPOPolicy(net)
agent_0 = AgentFromPolicy(nn_ppo_policy)
agent_1 = AgentFromPolicy(nn_ppo_policy)

agent_pair = AgentPair(agent_0, agent_1)
agent_pair.set_mdp(mdp)


steps = 400
gamma = 0.985
trace_lambda = 0.95
epochs = 10
mini_batch_size = 400
workers = 3
batch_size = workers * steps
mini_batch_size = batch_size

writer = tf.summary.create_file_writer(f"runs/time_compare_gym_default3(diff_params)_env")

mdps = [OvercookedGridworld.from_layout_name("cramped_room") for _ in range(workers)]
mlams = [MediumLevelActionManager(mdp, NO_COUNTERS_PARAMS) for mdp in mdps]
overcooked_envs = [OvercookedEnv.from_mdp(mdp, horizon=400) for mdp in mdps]


envs = [Overcooked() for _ in range(workers)]
feature_fns = [lambda _, state: overcooked_env.featurize_state_mdp(state) for overcooked_env in overcooked_envs]
for env, overcooked_env, feature_fn in zip(envs, overcooked_envs, feature_fns):
    env.custom_init(overcooked_env, feature_fn)

def create_single_gym_env():
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    mlam = MediumLevelActionManager(mdp, NO_COUNTERS_PARAMS)
    overcooked_env = OvercookedEnv.from_mdp(mdp, horizon=400)

    env = Overcooked()
    feature_fn = lambda _, state: overcooked_env.featurize_state_mdp(state)
    env.custom_init_parallel(overcooked_env, feature_fn)
    return env

venv = gym.vector.AsyncVectorEnv([lambda : create_single_gym_env()] * workers)
# venv.seed(42)

def train(env, net):
    environment_steps = 0


    for _ in range(10000):
        done = False
        states, next_states, actions, action_probs, rewards, dones, values = [], [], [], [], [], [], []
        env.reset()
        while not done:
            state = env.state
            state_observation = env.mdp.featurize_state(state, env.mlam)


            joint_action_and_infos = agent_pair.joint_action(state_observation)
            joint_action, a_info_t = zip(*joint_action_and_infos)
            next_state, reward, done, info = env.step(joint_action)

            # agent 0
            for agent in [0,1]:
                probs = a_info_t[agent]["action_probs"]
                action_index = Action.ACTION_TO_INDEX[joint_action[agent]]
                # action_prob = np.take_along_axis(action_probs, Action.ACTION_TO_INDEX[joint_action[0]], axis=0).reshape((-1))
                action_prob = probs[action_index]


                agent_reward = reward + info["shaped_r_by_agent"][agent]

                next_state_observation = env.mdp.featurize_state(next_state, env.mlam)

                actions.append(action_index)
                rewards.append(agent_reward)
                states.append(state_observation[agent])
                next_states.append(next_state_observation[agent])
                dones.append(done)
                action_probs.append(action_prob)





            value = net.predict_values(np.array([state_observation[0], state_observation[1]]))
            values.append(value[0])
            values.append(value[1])

        value = net.predict_values(np.array([state_observation[0], state_observation[1]]))
        values.append(value[0])
        values.append(value[1])



        values = tf.cast(values, dtype=tf.float32)
        deltas = np.zeros(len(rewards))

        for i in range(len(rewards)):
            deltas[i] = rewards[i] + gamma * (1 - dones[i]) * values[i + 1] - \
                                    values[i]

        advantages = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            advantages[t] = (
                    advantages[t]
                    + (1 - dones[t]) * gamma * trace_lambda * advantages[t + 1]
            )

        advantages = advantages.reshape((-1, len(rewards)))

        values = np.array(values[:-2]).reshape((-1, len(rewards)))
        returns = advantages + values

        advantages -= tf.reduce_mean(advantages)
        advantages /= tf.math.reduce_std(advantages) + 1e-8
        advantages = tf.cast(advantages, dtype=tf.float32)

        # states = np.concatenate(states)
        # actions = np.concatenate(actions)
        # action_probs = np.concatenate(action_probs)
        # advantages = np.concatenate(advantages)
        # returns = np.concatenate(returns)


        def take_along_axis(arr, inds):
            temp_arr = []
            for ind in inds:
                temp_arr.append(arr[ind])
            return np.array(temp_arr)

        b_inds = np.arange(len(rewards))
        for epoch in range(epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(rewards), mini_batch_size):
                end = start + mini_batch_size
                mb_inds = b_inds[start:end]


                mb_states = take_along_axis(states, mb_inds)
                mb_targets = {"actions": np.array(take_along_axis(actions, mb_inds), dtype=np.int32),
                              "action_probs": take_along_axis(action_probs, mb_inds),
                              "advantages": take_along_axis(advantages[0], mb_inds),
                              "returns": take_along_axis(returns[0], mb_inds)}

                losses = net.train_step((mb_states, mb_targets))

                with writer.as_default():
                    tf.summary.scalar("actor loss", losses["actor_loss"], step=environment_steps)
                    tf.summary.scalar("critic loss", losses["critic_loss"], step=environment_steps)
                    tf.summary.scalar("entropy", losses["entropy"], step=environment_steps)
                    tf.summary.scalar("average rewards", np.mean(rewards), step=environment_steps)

        environment_steps += 1 * steps

def take_along_axis(arr, inds):
    temp_arr = []
    for ind in inds:
        temp_arr.append(arr[ind])
    return np.array(temp_arr)

def take_along_axis_2(arr, inds):
    temp_arr = []
    for element, ind in zip(arr, inds):
        temp_arr.append(element[ind])
    return np.array(temp_arr)

def train_gym(env, net):
    environment_steps = 0


    for _ in range(10000):
        done = False
        states, next_states, actions, action_probs, rewards, dones, values = [], [], [], [], [], [], []
        state_obs_dict = env.reset()

        while not done:
            state_observation = state_obs_dict["both_agent_obs"]
            # state = env.state
            # state_observation = env.mdp.featurize_state(state, env.mlam)


            joint_action_and_infos = agent_pair.joint_action(state_observation)
            joint_action, a_info_t = zip(*joint_action_and_infos)
            joint_action = [Action.ACTION_TO_INDEX[a] for a in joint_action]
            next_state_obs_dict, reward, done, info = env.step(joint_action)

            # agent 0
            for agent in [0,1]:
                probs = a_info_t[agent]["action_probs"]
                action_index = joint_action[agent]
                # action_prob = np.take_along_axis(action_probs, Action.ACTION_TO_INDEX[joint_action[0]], axis=0).reshape((-1))
                action_prob = probs[action_index]


                agent_reward = reward + info["shaped_r_by_agent"][agent]

                # next_state_observation = env.mdp.featurize_state(next_state, env.mlam)

                actions.append(action_index)
                rewards.append(agent_reward)
                states.append(state_observation[agent])
                next_states.append(next_state_obs_dict["both_agent_obs"][agent])
                dones.append(done)
                action_probs.append(action_prob)





            value = net.predict_values(np.array([state_observation[0], state_observation[1]]))
            values.append(value[0])
            values.append(value[1])

            state_obs_dict = next_state_obs_dict



        value = net.predict_values(np.array([state_observation[0], state_observation[1]]))
        values.append(value[0])
        values.append(value[1])



        values = tf.cast(values, dtype=tf.float32)
        deltas = np.zeros(len(rewards))

        for i in range(len(rewards)):
            deltas[i] = rewards[i] + gamma * (1 - dones[i]) * values[i + 1] - \
                                    values[i]

        advantages = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            advantages[t] = (
                    advantages[t]
                    + (1 - dones[t]) * gamma * trace_lambda * advantages[t + 1]
            )

        advantages = advantages.reshape((-1, len(rewards)))

        values = np.array(values[:-2]).reshape((-1, len(rewards)))
        returns = advantages + values

        advantages -= tf.reduce_mean(advantages)
        advantages /= tf.math.reduce_std(advantages) + 1e-8
        advantages = tf.cast(advantages, dtype=tf.float32)

        # states = np.concatenate(states)
        # actions = np.concatenate(actions)
        # action_probs = np.concatenate(action_probs)
        # advantages = np.concatenate(advantages)
        # returns = np.concatenate(returns)




        b_inds = np.arange(len(rewards))
        for epoch in range(epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(rewards), mini_batch_size):
                end = start + mini_batch_size
                mb_inds = b_inds[start:end]


                mb_states = take_along_axis(states, mb_inds)
                mb_targets = {"actions": np.array(take_along_axis(actions, mb_inds), dtype=np.int32),
                              "action_probs": take_along_axis(action_probs, mb_inds),
                              "advantages": take_along_axis(advantages[0], mb_inds),
                              "returns": take_along_axis(returns[0], mb_inds)}

                losses = net.train_step((mb_states, mb_targets))

                with writer.as_default():
                    tf.summary.scalar("actor loss", losses["actor_loss"], step=environment_steps)
                    tf.summary.scalar("critic loss", losses["critic_loss"], step=environment_steps)
                    tf.summary.scalar("entropy", losses["entropy"], step=environment_steps)
                    tf.summary.scalar("average rewards", np.mean(rewards), step=environment_steps)

        environment_steps += 1 * steps

def train_gym_parallel(venv, net):
    environment_steps = 0


    for _ in range(10000):
        done = [False for _ in range(workers)]
        states_data, next_states_data, actions_data, action_probs_data, rewards_data, dones_data, values_data = [[] for _ in range(2)], [[] for _ in range(2)], [[] for _ in range(2)], [[] for _ in range(2)], [[] for _ in range(2)], [[] for _ in range(2)], [[] for _ in range(2)]

        state_obs, info_dict = venv.reset()
        i = 0
        while not all(done):
            joint_action = []
            joint_probs = []
            for agent_index, agent in zip(range(2), [agent_pair.a0, agent_pair.a1]):
                joint_action_and_infos = agent.actions(state_obs, [agent_index for _ in range(workers)])
                action_repr_batch, a_info_t = zip(*joint_action_and_infos)
                action_batch = [Action.ACTION_TO_INDEX[a] for a in action_repr_batch]
                joint_action.append(action_batch)
                joint_probs.append([info["action_probs"] for info in a_info_t[:]])

            joint_action = np.array(joint_action).transpose((1, 0))     # Reshaping from [agent, worker] to [worker, agent]
            joint_probs = np.array(joint_probs).transpose((1, 0, 2))       # Reshaping from [agent, worker,action] to [worker, agent, action]
            next_state_obs, reward, done, _,  info = venv.step(joint_action)    # info now includes values from obs_dict that was intended to be returned from overcooked_env as first value (obs)

            # Dividing data to two sets corresponding to agents
            # In case of self play both datasets can be concatenated together for training since agents use same network
            # In case of agent learning against static agent, only data of learned agent must be used
            # At each venv step chunks of size [workers] are appended to data sets

            for agent_index, agent in zip(range(2), [agent_pair.a0, agent_pair.a1]):
                # TODO: Check if data collection is necessary for agent
                # TODO: agent indecies are randomized by overcooked_env, check to which agent data set data belong to
                # TODO: Maybe reconsider manipulate overcooked_env init state generation rather than agent index switching???


                probs = joint_probs[:,agent_index]
                action_indecies = joint_action[:,agent_index]
                action_probs = take_along_axis_2(probs, action_indecies)
                if all(done):
                    agent_info = info["final_info"]
                    agent_reward = reward + np.array([info_by_worker["shaped_r_by_agent"][agent_index] for info_by_worker in agent_info])
                else:
                    agent_reward = reward + np.array([info_by_worker[agent_index] for info_by_worker in info["shaped_r_by_agent"]])

                actions_data[agent_index].append(action_indecies)
                rewards_data[agent_index].append(agent_reward)
                states_data[agent_index].append(state_obs[:,agent_index])
                next_states_data[agent_index].append(next_state_obs[:,agent_index])
                dones_data[agent_index].append(done)
                action_probs_data[agent_index].append(action_probs)


                values = agent.policy.network.predict_values(state_obs[:,agent_index])
                values_data[agent_index].append(values.numpy().reshape(-1))


            state_obs = next_state_obs
            i += 1

        for agent_index, agent in zip(range(2), [agent_pair.a0, agent_pair.a1]):
            # TODO: Check if data collection is necessary for agent
            # TODO: agent indecies are randomized by overcooked_env, check to which agent data set data belong to
            values = agent.policy.network.predict_values(state_obs[:,agent_index])
            values_data[agent_index].append(values.numpy().reshape(-1))

        # TODO: Data transposition from [agent, step, worker] to [step, worker, agent]
        states_data = np.array(states_data).transpose((1,2,0, 3))
        actions_data = np.array(actions_data).transpose((1, 2, 0))
        action_probs_data = np.array(action_probs_data).transpose((1, 2, 0))
        rewards_data = np.array(rewards_data).transpose((1,2,0))
        dones_data = np.array(dones_data).transpose((1, 2, 0))
        values_data = np.array(values_data).transpose((1, 2, 0))


        values_data = tf.cast(values_data, dtype=tf.float32)
        deltas = np.zeros((len(rewards_data), workers))

        steps = len(rewards_data)

        for agent_index, agent in zip(range(2), [agent_pair.a0, agent_pair.a1]):
            # TODO: Check if data collection and network learning is expected for agent
            # TODO: agent indecies are randomized by overcooked_env, check to which agent data set data belong to

            # in mappo: [step,worker,agent], here [agent, step, worker], this will be solved after data transposition
            for i in range(steps):
                for worker in range(workers):
                    deltas[i][worker] = rewards_data[i][worker][agent_index] + gamma * (1 - dones_data[i][worker][agent_index]) * values_data[i + 1][
                        worker][agent_index] - values_data[i][worker][agent_index]

            advantages = copy.deepcopy(deltas)
            for t in reversed(range(len(deltas) - 1)):
                for worker in range(workers):
                    advantages[t][worker] = (
                            advantages[t][worker]
                            + (1 - dones_data[t][worker][agent_index]) * gamma * trace_lambda * advantages[t + 1][worker]
                    )

            advantages = advantages.reshape((-1, workers))
            agent_values = np.array(values_data[:-1,:,agent_index]).reshape((-1, workers))
            returns = advantages + agent_values

            advantages -= tf.reduce_mean(advantages)
            advantages /= tf.math.reduce_std(advantages) + 1e-8
            advantages = tf.cast(advantages, dtype=tf.float32)

            agent_states = np.concatenate(states_data)[:, agent_index]
            agent_actions = np.concatenate(actions_data)[:, agent_index]
            agent_action_probs = np.concatenate(action_probs_data)[:, agent_index]
            advantages = np.concatenate(advantages)
            returns = np.concatenate(returns)

            b_inds = np.arange(batch_size)
            actor_losses = []
            critic_losses = []
            entropy_losses = []
            for epoch in range(epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, mini_batch_size):
                    end = start + mini_batch_size
                    mb_inds = b_inds[start:end]

                    mb_states = agent_states[mb_inds]
                    mb_targets = {"actions": np.array(agent_actions[mb_inds], dtype=np.int32),
                                  "action_probs": agent_action_probs[mb_inds],
                                  "advantages": advantages[mb_inds],
                                  "returns": returns[mb_inds]}

                    losses = agent.policy.network.train_step((mb_states, mb_targets))
                    actor_losses.append(losses["actor_loss"])
                    critic_losses.append(losses["critic_loss"])
                    entropy_losses.append(losses["entropy"])



            with writer.as_default():
                tf.summary.scalar(f"actor loss agent {agent_index}", np.mean(actor_losses), step=environment_steps)
                tf.summary.scalar(f"critic loss agent {agent_index}", np.mean(critic_losses), step=environment_steps)
                tf.summary.scalar(f"entropy agent {agent_index}", np.mean(entropy_losses), step=environment_steps)
                tf.summary.scalar(f"average rewards agent {agent_index}", np.mean(np.array(rewards_data)[:,:,agent_index]), step=environment_steps)
                # tf.summary.scalar(f"average global rewards", np.mean(global_rewards), step=update_num)
                # tf.summary.scalar(f"actor loss", losses["actor_loss"], step=update_num)
                # tf.summary.scalar(f"critic loss", losses["critic_loss"], step=update_num)
                # tf.summary.scalar(f"entropy", losses["entropy"], step=update_num)
                # tf.summary.scalar(f"average rewards", np.mean(global_rewards), step=update_num)

            environment_steps += workers * steps











start = timer()
train(overcooked_env, net)
# train_gym(env, net)
# train_gym_parallel(venv, net)
end = timer()
print(end - start)


