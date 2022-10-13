    #!/usr/bin/env python3
import argparse
import os
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import copy

import multi_collect_environment
import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--agents", default=3, type=int, help="Agents to use.")
# parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=46, type=int, help="Random seed.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
parser.add_argument("--exp_name", default="Multi_collect_shared_network", help="Experiment name")
# For these and any other arguments you add, ReCodEx will keep your default value.
# parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--mini_batch_size", default=3200, type=int, help="Mini batch size.")
parser.add_argument("--clip_epsilon", default=0.15, type=float, help="Clipping epsilon.")
parser.add_argument("--entropy_regularization", default=0.02, type=float, help="Entropy regularization weight.")
parser.add_argument("--epochs", default=6, type=int, help="Epochs to train each iteration.")
parser.add_argument("--evaluate_each", default=10, type=int, help="Evaluate each given number of iterations.")
parser.add_argument("--evaluate_for", default=3, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.985, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--activation", default="relu", help="Size of hidden layer.")
parser.add_argument("--std", default=0.1, help="std of init")
parser.add_argument("--actor_learning_rate", default=0.0002, type=float, help="Learning rate.")
parser.add_argument("--critic_learning_rate", default=0.0021, type=float, help="Learning rate.")
parser.add_argument("--trace_lambda", default=0.95, type=float, help="Traces factor lambda.")
parser.add_argument("--workers", default=32, type=int, help="Workers during experience collection.")
parser.add_argument("--worker_steps", default=100, type=int, help="Steps for each worker to perform.")
parser.add_argument("--total_timesteps", default=60000000, type=int, help="Total timesteps of experiments")
parser.add_argument("--learning_variation_steps", default=3, help="agent variation learning")
parser.add_argument("--single_learner", default=False, help="Single agent is being learned")

# TODO(ppo): We use the exactly same Network as in the `ppo` assignment.
class Network(tf.keras.Model):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, args: argparse.Namespace) -> None:
        super(Network, self).__init__()
        self.args = args

        inputs = tf.keras.layers.Input((env.observation_space.shape[0] + args.agents, ))
        # inputs = tf.keras.layers.Input(observation_space.shape)

        hidden_policy = tf.keras.layers.Dense(args.hidden_layer_size, activation=args.activation, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=args.std))(inputs)
        hidden_policy = tf.keras.layers.Dense(args.hidden_layer_size, activation=args.activation, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=args.std))(hidden_policy)
        policy = tf.keras.layers.Dense(action_space.n, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=args.std))(hidden_policy)

        self._actor_model = tf.keras.Model(inputs=inputs, outputs=policy)
        self._actor_optimizer = tf.optimizers.Adam(args.actor_learning_rate)
        self._actor_model.compile(optimizer=self._actor_optimizer)

        hidden_value = tf.keras.layers.Dense(args.hidden_layer_size, activation=args.activation, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=args.std))(inputs)
        hidden_value = tf.keras.layers.Dense(args.hidden_layer_size, activation=args.activation, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=args.std))(hidden_value)
        value = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=args.std))(hidden_value)

        self._critic_model = tf.keras.Model(inputs=inputs, outputs=value)
        self._critic_optimizer = tf.optimizers.Adam(args.critic_learning_rate)
        self._critic_model.compile(optimizer=self._critic_optimizer)

    @tf.function
    def train_step(self, data):
        states, targets = data
        old_probs = targets["action_probs"]

        with tf.GradientTape() as actor_tape:
            # probs, critic_value = self(states, training=True)
            probs_logits = self._actor_model(states, training=True)
            dist = tfp.distributions.Categorical(logits=probs_logits, dtype=tf.int64)

            probs = tf.nn.softmax(probs_logits)

            ind_pairs = tf.stack([tf.range(tf.shape(targets["actions"])[0]), targets["actions"]],axis=1)
            ind_pairs = tf.reshape(ind_pairs, (-1, 2))

            action_prob = tf.gather_nd(indices=ind_pairs, params=probs)

            entropy = tf.reduce_mean(dist.entropy())

            # log_ratio = tf.math.log(action_prob) - tf.math.log(old_probs)
            # r_theta = tf.math.exp(log_ratio)
            r_theta = action_prob / old_probs

            policy_obj = r_theta * targets["advantages"]
            clipped_r_theta = tf.clip_by_value(
                r_theta, 1 - self.args.clip_epsilon, 1 + self.args.clip_epsilon
            ) * targets["advantages"]

            act_loss = -tf.reduce_mean(
                tf.minimum(policy_obj, clipped_r_theta)
            )

            actor_loss = act_loss - self.args.entropy_regularization * entropy

        self._actor_optimizer.minimize(actor_loss, self._actor_model.trainable_variables, tape=actor_tape)

        with tf.GradientTape() as critic_tape:
            critic_value = self._critic_model(states, training=True)

            critic_value = tf.squeeze(critic_value)
            critic_loss = tf.reduce_mean(
                tf.square(tf.cast(targets["returns"], dtype=tf.float32) - critic_value)
            )

        self._critic_optimizer.minimize(critic_loss, self._critic_model.trainable_variables, tape=critic_tape)

        return {"actor_loss": actor_loss, "critic_loss": critic_loss, "entropy": entropy}

    @tf.function
    def train_actor_step(self, data):
        states, targets = data
        old_probs = targets["action_probs"]

        with tf.GradientTape() as actor_tape:
            # probs, critic_value = self(states, training=True)
            probs_logits = self._actor_model(states, training=True)
            dist = tfp.distributions.Categorical(logits=probs_logits, dtype=tf.int64)

            probs = tf.nn.softmax(probs_logits)

            ind_pairs = tf.stack([tf.range(tf.shape(targets["actions"])[0]), targets["actions"]],axis=1)
            ind_pairs = tf.reshape(ind_pairs, (-1, 2))

            action_prob = tf.gather_nd(indices=ind_pairs, params=probs)

            entropy = tf.reduce_mean(dist.entropy())

            # log_ratio = tf.math.log(action_prob) - tf.math.log(old_probs)
            # r_theta = tf.math.exp(log_ratio)
            r_theta = action_prob / old_probs

            policy_obj = r_theta * targets["advantages"]
            clipped_r_theta = tf.clip_by_value(
                r_theta, 1 - self.args.clip_epsilon, 1 + self.args.clip_epsilon
            ) * targets["advantages"]

            act_loss = -tf.reduce_mean(
                tf.minimum(policy_obj, clipped_r_theta)
            )

            actor_loss = act_loss - self.args.entropy_regularization * entropy

        self._actor_optimizer.minimize(actor_loss, self._actor_model.trainable_variables, tape=actor_tape)

        return {"actor_loss": actor_loss, "entropy": entropy}

    @tf.function
    def train_critic_step(self, data):
        states, targets = data

        with tf.GradientTape() as critic_tape:
            critic_value = self._critic_model(states, training=True)

            critic_value = tf.squeeze(critic_value)
            critic_loss = tf.reduce_mean(
                tf.square(tf.cast(targets["returns"], dtype=tf.float32) - critic_value)
            )

        self._critic_optimizer.minimize(critic_loss, self._critic_model.trainable_variables, tape=critic_tape)

        return {"critic_loss": critic_loss}

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_action_prob_logits(self, states):
        return self._actor_model(states)

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_values(self, states):
        return self._critic_model(states)


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace):
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the networks, each for the same observation space and corresponding action space.
    networks = [Network(env.observation_space, env.action_space[i], args) for i in range(args.agents)]

    def evaluate_episode(start_evaluation:bool = False) -> float:
        rewards, state, done = 0, env.reset(start_evaluation), False
        while not done:
            # if args.render_each and (env.episode + 1) % args.render_each == 0:
            #     env.render()

            action = []
            # TODO: Predict the vector of actions using the greedy policy
            for agent in range(args.agents):
                id = np.zeros(args.agents)
                id[agent] = 1

                id_st = np.concatenate((id, state))
                action_prob = networks[0].predict_action_prob_logits([id_st])
                act = np.argmax(action_prob)
                action.append(act)

            state, reward, done, _ = env.step(action)
            rewards += reward
        # print(rewards)
        return rewards

    # Create the vectorized environment
    venv = gym.vector.AsyncVectorEnv([lambda: gym.make(env.spec.id)] * args.workers)
    venv.seed(args.seed)


    exp_name = f"exp_name:{args.exp_name};hid_size:{args.hidden_layer_size};clip_eps:{args.clip_epsilon};entr_reg:{args.entropy_regularization};act_lr:{args.actor_learning_rate};crit_lr:{args.critic_learning_rate};mini_b_size:{args.mini_batch_size};workers:{args.workers};worker_steps:{args.worker_steps};activation:{args.activation};epochs:{args.epochs};agents:{args.agents}"
    hyperparameters = [tf.convert_to_tensor([k, str(v)]) for k, v in vars(args).items()]
    writer = tf.summary.create_file_writer(f"runs/{exp_name}")
    with writer.as_default():
        tf.summary.text("hyperparameters", tf.stack(hyperparameters),0)

    # Training
    st = venv.reset()
    training = True
    update_num = 0
    num_updates = args.total_timesteps // args.batch_size
    learned_agent = 0
    print(num_updates)
    while training:
        print(f"update_num: {update_num}")
        if update_num > num_updates:
            break

        states, actions, action_probs, rewards, global_rewards, dones, values = [], [], [], [], [], [], []
        for _ in range(args.worker_steps):
            action = []
            action_prob = []
            value = []
            state = []
            for a in range(args.agents):
                id = np.zeros(args.agents)
                id[a] = 1
                id = np.vstack([id]*args.workers)

                id_st = np.concatenate((id, st),axis=1)

                action_prob_logits = networks[0].predict_action_prob_logits(id_st)

                probs = tf.nn.softmax(action_prob_logits).numpy()

                dist = tfp.distributions.Categorical(logits=action_prob_logits, dtype=tf.int32)
                act = dist.sample()
                act_prob = np.take_along_axis(probs, act.numpy().reshape((-1,1)), axis=1).reshape((-1))

                val = networks[0].predict_values(id_st)

                state.append(id_st)

                action.append(act)
                action_prob.append(act_prob)
                value.append(val)

            state = np.array(state).transpose((1,0,2))
            action = np.array(action).transpose((1, 0))
            action_prob = np.array(action_prob).transpose((1, 0))
            value = np.array(value).transpose((1,0,2))

            action = np.array(action).reshape((-1,args.agents))
            action_prob = np.array(action_prob).reshape((-1, args.agents))
            value = np.array(value).reshape((-1, args.agents))

            next_state, global_reward, done, info = venv.step(action)
            reward = np.array([i["agent_rewards"] for i in info])



            # TODO: Collect the required quantities
            actions.append(action)
            values.append(value)
            rewards.append(reward)
            global_rewards.append(global_reward)
            states.append(state)
            dones.append(done)
            action_probs.append(action_prob)

            st = next_state

        values.append(value)

        values = tf.cast(values, dtype=tf.float32)
        deltas = np.zeros((len(rewards), args.workers))

        for a in range(args.agents):
            if args.learning_variation_steps and learned_agent != a:
                continue
            if args.single_learner and a == 1:
                continue
            for i in range(args.worker_steps):
                for worker in range(args.workers):
                    deltas[i][worker] = rewards[i][worker][a] + args.gamma * (1 - dones[i][worker]) * values[i + 1][
                        worker][a] - values[i][worker][a]

            advantages = copy.deepcopy(deltas)
            for t in reversed(range(len(deltas) - 1)):
                for worker in range(args.workers):
                    advantages[t][worker] = (
                            advantages[t][worker]
                            + (1 - dones[t][worker]) * args.gamma * args.trace_lambda * advantages[t + 1][worker]
                    )

            advantages = advantages.reshape((-1, args.workers))
            agent_values = np.array(values[:-1,:,a]).reshape((-1, args.workers))
            returns = advantages + agent_values

            advantages -= tf.reduce_mean(advantages)
            advantages /= tf.math.reduce_std(advantages) + 1e-8
            advantages = tf.cast(advantages, dtype=tf.float32)

            agent_states = np.concatenate(states)[:, a]
            agent_actions = np.concatenate(actions)[:, a]
            agent_action_probs = np.concatenate(action_probs)[:, a]
            advantages = np.concatenate(advantages)
            returns = np.concatenate(returns)

            b_inds = np.arange(args.batch_size)
            actor_losses = []
            critic_losses = []
            entropy_losses = []
            for epoch in range(args.epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.mini_batch_size):
                    end = start + args.mini_batch_size
                    mb_inds = b_inds[start:end]

                    mb_states = agent_states[mb_inds]
                    mb_targets = {"actions": agent_actions[mb_inds],
                                  "action_probs": agent_action_probs[mb_inds],
                                  "advantages": advantages[mb_inds],
                                  "returns": returns[mb_inds]}

                    losses = networks[0].train_step((mb_states, mb_targets))
                    actor_losses.append(losses["actor_loss"])
                    critic_losses.append(losses["critic_loss"])
                    entropy_losses.append(losses["entropy"])


            with writer.as_default():
                tf.summary.scalar(f"actor loss agent {a}", np.mean(actor_losses), step=update_num)
                tf.summary.scalar(f"critic loss agent {a}", np.mean(critic_losses), step=update_num)
                tf.summary.scalar(f"entropy agent {a}", np.mean(entropy_losses), step=update_num)
                tf.summary.scalar(f"average rewards agent {a}", np.mean(np.array(rewards)[:,:,a]), step=update_num)
                tf.summary.scalar(f"average global rewards", np.mean(global_rewards), step=update_num)
                # tf.summary.scalar(f"actor loss", losses["actor_loss"], step=update_num)
                # tf.summary.scalar(f"critic loss", losses["critic_loss"], step=update_num)
                # tf.summary.scalar(f"entropy", losses["entropy"], step=update_num)
                # tf.summary.scalar(f"average rewards", np.mean(global_rewards), step=update_num)

        update_num += 1
        if args.learning_variation_steps and (update_num % args.learning_variation_steps) == 0:
            learned_agent = (learned_agent + 1) % args.agents

        if update_num % args.evaluate_each == 0:
            returns = []
            for _ in range(args.evaluate_for):
                returns.append(evaluate_episode())
            with writer.as_default():
                tf.summary.scalar(f"evaluation rewards", np.mean(returns), step=update_num)

    evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'

    args = parser.parse_args([] if "__file__" not in globals() else None)
    args.batch_size = int(args.workers * args.worker_steps)
    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("MultiCollect{}-v0".format(args.agents)), args.seed)



    main(env, args)
