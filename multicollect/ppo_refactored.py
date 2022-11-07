#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import copy

# print(tf.config.list_physical_devices('GPU'))
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'
# exit()
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# exit()

import multi_collect_environment
import wrappers
import time

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=44, type=int, help="Random seed.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
parser.add_argument("--exp_name", default="single_collect", help="Experiment name")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--mini_batch_size", default=128, type=int, help="Mini batch size.")
parser.add_argument("--clip_epsilon", default=0.15, type=float, help="Clipping epsilon.")
parser.add_argument("--entropy_regularization", default=0.05, type=float, help="Entropy regularization weight.")
parser.add_argument("--epochs", default=15, type=int, help="Epochs to train each iteration.")
parser.add_argument("--evaluate_each", default=15, type=int, help="Evaluate each given number of iterations.")
parser.add_argument("--evaluate_for", default=100, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.985, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--activation", default="tanh", help="Size of hidden layer.")
parser.add_argument("--actor_learning_rate", default=0.0002, type=float, help="Learning rate.")
parser.add_argument("--critic_learning_rate", default=0.0002, type=float, help="Learning rate.")
parser.add_argument("--trace_lambda", default=0.95, type=float, help="Traces factor lambda.")
parser.add_argument("--workers", default=16, type=int, help="Workers during experience collection.")
parser.add_argument("--worker_steps", default=100, type=int, help="Steps for each worker to perform.")
parser.add_argument("--total_timesteps", default=350000, type=int, help="Total timesteps of experiments")

# TODO: Note that this time we derive the Network directly from `tf.keras.Model`.
# The reason is that the high-level Keras API is useful in PPO, where we need
# to train on an unchanging dataset (generated batches, train for several epochs, ...).
# That means that:
# - we define training in `train_step` method, which the Keras API automatically uses
# - we still provide custom `predict` method, because it is fastest this way
# - loading and saving should be performed using `save_weights` and `load_weights`, so that
#   the `predict` method and the `Network` type is preserved. If you use `.h5` suffix, the
#   checkpoint will be a single file, which is useful for ReCodEx submission.
class Network(tf.keras.Model):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, args: argparse.Namespace) -> None:
        super(Network, self).__init__()
        self.args = args

        # Create a suitable model for the given observation and action spaces.
        inputs = tf.keras.layers.Input(observation_space.shape)

        # TODO: Using a single hidden layer with args.hidden_layer_size and ReLU activation,
        # produce a policy with `action_space.n` discrete actions.
        hidden_policy = tf.keras.layers.Dense(args.hidden_layer_size, activation=args.activation,kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001))(inputs)
        hidden_policy = tf.keras.layers.Dense(args.hidden_layer_size, activation=args.activation,kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001))(hidden_policy)
        policy = tf.keras.layers.Dense(action_space.n)(hidden_policy)

        self._actor_model = tf.keras.Model(inputs=inputs, outputs=policy)
        self._actor_optimizer = tf.optimizers.Adam(args.actor_learning_rate)
        self._actor_model.compile(optimizer=self._actor_optimizer)

        # TODO: Using an independent single hidden layer with args.hidden_layer_size and ReLU activation,
        # produce a value function estimate. It is best to generate it as a scalar, not
        # a vector of length one, to avoid broadcasting errors later.
        hidden_value = tf.keras.layers.Dense(args.hidden_layer_size, activation=args.activation,kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001))(inputs)
        hidden_value = tf.keras.layers.Dense(args.hidden_layer_size, activation=args.activation,kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001))(hidden_value)
        value = tf.keras.layers.Dense(1)(hidden_value)

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



    # Construct the network
    network = Network(env.observation_space, env.action_space, args)


    def evaluate_episode(start_evaluation:bool = False) -> float:
        rewards, state, done = 0, env.reset(start_evaluation), False
        while not done:
            if args.render_each and (env.episode + 1) % args.render_each == 0:
                env.render()

            action_prob = network.predict_action_prob_logits([state])
            action = np.argmax(action_prob)
            state, reward, done, _ = env.step(action)
            rewards += reward
        return rewards

    # Create the vectorized environment
    venv = gym.vector.AsyncVectorEnv([lambda: gym.make(env.spec.id)] * args.workers)
    venv.seed(args.seed)

    exp_name = f"exp_name:{args.exp_name};hid_size:{args.hidden_layer_size};clip_eps:{args.clip_epsilon};entr_reg:{args.entropy_regularization};act_lr:{args.actor_learning_rate};crit_lr:{args.critic_learning_rate};mini_b_size:{args.mini_batch_size};workers:{args.workers};worker_steps:{args.worker_steps};activation:{args.activation};epochs:{args.epochs}"
    hyperparameters = [tf.convert_to_tensor([k, str(v)]) for k, v in vars(args).items()]
    writer = tf.summary.create_file_writer(f"runs/{exp_name}")
    with writer.as_default():
        tf.summary.text("hyperparameters", tf.stack(hyperparameters), 0)

    # Training
    state = venv.reset()
    training = True
    update_num = 0
    num_updates = args.total_timesteps // args.batch_size
    print(num_updates)
    while training:
        print(f"update_num: {update_num}")
        if update_num > num_updates:
            break

        states, next_states, actions, action_probs, rewards, dones, values = [], [], [], [], [], [], []
        for _ in range(args.worker_steps):
            action_prob_logits = network.predict_action_prob_logits(state)
            probs = tf.nn.softmax(action_prob_logits).numpy()
            dist = tfp.distributions.Categorical(logits=action_prob_logits, dtype=tf.int32)
            action = dist.sample()

            next_state, reward, done, _ = venv.step(action)

            value = network.predict_values(state)
            action_prob = np.take_along_axis(probs, action.numpy().reshape((-1, 1)), axis=1).reshape((-1))

            actions.append(action)
            values.append(value)
            rewards.append(reward)
            states.append(state)
            next_states.append(next_state)
            dones.append(done)
            action_probs.append(action_prob)

            state = next_state

        value = network.predict_values(state)
        values.append(value)

        values = tf.cast(values, dtype=tf.float32)
        deltas = np.zeros((len(rewards), args.workers))

        for i in range(args.worker_steps):
            for worker in range(args.workers):
                deltas[i][worker] = rewards[i][worker] + args.gamma * (1 - dones[i][worker]) * values[i+1][worker] - values[i][worker]

        advantages = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            for worker in range(args.workers):
                advantages[t][worker] = (
                        advantages[t][worker]
                        + (1 - dones[t][worker]) * args.gamma * args.trace_lambda * advantages[t + 1][worker]
                )

        advantages = advantages.reshape((-1,args.workers))
        values = np.array(values[:-1]).reshape((-1,args.workers))
        returns = advantages + values


        advantages -= tf.reduce_mean(advantages)
        advantages /= tf.math.reduce_std(advantages) + 1e-8
        advantages = tf.cast(advantages, dtype=tf.float32)

        states = np.concatenate(states)
        actions = np.concatenate(actions)
        action_probs = np.concatenate(action_probs)
        advantages = np.concatenate(advantages)
        returns = np.concatenate(returns)

        b_inds = np.arange(args.batch_size)
        for epoch in range(args.epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.mini_batch_size):
                end = start + args.mini_batch_size
                mb_inds = b_inds[start:end]

                mb_states = states[mb_inds]
                mb_targets = {"actions": actions[mb_inds],
                           "action_probs": action_probs[mb_inds],
                           "advantages": advantages[mb_inds],
                           "returns": returns[mb_inds]}

                losses = network.train_step((mb_states, mb_targets))

        with writer.as_default():
            tf.summary.scalar("actor loss", losses["actor_loss"], step=update_num)
            tf.summary.scalar("critic loss", losses["critic_loss"], step=update_num)
            tf.summary.scalar("entropy", losses["entropy"], step=update_num)
            tf.summary.scalar("average rewards", np.mean(rewards), step=update_num)

        update_num+=1


    print(evaluate_episode(start_evaluation=True))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    args.batch_size = int(args.workers * args.worker_steps)



    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("SingleCollect-v0"), args.seed)
    print(env.observation_space)
    print(env.action_space)

    main(env, args)