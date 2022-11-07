#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import copy


import multi_collect_environment
import wrappers
import time

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=44, type=int, help="Random seed.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--clip_epsilon", default=0.15, type=float, help="Clipping epsilon.")
parser.add_argument("--entropy_regularization", default=0.10, type=float, help="Entropy regularization weight.")
parser.add_argument("--epochs", default=8, type=int, help="Epochs to train each iteration.")
parser.add_argument("--evaluate_each", default=10, type=int, help="Evaluate each given number of iterations.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.985, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate.")
parser.add_argument("--trace_lambda", default=0.95, type=float, help="Traces factor lambda.")
parser.add_argument("--workers", default=16, type=int, help="Workers during experience collection.")
parser.add_argument("--worker_steps", default=100, type=int, help="Steps for each worker to perform.")

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
        self.args = args

        # Create a suitable model for the given observation and action spaces.
        inputs = tf.keras.layers.Input(observation_space.shape)

        # TODO: Using a single hidden layer with args.hidden_layer_size and ReLU activation,
        # produce a policy with `action_space.n` discrete actions.
        hidden_policy = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(inputs)
        hidden_policy = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(hidden_policy)
        policy = tf.keras.layers.Dense(action_space.n, activation="softmax")(hidden_policy)

        # TODO: Using an independent single hidden layer with args.hidden_layer_size and ReLU activation,
        # produce a value function estimate. It is best to generate it as a scalar, not
        # a vector of length one, to avoid broadcasting errors later.
        hidden_value = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(inputs)
        hidden_value = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(hidden_value)
        value = tf.keras.layers.Dense(1)(hidden_value)

        # Construct the model
        super().__init__(inputs=inputs, outputs=[policy, value])

        # Compile using Adam optimizer with the given learning rate.
        self.compile(optimizer=tf.optimizers.Adam(args.learning_rate))

    # TODO: Define a training method `train_step`, which is automatically used by Keras.
    def train_step(self, data):
        # Unwrap the data. The targets is a dictionary of several tensors, containing keys
        # - "actions"
        # - "action_probs"
        # - "advantages"
        # - "returns"

        states, targets = data
        old_probs = targets["action_probs"]

        with tf.GradientTape() as tape:
            probs, critic_value = self(states, training=True)
            dist = tfp.distributions.Categorical(probs=probs, dtype=tf.int64)

            ind_pairs = tf.stack([tf.range(tf.shape(targets["actions"])[0]), targets["actions"]],axis=1)
            ind_pairs = tf.reshape(ind_pairs, (-1, 2))

            action_prob = tf.gather_nd(indices=ind_pairs, params=probs)


            critic_value = tf.squeeze(critic_value)

            entropy = tf.reduce_mean(dist.entropy())

            r_theta = action_prob / old_probs
            policy_obj = r_theta * targets["advantages"]
            clipped_r_theta = tf.clip_by_value(
                r_theta, 1 - self.args.clip_epsilon, 1 + self.args.clip_epsilon
            ) * targets["advantages"]

            actor_loss = -tf.reduce_mean(
                tf.minimum(policy_obj, clipped_r_theta)
            )

            critic_loss = tf.reduce_mean(
                tf.square(targets["returns"] - critic_value)
            )

            loss = actor_loss + critic_loss - self.args.entropy_regularization * entropy


            # TODO: Sum the following three losses
            # - the PPO loss, where `self.args.clip_epsilon` is used to clip the probability ratio
            # - the MSE error between the predicted value function and target regurns
            # - the entropy regularization with coefficient `self.args.entropy_regularization`.
            #   You can compute it for example using `tf.losses.CategoricalCrossentropy()`
            #   by realizing that entropy can be computed using cross-entropy.



        # Perform an optimizer step and return the loss for reporting and visualization.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"loss": loss}

    # Predict method, with manual @tf.function for efficiency.
    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, states) :
        return self(states)


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

            # TODO: Predict the action using the greedy policy

            action_prob, value = network.predict([state])

            #dist = tfp.distributions.Categorical(probs=action_prob, dtype=tf.float32)

            #action = np.argmax(tf.squeeze(dist.probs_parameter()))
            action = np.argmax(action_prob)
            state, reward, done, _ = env.step(action)
            rewards += reward
        return rewards

    # Create the vectorized environment
    venv = gym.vector.AsyncVectorEnv([lambda: gym.make(env.spec.id)] * args.workers)
    venv.seed(args.seed)

    # Training
    state = venv.reset()


    training = True
    iteration = 0
    c = 0
    while training:
        temp_state = state
        temp_action_prob = None
        temp_value = None
        # Collect experience. Notably, we collect the following quantities
        # as tensors with the first two dimensions `[self.worker_steps, self.workers]`.
        states, next_states, actions, action_probs, rewards, dones, values = [], [], [], [], [], [], []
        for _ in range(args.worker_steps):
            # TODO: Choose `action`, which is a vector of `args.workers` actions, each
            # sampled from the corresponding policy generated by the `network.predict`
            # executed on the vector `state`.
            a = time.time()
            action_prob, value = network.predict(state)
            if temp_value is None:
                temp_value = value
            if temp_action_prob is None:
                temp_action_prob = action_prob
            b = time.time()
            c = c + (b - a)



            dist = tfp.distributions.Categorical(probs=action_prob, dtype=tf.int32)
            action = dist.sample()


            #todo: mozna bez numpy()

            action_prob = np.take_along_axis(action_prob, action.numpy().reshape((-1,1)), axis=1).reshape((-1))



            # Perform the step
            next_state, reward, done, _ = venv.step(action)

            # TODO: Collect the required quantities
            actions.append(action)
            values.append(value)
            rewards.append(reward)
            states.append(state)
            next_states.append(next_state)
            dones.append(done)
            action_probs.append(action_prob)


            state = next_state

        a = time.time()
        _, value = network.predict(state)
        b = time.time()
        c = c + (b - a)

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
        # print(advantages.shape)
        # print(np.array(values).shape)
        # print(np.array(values[:-1]).shape)
        # exit()
        values = np.array(values[:-1]).reshape((-1,args.workers))

        returns = advantages + values

        advantages -= tf.reduce_mean(advantages)
        advantages /= tf.math.reduce_std(advantages) + 1e-8
        advantages = tf.cast(advantages, dtype=tf.float32)


        # TODO: Estimate `advantages` and `returns` (they differ only by the value function estimate)
        # using lambda-return with coefficients `args.trace_lambda` and `args.gamma`.
        # You need to process episodes of individual workers independently, and note that
        # each worker might have generated multiple episodes, the last one probably unfinished.



        states = np.concatenate(states)




        targets = {"actions": np.concatenate(actions),
                 "action_probs": np.concatenate(action_probs),
                 "advantages": np.concatenate(advantages),
                 "returns": np.concatenate(returns)}

        # Train using the Keras API.
        # - The below code assumes that the first two dimensions of the used quantities are
        #   `[self.worker_steps, self.workers]` and concatenates them together.
        # - We do not log the training by passing `verbose=0`; feel free to change it.

        network.fit(
            states,
            targets,
            batch_size=args.batch_size,
            epochs=args.epochs, verbose=0,
        )

        returns = []
        # Periodic evaluation
        for _ in range(args.evaluate_for):
            val = evaluate_episode()
            returns.append(val)

        # print(np.mean(returns))
        # print(c)
        if np.mean(returns) > 498:

            for _ in range(7 * args.evaluate_for):
                val = evaluate_episode()
                returns.append(val)

            print('more accurate returns')
            print(np.mean(returns))

            if np.mean(returns) > 498:
                network.save_weights("ppo_weights")
                training = False


            #tf.keras.models.save_model(network._sds_model, f"paac_continuous_sd_model")

        # Periodic evaluation
        iteration += 1
        if iteration % args.evaluate_each == 0:
            for _ in range(args.evaluate_for):
                evaluate_episode()

    # Final evaluation
    while True:
        import zipfile

        # Get the current working directory
        #with zipfile.ZipFile("13.zip", 'r') as zip_ref:
        #    zip_ref.extractall()
        #network = Network(env, args)

        #network = tf.keras.models.load_model("ppo_model.h5", custom_objects={"Network": Network})

        del network
        network = Network(env.observation_space, env.action_space, args)
        network.load_weights("ppo_weights")
        # network = tf.keras.models.load_model("ppo_direct_model_save" , custom_objects={"Network": Network})

        #print("model loaded")
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)



    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("SingleCollect-v0"), args.seed)

    main(env, args)