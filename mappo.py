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
parser.add_argument("--agents", default=2, type=int, help="Agents to use.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=46, type=int, help="Random seed.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--clip_epsilon", default=0.15, type=float, help="Clipping epsilon.")
parser.add_argument("--entropy_regularization", default=0.2, type=float, help="Entropy regularization weight.")
parser.add_argument("--epochs", default=15, type=int, help="Epochs to train each iteration.")
parser.add_argument("--evaluate_each", default=10, type=int, help="Evaluate each given number of iterations.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=5e-3, type=float, help="Learning rate.")
parser.add_argument("--trace_lambda", default=0.95, type=float, help="Traces factor lambda.")
parser.add_argument("--workers", default=32, type=int, help="Workers during experience collection.")
parser.add_argument("--worker_steps", default=200, type=int, help="Steps for each worker to perform.")

# TODO(ppo): We use the exactly same Network as in the `ppo` assignment.
class Network(tf.keras.Model):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, args: argparse.Namespace) -> None:
        self.args = args

        # Create a suitable model for the given observation and action spaces.
        inputs = tf.keras.layers.Input(observation_space.shape)

        # TODO(ppo): Using a single hidden layer with args.hidden_layer_size and ReLU activation,
        # produce a policy with `action_space.n` discrete actions.
        hidden_policy = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(inputs)
        hidden_policy = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(hidden_policy)
        policy = tf.keras.layers.Dense(action_space.n, activation="softmax")(hidden_policy)

        # TODO(ppo): Using an independent single hidden layer with args.hidden_layer_size and ReLU activation,
        # produce a value function estimate. It is best to generate it as a scalar, not
        # a vector of length one, to avoid broadcasting errors later.
        hidden_value = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(inputs)
        hidden_value = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(hidden_value)
        value = tf.keras.layers.Dense(1)(hidden_value)

        # Construct the model
        super().__init__(inputs=inputs, outputs=[policy, value])

        # Compile using Adam optimizer with the given learning rate.
        self.compile(optimizer=tf.optimizers.Adam(args.learning_rate))

    # TODO(ppo): Define a training method `train_step`, which is automatically used by Keras.
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
            tf.summary.scalar("actor_loss", data = actor_loss, step=1)

        # Perform an optimizer step and return the loss for reporting and visualization.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"loss": loss}

    # Predict method, with manual @tf.function for efficiency.
    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, states: np.ndarray):
        return self(states)


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace):
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()

    # Construct the networks, each for the same observation space and corresponding action space.
    networks = [Network(env.observation_space, env.action_space[i], args) for i in range(args.agents)]

    def evaluate_episode(start_evaluation:bool = False) -> float:
        rewards, state, done = 0, env.reset(start_evaluation), False
        while not done:
            if args.render_each and (env.episode + 1) % args.render_each == 0:
                env.render()

            action = []
            # TODO: Predict the vector of actions using the greedy policy
            for agent in range(args.agents):
                action_prob, value = networks[agent].predict([state])
                act = np.argmax(action_prob)
                action.append(act)

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
    while training:
        # Collect experience. Notably, we collect the following quantities
        # as tensors with the first two dimensions `[self.worker_steps, self.workers]`,
        # and the third dimension being `self.agents` for `action*`, `rewards`, `values`.
        states, actions, action_probs, rewards, dones, values = [], [], [], [], [], []
        for _ in range(args.worker_steps):
            # TODO: Choose `action`, which is a vector of `args.agents` actions for each worker,
            # each action sampled from the corresponding policy generated by the `predict` of the
            # networks executed on the vector `state`.

            action = []
            action_prob = []
            value = []
            for a in range(args.agents):

                act_prob, val = networks[a].predict(state)

                # print(act_prob)
                dist = tfp.distributions.Categorical(probs=act_prob, dtype=tf.int32)
                act = dist.sample()
                act_prob = np.take_along_axis(act_prob, act.numpy().reshape((-1,1)), axis=1).reshape((-1))

                action.append(act)
                action_prob.append(act_prob)
                value.append(val)



            # Perform the step, extracting the per-agent rewards for training

            action = np.array(action).reshape((-1,args.agents))
            action_prob = np.array(action_prob).reshape((-1, args.agents))
            value = np.array(value).reshape((-1, args.agents))

            next_state, _, done, info = venv.step(action)
            reward = np.array([i["agent_rewards"] for i in info])



            # TODO: Collect the required quantities
            actions.append(action)
            values.append(value)
            rewards.append(reward)
            states.append(state)
            dones.append(done)
            action_probs.append(action_prob)

            state = next_state


        values.append(value)

        values = tf.cast(values, dtype=tf.float32)
        deltas = np.zeros((len(rewards), args.workers))

        for a in range(args.agents):
            # TODO: For the given agent, estimate `advantages` and `returns` (they differ only by the value
            # function estimate) using lambda-return with coefficients `args.trace_lambda` and `args.gamma`.
            # You need to process episodes of individual workers independently, and note that
            # each worker might have generated multiple episodes, the last one probably unfinished.

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

            mean_advantages = np.nanmean(advantages)
            std_advantages = np.nanstd(advantages)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-8)
            # advantages -= tf.reduce_mean(advantages)
            # advantages /= tf.math.reduce_std(advantages) + 1e-8
            # advantages = tf.cast(advantages, dtype=tf.float32)



            # Train the agent `a` using the Keras API.
            # - The below code assumes that the first two dimensions of the used quantities are
            #   `[self.worker_steps, self.workers]` and concatenates them together.
            # - The code further assumes `actions` and `action_probs` have shape
            #   `[self.worker_steps, self.workers, self.agents]`, and uses only values of agent `a`.
            #   If you use a different shape, please update the code accordingly.
            # - We do not log the training by passing `verbose=0`; feel free to change it.
            # print(np.array(states).shape)
            # print(np.array(actions)[:,a].shape)
            # print(np.array(action_probs)[:,a].shape)
            # print(np.array(advantages).shape)
            # print(np.array(returns).shape)


            networks[a].fit(
                np.concatenate(states),
                {"actions": np.concatenate(actions)[:, a],
                 "action_probs": np.concatenate(action_probs)[:, a],
                 "advantages": np.concatenate(advantages),
                 "returns": np.concatenate(returns)},
                batch_size=args.batch_size, epochs=args.epochs, verbose=0,
            )



        # Periodic evaluation
        iteration += 1
        if iteration % args.evaluate_each == 0:
            for _ in range(args.evaluate_for):
                evaluate_episode()

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("MultiCollect{}-v0".format(args.agents)), args.seed)



    main(env, args)
