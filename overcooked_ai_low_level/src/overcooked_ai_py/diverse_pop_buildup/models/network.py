import tensorflow as tf
import tensorflow_probability as tfp
import gym
import argparse

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
parser.add_argument("--hidden_layer_size", default=32, type=int, help="Size of hidden layer.")
parser.add_argument("--layer_stddev_init", default=0.01, type=float, help="stddev of layer init.")
parser.add_argument("--activation", default="tanh", help="Size of hidden layer.")
parser.add_argument("--actor_learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--critic_learning_rate", default=0.003, type=float, help="Learning rate.")
parser.add_argument("--trace_lambda", default=0.95, type=float, help="Traces factor lambda.")
parser.add_argument("--workers", default=16, type=int, help="Workers during experience collection.")
parser.add_argument("--worker_steps", default=100, type=int, help="Steps for each worker to perform.")
parser.add_argument("--total_timesteps", default=350000, type=int, help="Total timesteps of experiments")

args = parser.parse_args([] if "__file__" not in globals() else None)
args.batch_size = int(args.workers * args.worker_steps)

class Network(tf.keras.Model):
    def __init__(self, observation_space_shape, action_space_shape) -> None:
        super(Network, self).__init__()
        self.args = args

        inputs = tf.keras.layers.Input(observation_space_shape)

        hidden_policy = tf.keras.layers.Dense(args.hidden_layer_size, activation=args.activation,kernel_initializer=tf.keras.initializers.RandomNormal(stddev=args.layer_stddev_init))(inputs)
        hidden_policy = tf.keras.layers.Dense(args.hidden_layer_size, activation=args.activation,kernel_initializer=tf.keras.initializers.RandomNormal(stddev=args.layer_stddev_init))(hidden_policy)
        policy = tf.keras.layers.Dense(action_space_shape)(hidden_policy)

        self._actor_model = tf.keras.Model(inputs=inputs, outputs=policy)
        self._actor_optimizer = tf.optimizers.Adam(args.actor_learning_rate)
        self._actor_model.compile(optimizer=self._actor_optimizer)

        hidden_value = tf.keras.layers.Dense(args.hidden_layer_size, activation=args.activation,kernel_initializer=tf.keras.initializers.RandomNormal(stddev=args.layer_stddev_init))(inputs)
        hidden_value = tf.keras.layers.Dense(args.hidden_layer_size, activation=args.activation,kernel_initializer=tf.keras.initializers.RandomNormal(stddev=args.layer_stddev_init))(hidden_value)
        value = tf.keras.layers.Dense(1)(hidden_value)

        self._critic_model = tf.keras.Model(inputs=inputs, outputs=value)
        self._critic_optimizer = tf.optimizers.Adam(args.critic_learning_rate)
        self._critic_model.compile(optimizer=self._critic_optimizer)

    @tf.function
    def train_step(self, data):
        states, targets = data
        old_probs = targets["action_probs"]

        with tf.GradientTape() as actor_tape:
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
    def predict_action_prob_logits(self, states):
        return self._actor_model(states)

    @tf.function
    def predict_values(self, states):
        return self._critic_model(states)