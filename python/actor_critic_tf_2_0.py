import gym
import numpy as np
import tensorflow as tf


# https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/8_Actor_Critic_Advantage/AC_CartPole.py
# https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752

class Critic:
    def __init__(self, hidden_size, lr, reward_discount_rate):
        self.reward_discount_rate = reward_discount_rate

        init_weight = tf.random_normal_initializer(mean=0, stddev=0.12)
        init_bias = tf.constant_initializer(0.12)
        self.model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_size, activation='relu', kernel_initializer=init_weight, bias_initializer=init_bias),
        tf.keras.layers.Dense(units=1, activation=None, kernel_initializer=init_weight, bias_initializer=init_bias)
        ])

        self.optimizer = tf.optimizers.Adam(lr)

    @tf.function
    def train(self, state, reward, state_new):
        with tf.GradientTape() as tape:
            state, state_new = state[np.newaxis, :], state_new[np.newaxis, :]
            td_error = reward + self.reward_discount_rate * self.model(state_new) - self.model(state)
            loss = tf.square(td_error)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return td_error


class Actor:
    def __init__(self, hidden_size, lr, env_actions):
        init_weight = tf.random_normal_initializer(mean=0, stddev=0.12)
        init_bias = tf.constant_initializer(0.12)

        self.model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_size, activation='relu', kernel_initializer=init_weight, bias_initializer=init_bias),
        tf.keras.layers.Dense(units=env_actions, activation="softmax", kernel_initializer=init_weight, bias_initializer=init_bias)
        ])

        self.optimizer = tf.optimizers.Adam(lr)

    @tf.function
    def train(self, state, action, temp_diff_error):
        with tf.GradientTape() as tape:
            state = state[np.newaxis, :]
            log_prob = tf.math.log(self.model(state)[0, action])
            exp_v = tf.reduce_mean(input_tensor=log_prob * temp_diff_error)
            loss = -exp_v

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return exp_v

    def predict(self, s):
        s = s[np.newaxis, :]
        probs = self.model(s)
        return np.random.choice(np.arange(probs.shape[1]), p=probs.numpy().flatten())


def train_actor(params):
    print("simultaneous training of actor and critic")
    # create open ai environment
    gym_env = gym.make('CartPole-v0').unwrapped

    # actor class
    actor = Actor(params.hidden_size, params.learning_rate_actor, gym_env.action_space.n)

    # critic class; usually lr for critic is bigger (we want strong critic)
    critic = Critic(params.hidden_size, params.learning_rate_critic, params.reward_discount_rate)

    running_reward = 0

    for ep in range(params.episode_threshold):
        # reset state to simulate new episode
        state = gym_env.reset()

        reward_ep = 0

        for time in range(params.timestamp_threshold):
            # predict action with actor
            a = actor.predict(state)

            # simulate environment state with this action
            state_new, reward_step, done, info = gym_env.step(a)

            # if we are done we need to inroduce some penalty; we do not want in done (game over) state
            reward_step = -params.reward_done_penalty if done else reward_step

            reward_ep += reward_step

            # train critic using old/new state and reward
            temp_diff_error = critic.train(state, reward_step, state_new)

            # train actor using error from critic
            actor.train(state, a, temp_diff_error)

            state = state_new

            if done:
                break


        running_reward = running_reward * 0.9 + reward_ep * 0.1
        print("episode : {} reward : {:.2f}".format(ep, running_reward))

        # if we are already good enough let's do early stop
        if running_reward > params.reward_threshold:
            print("training is finished by reaching early stop condition")
            break

    gym_env.close()



    return actor


def try_actor(actor, params):
    print("actor testing")

    # create open ai environment
    gym_env = gym.make('CartPole-v0').unwrapped
    state = gym_env.reset()

    for time in range(params.timestamp_threshold):
        # render current environment
        gym_env.render()

        # take an action
        a = actor.predict(state)

        # simulate environment with this action
        state, reward_step, done, info = gym_env.step(a)

        if done:
            break

    gym_env.close()

class Param():
    def __init__(self):
        # how many learning episodes we have at a maximum
        self.episode_threshold = 1500
        # early stop reward threshold
        self.reward_threshold = 200
        # penalty if we are done (game over)
        self.reward_done_penalty = 16

        # hiddens size of our network
        self.hidden_size = 24
        # reward sicount rate
        self.reward_discount_rate = 0.9
        # learning rate for actor
        self.learning_rate_actor = 0.001
        # learning rate for critic; we want our critic to learn faster
        self.learning_rate_critic = 0.01
        # timestamps in one episode
        self.timestamp_threshold = 1000


def main():
    # default training params
    params = Param()

    # train actor with reinforcement learning
    actor = train_actor(params)

    # test trained actor
    try_actor(actor, params)

if __name__ == "__main__":
    main()
