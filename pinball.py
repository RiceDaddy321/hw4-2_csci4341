import os
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.optimizers import Adam
from keras.layers import Conv2D, Dense, Flatten

random.seed(0)  # make results reproducible
# tf.set_random_seed(0)

# num_episodes = 10

# input states, output q values
class DQN(tf.keras.Model):
    def __init__(self, action_size, state_size):
        super(DQN, self).__init__()
        self.conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                            input_shape=state_size)
        self.conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')
        self.flatten = Flatten()
        self.fc = Dense(512, activation='relu')
        self.fc_out = Dense(action_size)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        q = self.fc_out(x)
        return q


# Agent
class DQNAgent:
    def __init__(self, action_size, state_size=(84, 84, 4)):

        # states and actions
        self.state_size = state_size
        self.action_size = action_size

        # Hyper-parameters
        # Erik was here
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.02
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = self.epsilon_start - self.epsilon_end
        self.epsilon_decay_step /= self.exploration_steps
        self.batch_size = 32
        self.train_start = 50000
        self.update_target_rate = 10000

        # replay memory
        self.memory = deque(maxlen=100000)
        # no actions in the beginning
        self.no_op_steps = 30

        # model and target model
        self.model = DQN(action_size, state_size)
        self.target_model = DQN(action_size, state_size)
        self.optimizer = Adam(self.learning_rate, clipnorm=10.)
        # target model initialization
        self.update_target_model()

        self.avg_q_max, self.avg_loss = 0, 0

        self.writer = tf.summary.create_file_writer('summary/breakout_dqn')
        # self.model_path = os.path.join(os.getcwd(), 'save_model', 'model')

    # update target model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # select action
    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(history)
            return np.argmax(q_value[0])

    # store sample
    def append_sample(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    # record log
    def draw_tensorboard(self, score, step, episode):
        with self.writer.as_default():
            tf.summary.scalar('Total Reward/Episode', score, step=episode)
            tf.summary.scalar('Average Max Q/Episode',
                              self.avg_q_max / float(step), step=episode)
            tf.summary.scalar('Duration/Episode', step, step=episode)
            tf.summary.scalar('Average Loss/Episode',
                              self.avg_loss / float(step), step=episode)

    # training with randomly selected samples
    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        # random samples
        batch = random.sample(self.memory, self.batch_size)

        history = np.array([sample[0][0] / 255. for sample in batch],
                           dtype=np.float32)
        actions = np.array([sample[1] for sample in batch])
        rewards = np.array([sample[2] for sample in batch])
        next_history = np.array([sample[3][0] / 255. for sample in batch],
                                dtype=np.float32)
        dones = np.array([sample[4] for sample in batch])

        # parameter to train
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            # predict
            predicts = self.model(history)
            one_hot_action = tf.one_hot(actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

            # target
            target_predicts = self.target_model(next_history)
            max_q = np.amax(target_predicts, axis=1)
            targets = rewards + (1 - dones) * self.discount_factor * max_q

            # loss
            # error = tf.abs(targets - predicts)
            # quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
            # linear_part = error - quadratic_part
            loss = tf.reduce_mean(tf.square(targets - predicts))

            self.avg_loss += loss.numpy()

        # update cnn
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))


# color to gray
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    # making env and agent
    #env = gym.make('BreakoutDeterministic-v4')  # without rendering
    #env = gym.make("ALE/VideoPinball-v5", render_mode='human') # with rendering

    env = gym.make("ALE/VideoPinball-v5") # without rendering

    agent = DQNAgent(action_size=3)

    save_path = './save_model/my_model'
    if os.path.exists(save_path):
        agent.model = tf.keras.models.load_model(save_path)

    global_step = 0
    score_avg = 0
    score_max = 0

    action_dict = {0: 1, 1: 2, 2: 3}

    num_episode = 5000
    for e in range(num_episode):
        done = False
        dead = False

        step, score, start_life = 0, 0, 2
        # env initialization
        observe = env.reset()

        # no actions for first 30 steps
        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(1)

        # preparing states (4 images) only for the first history.
        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:

            global_step += 1
            step += 1

            # action
            action = agent.get_action(history)
            real_action = action_dict[action]

            # when missing ball, give an action to continue game
            if dead:
                action, real_action, dead = 0, 1, False

            # send action to env and receive state and reward
            observe, reward, done, info = env.step(real_action)
            # preprocessing per step
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            agent.avg_q_max += np.amax(agent.model(np.float32(history / 255.))[0])

            if start_life > info['lives']:
                dead = True
                start_life = info['lives']

            score += reward
            # storing sample
            agent.append_sample(history, action, reward, next_history, dead)

            # enough samples in memory, then start tainning
            if len(agent.memory) >= agent.train_start:
                agent.train_model()
                # update target model
                if global_step % agent.update_target_rate == 0:
                    agent.update_target_model()

            if dead:
                history = np.stack((next_state, next_state,
                                    next_state, next_state), axis=2)
                history = np.reshape([history], (1, 84, 84, 4))
            else:
                history = next_history

            if done:
                # record log per episode
                if global_step > agent.train_start:
                    agent.draw_tensorboard(score, step, e)

                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                score_max = score if score > score_max else score_max

                # printing log
                log = "episode: {:5d} | ".format(e)
                log += "score: {:4.1f} | ".format(score)
                log += "score max : {:4.1f} | ".format(score_max)
                log += "score avg: {:4.1f} | ".format(score_avg)
                log += "memory length: {:5d} | ".format(len(agent.memory))
                log += "epsilon: {:.3f} | ".format(agent.epsilon)
                log += "q avg : {:3.2f} | ".format(agent.avg_q_max / float(step))
                log += "avg loss : {:3.2f}".format(agent.avg_loss / float(step))
                print(log)

                agent.avg_q_max, agent.avg_loss = 0, 0

        # save model
        if e % 100 == 0:
            agent.model.save("./save_model/my_model")
