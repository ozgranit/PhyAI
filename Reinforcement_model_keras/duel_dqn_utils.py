import os
import pickle
import matplotlib.pyplot as plt
from rl.callbacks import Callback


class SaveAndPlot(Callback):
    def __init__(self, filepath, interval):
        super().__init__()
        self.filepath = filepath
        self.interval = interval
        self.total_steps, self.episode_rewards = load_reward()


    def on_step_end(self, step, logs={}):
        """ Save weights at interval steps during training """
        self.total_steps += 1
        if self.total_steps % self.interval != 0:
            # Nothing to do.
            return

        print('Step {}: saving model to {}'.format(self.total_steps, self.filepath))
        self.model.save_weights(self.filepath, overwrite=True)
        save_and_plot_reward(self.episode_rewards, self.total_steps)

    def on_episode_end(self, episode, logs):
        """ Update reward value at the end of each episode """
        self.episode_rewards.append(logs['episode_reward'])


def plot_loss(TrainReward, timestep):
    plt.clf()
    plt.xlabel('Sessions')
    plt.ylabel('Reward, Summed over entire session')
    plt.plot(range(len(TrainReward)), TrainReward, label="Train-Reward")
    plt.legend()
    plt.title("Performance, timstep={}".format(timestep))
    plt.savefig('Duel-DQN-Performance.png')


def load_reward():
    start = 0
    TrainReward = []

    TRAIN_REWARD_FILE = 'TrainReward.pkl'
    if os.path.isfile(TRAIN_REWARD_FILE):
        with open(TRAIN_REWARD_FILE, 'rb') as f:
            TrainReward = pickle.load(f)
            # stored start as last value, pop removes last val from lst
            start = TrainReward.pop()
            print('Load %s ...' % TRAIN_REWARD_FILE)
    return start, TrainReward


def save_and_plot_reward(TrainReward, timestep):
    TRAIN_REWARD_FILE = 'TrainReward.pkl'
    # Dump statistics to pickle
    with open(TRAIN_REWARD_FILE, 'wb') as f:
        # save time_step as last item at lst
        TrainReward.append(timestep)
        pickle.dump(TrainReward, f)
        # remove time_step (last item) from lst
        TrainReward.pop()
    plot_loss(TrainReward, timestep)
