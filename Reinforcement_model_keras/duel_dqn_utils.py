import os
import pickle
import matplotlib.pyplot as plt
from rl.callbacks import Callback


class SaveAndPlot(Callback):
    def __init__(self, filepath, interval):
        super().__init__()
        self.filepath = filepath
        self.interval = interval
        self.episode_rewards = load_reward()
        self.total_steps = 0

    def on_step_end(self, step, logs={}):
        """ Save weights at interval steps during training """
        self.total_steps += 1
        if self.total_steps % self.interval != 0:
            # Nothing to do.
            return

        print('Step {}: saving model to {}'.format(self.total_steps, self.filepath))
        self.model.save_weights(self.filepath, overwrite=True)
        save_and_plot_reward(self.episode_rewards)

    def on_episode_end(self, episode, logs):
        """ Update reward value at the end of each episode """
        self.episode_rewards.append(logs['episode_reward'])


def plot_loss(TrainReward):
    plt.clf()
    plt.xlabel('Sessions')
    plt.ylabel('Reward, Summed over entire session')
    plt.plot(range(len(TrainReward)), TrainReward, label="Train-Reward")
    plt.legend()
    plt.title("Performance")
    plt.savefig('Duel-DQN-Performance.png')


def load_reward():
    TrainReward = []

    TRAIN_REWARD_FILE = 'TrainReward.pkl'
    if os.path.isfile(TRAIN_REWARD_FILE):
        with open(TRAIN_REWARD_FILE, 'rb') as f:
            TrainReward = pickle.load(f)
            print('Load %s ...' % TRAIN_REWARD_FILE)
    return TrainReward


def save_and_plot_reward(TrainReward):
    TRAIN_REWARD_FILE = 'TrainReward.pkl'
    # Dump statistics to pickle
    with open(TRAIN_REWARD_FILE, 'wb') as f:
        # save time_step as last item at lst
        pickle.dump(TrainReward, f)
    plot_loss(TrainReward)
