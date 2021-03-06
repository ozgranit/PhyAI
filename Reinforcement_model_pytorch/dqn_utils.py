import os
import sys
import pickle
import matplotlib.pyplot as plt

import torch


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


def plot_loss(TrainReward):
    plt.clf()
    plt.xlabel('Sessions')
    plt.ylabel('Reward, Summed over entire session')
    plt.plot(range(len(TrainReward)), TrainReward, label="Train-Reward")
    plt.legend()
    plt.title("Performance")
    plt.savefig('DQN-Performance.png')


def load_model_and_reward(model, model_target):
    if os.path.isfile('Q_params.pkl'):
        print('Load Q parameters ...')
        model.load_state_dict(torch.load('Q_params.pkl'))

    if os.path.isfile('Q_target_params.pkl'):
        print('Load Q_target parameters ...')
        model_target.load_state_dict(torch.load('Q_target_params.pkl'))

    # load prev Stats
    start = 0
    TrainReward = []  # TrainLoss holds the average loss of last 100 time_steps

    TRAIN_REWARD_FILE = 'TrainReward.pkl'
    if os.path.isfile(TRAIN_REWARD_FILE):
        with open(TRAIN_REWARD_FILE, 'rb') as f:
            TrainReward = pickle.load(f)
            # stored start as last value, pop removes last val from lst
            start = TrainReward.pop()
            print('Load %s ...' % TRAIN_REWARD_FILE)

    return start, TrainReward


def save_model_and_plot(model, model_target, TrainReward, time_step):
    print("Timestep %d" % (time_step,))
    # print("Train loss %f" % TrainLoss[-1])
    sys.stdout.flush()

    # Save the trained model
    torch.save(model.state_dict(), 'Q_params.pkl')
    torch.save(model_target.state_dict(), 'Q_target_params.pkl')
    TRAIN_REWARD_FILE = 'TrainReward.pkl'
    # Dump statistics to pickle
    with open(TRAIN_REWARD_FILE, 'wb') as f:
        # save time_step as last item at lst
        TrainReward.append(time_step)
        pickle.dump(TrainReward, f)
        # remove time_step (last item) from lst
        TrainReward.pop()

    print("Saved Stats")
    plot_loss(TrainReward)
