import numpy as np
import os
from gym_env import PhyloTree
from duel_dqn_utils import SaveAndPlot

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

LEARNING_RATE = 1e-5
GAMMA = .999
STEPS_LIMIT = 5
MEM_SIZE = 100000
TARGET_UPDATE_FREQ = 10000
HIDDEN_LAYER = 16

ENV_NAME = 'PhyloTree'
WEIGHT_FILENAME = 'duel_dqn_{}_weights.h5f'.format(ENV_NAME)
LOG_EVERY_N_STEPS = 100000


def main(time_steps):
	# Get the environment and extract the number of actions.
	env = PhyloTree()
	np.random.seed(123)
	env.seed(123)
	nb_actions = env.action_space.n

	# Next, we build a very simple model regardless of the dueling architecture
	# if you enable dueling network in DQN , DQN will build a dueling network base on your model automatically
	# Also, you can build a dueling network by yourself and turn off the dueling network in DQN.
	model = Sequential()
	model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
	model.add(Dense(HIDDEN_LAYER))
	model.add(Activation('relu'))
	model.add(Dense(HIDDEN_LAYER))
	model.add(Activation('relu'))
	model.add(Dense(HIDDEN_LAYER))
	model.add(Activation('relu'))
	model.add(Dense(nb_actions, activation='linear'))
	print(model.summary())

	memory = SequentialMemory(limit=MEM_SIZE, window_length=1)
	policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=0, nb_steps=time_steps*0.75)
	# enable the dueling network
	# you can specify the dueling_type to one of {'avg','max','naive'}
	dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50, gamma=GAMMA, enable_double_dqn=True,
	               enable_dueling_network=True, dueling_type='avg', target_model_update=TARGET_UPDATE_FREQ, policy=policy)

	# load model if exists
	if os.path.isfile(WEIGHT_FILENAME):
		print('Loading DQN parameters ...')
		dqn.load_weights(WEIGHT_FILENAME)

	dqn.compile(Adam(lr=LEARNING_RATE), metrics=['mae'])

	logger = SaveAndPlot(filepath=WEIGHT_FILENAME, interval=LOG_EVERY_N_STEPS)
	train_history = dqn.fit(env, nb_steps=time_steps, visualize=False, verbose=0, nb_max_episode_steps=STEPS_LIMIT, callbacks=[logger])
	train_rewards = train_history.history['episode_reward']

	# After training is done, we save the final weights.
	dqn.save_weights(WEIGHT_FILENAME, overwrite=True)

	# Finally, evaluate our algorithm for 5 episodes.
	dqn.test(env, nb_episodes=5, visualize=False)


if __name__ == '__main__':

	time_steps = 50
	# Run training
	main(time_steps)
