import numpy as np
import gym
from reinforcement_model import NUM_ACTIONS
from gym_env import PhyloTree

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

LEARNING_RATE = .001
GAMMA = .999

ACTOR_HIDDEN_LAYER = 512
CRITIC_HIDDEN_LAYER = 1024
ENV_NAME = 'PhyloTree-v0'


def main(time_steps):
	# Get the environment and extract the number of actions.
	env = PhyloTree()

	assert len(env.action_space.shape) == 2
	assert NUM_ACTIONS == env.action_space.shape[0] * env.action_space.shape[1]
	nb_actions = NUM_ACTIONS

	# Next, we build a very simple model.
	actor = Sequential()
	actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
	actor.add(Dense(ACTOR_HIDDEN_LAYER))
	actor.add(Activation('relu'))
	actor.add(Dense(ACTOR_HIDDEN_LAYER))
	actor.add(Activation('relu'))
	actor.add(Dense(ACTOR_HIDDEN_LAYER))
	actor.add(Activation('relu'))
	actor.add(Dense(nb_actions))
	actor.add(Activation('linear'))
	print(actor.summary())

	action_input = Input(shape=(nb_actions,), name='action_input')
	observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
	flattened_observation = Flatten()(observation_input)
	x = Concatenate()([action_input, flattened_observation])
	x = Dense(CRITIC_HIDDEN_LAYER)(x)
	x = Activation('relu')(x)
	x = Dense(CRITIC_HIDDEN_LAYER)(x)
	x = Activation('relu')(x)
	x = Dense(CRITIC_HIDDEN_LAYER)(x)
	x = Activation('relu')(x)
	x = Dense(1)(x)
	x = Activation('linear')(x)
	critic = Model(inputs=[action_input, observation_input], outputs=x)
	print(critic.summary())

	# Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
	# even the metrics!
	memory = SequentialMemory(limit=100000, window_length=1)
	random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
	agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
	                  memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
	                  random_process=random_process, gamma=GAMMA, target_model_update=1e-3)
	agent.compile(Adam(lr=LEARNING_RATE, clipnorm=1.), metrics=['mae'])

	# Okay, now it's time to learn something! We visualize the training here for show, but this
	# slows down training quite a lot. You can always safely abort the training prematurely using
	# Ctrl + C.
	agent.fit(env, nb_steps=time_steps, visualize=True, verbose=1, nb_max_episode_steps=200)

	# After training is done, we save the final weights.
	agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

	# Finally, evaluate our algorithm for 5 episodes.
	agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)


if __name__ == '__main__':

	time_steps = 50
	# Run training
	main(time_steps)
