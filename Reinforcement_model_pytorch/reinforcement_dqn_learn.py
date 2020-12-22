import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from dqn_utils import load_model_and_reward, save_model_and_plot
from enviorment import env_reset, play_action


USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
print('******* Running on {} *******'.format('CUDA' if USE_CUDA else 'CPU'))


# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
	classname = m.__class__.__name__
	# for every Linear layer in a model..
	if classname.find('Linear') != -1:
		# get the number of the inputs
		n = m.in_features
		y = 1.0 / np.sqrt(n)
		m.weight.data.uniform_(-y, y)
		m.bias.data.fill_(0)


def dqn_learning(
		q_func,
		time_steps,
		exploration,
		input_size,
		num_actions,
		steps_limit,
		learning_rate=1e-5,
		batch_size=32,
		target_update_freq=10000,
):
	###########################################
	# HELPER FUNC - USES MAIN FUNC PARAMETERS #
	###########################################
	# Construct an epilson greedy policy with given exploration schedule
	def get_epsilon_greedy_action(model, state, step):
		sample = random.random()
		eps_threshold = exploration.value(step)
		if sample > eps_threshold:
			with torch.no_grad():
				max_action = model(state).argmax()
				return max_action.numpy()
		else:
			return random.randint(0, num_actions-1)

	###############
	# BUILD MODEL #
	###############

	if USE_CUDA:
		Q = q_func(input_size=input_size, num_actions=num_actions).cuda()
		Q_target = q_func(input_size=input_size, num_actions=num_actions).cuda()
	else:
		Q = q_func(input_size=input_size, num_actions=num_actions).double()
		Q_target = q_func(input_size=input_size, num_actions=num_actions).double()

	# initialize weights
	Q.apply(weights_init_uniform_rule)

	# start Q_target where Q starts
	Q_target.load_state_dict(Q.state_dict())

	################################
	# Check & load pretrained model and loss
	# TrainReward holds the rewards of every session,
	# the reward of a session is sum of rewards start to finish, rewards can be negative
	start, TrainReward = load_model_and_reward(Q, Q_target)
	total_reward = 0
	################################

	# Optimizer
	criterion = nn.L1Loss()
	optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

	LOG_EVERY_N_STEPS = 100000

	# reset environment
	state = env_reset()
	steps_taken = 0

	"""problems:
		1. no replay buffer yet - the training samples aren't iid.
		2. we don't use the finite horizon properly - the state space does not represent how many actions were taken.
		3. loss and optimier might need to change- common are Huberloss and RMSporp optimizer.
		4. a discounting factor gamma might be required although we limit the num of steps.
		5. 380 viable actions are too much for the model to converge on.
		6. a decent idea might be to use two graph NN in a RL-type way - one to choose the cut node, and another to choose the paste node,
		maybe train them in order - first train the cut network (and reward it using the best possible paste value, wich well calculate exsutivly)
		 then train the paste network.
		 a different approch will be to train them as we do on Q and Q_target, fix one and train the other, than replace.
		 maybe reading about DDQN will help
		 
	"""
	for t in range(start, time_steps):
		###################
		# ACTUAL TRAINING #
		###################
		action = get_epsilon_greedy_action(Q, state, t)
		action = torch.tensor(action)

		next_state, reward = play_action(state, action)
		total_reward += reward

		state_action_value = Q(state)[action]
		next_state_value = Q_target(next_state).detach().max()

		# compute the Bellman error
		expected_state_action_value = next_state_value + reward

		loss = criterion(expected_state_action_value, state_action_value)

		# Backward + Optimize
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		# advance to next state
		state = next_state
		steps_taken += 1
		# played max number of steps allowed or at local maxima - start new game
		# next_state_value < 0 - means the q_func estimates that taking more actions will yield loss
		if steps_taken >= steps_limit or next_state_value < 0:
			# start over
			state = env_reset()
			steps_taken = 0
			TrainReward.append(total_reward)
			total_reward = 0

		#####################
		# UPDATE TARGET DQN #
		#####################
		if t % target_update_freq == 0:
			Q_target.load_state_dict(Q.state_dict())

		##########################
		# STATISTICS AND LOGGING #
		##########################

		if t % LOG_EVERY_N_STEPS == 0 and t > 1:
			print(TrainReward[-1])
			save_model_and_plot(Q, Q_target, TrainReward, t)

	return TrainReward
