import numpy as np
import bio_methods
import env_aux
from rein_model import INPUT_SIZE, NUM_NODES
import gym
from gym import spaces


class PhyloTree(gym.Env):

	def render(self, mode='human'):
		pass

	def __init__(self):
		self.current_ete_tree = None
		self.current_bio_tree = None
		self.current_tree_str = None
		self.current_msa_path = None
		self.current_likelihood = 0
		self.likelihood_params = None

		# action_matrix contains the relevant Strings for action => cut_name, paste_name
		self.action_matrix = env_aux.get_action_matrix()

		# The multi-discrete action space consists of a series of discrete action spaces,
		# so first is cur, second is paste
		self.action_space = spaces.MultiDiscrete([NUM_NODES, NUM_NODES])

		# high value is completely arbitrary, change 4000 to whatever
		self.observation_space = spaces.Box(low=0.0, high=4000, shape=(NUM_NODES, NUM_NODES), dtype=np.float32)

		self.state = None
		self.reset()

	def step(self, action):
		assert self.action_space.contains(action)

		cut_name, paste_name = self.get_action(action)

		if cut_name is None and paste_name is None:
			"""choosing the same cut & paste is our reset button,
            this is the only way for the model to reset the run
            since we don't limit the steps taken"""
			return self.state, 0, True, {}

		if (self.current_ete_tree & cut_name).up.name == (self.current_ete_tree & paste_name).up.name:
			# check cut and paste have the same parent node
			# if preformed this causes issues, and is equivalent to no-op
			return self.state, 0, False, {}

		self.current_tree_str = bio_methods.SPR_by_edge_names(self.current_ete_tree, cut_name, paste_name)
		self.current_ete_tree, self.current_bio_tree = bio_methods.get_ete_and_bio_from_str(self.current_tree_str,
		                                                                                    self.current_msa_path)

		# make new tree into matrix
		next_state = bio_methods.tree_to_matrix(self.current_bio_tree)
		self.state = next_state

		# calculating reward
		new_likelihood = bio_methods.get_likelihood_simple(self.current_tree_str, self.current_msa_path,
		                                                   self.likelihood_params)
		reward = new_likelihood - self.current_likelihood
		self.current_likelihood = new_likelihood

		done = False
		return self.state, reward, done, {}

	def reset(self):
		"""take new tree from trainning datasets
		make tree into matrix
		save tree for play_action method
		return matrix as vector numpy"""

		# setting a random folder from the different msa folders
		self.current_msa_path = env_aux.set_random_msa_path()
		self.likelihood_params = bio_methods.calc_likelihood_params(self.current_msa_path)
		self.current_ete_tree, self.current_bio_tree, self.current_tree_str = bio_methods.get_tree_from_msa(
			self.current_msa_path)

		self.current_likelihood = bio_methods.get_likelihood_simple(self.current_tree_str, self.current_msa_path,
		                                                            self.likelihood_params)
		self.state = bio_methods.tree_to_matrix(self.current_bio_tree)

		return self.state

	def get_action(self, action_lst):
		assert len(action_lst) == 2

		# possible pairs: ('Sp000', 'Sp001')...('Sp019', 'Sp018')
		# allow duplicates ('Sp000', 'Sp001') and ('Sp001', 'Sp000')
		i = action_lst[0]
		j = action_lst[1]
		assert 0 <= i <= 38
		assert 0 <= j <= 38
		first, second = self.action_matrix[i][j]

		if first == second:
			return None, None  # no pairs of doubles allowed
		return first, second
