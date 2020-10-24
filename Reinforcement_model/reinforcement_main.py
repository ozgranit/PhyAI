from reinforcement_model import DQN
from reinforcement_dqn_learn import dqn_learning
from dqn_utils import LinearSchedule, plot_loss

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
STEPS_LIMITS = 100
NUM_ACTIONS = 2*2
INPUT_SIZE = 2*2


def main(time_steps):
	exploration_schedule = LinearSchedule(100000, 0.05)

	TrainReward = dqn_learning(
		q_func=DQN,
		time_steps=time_steps,
		exploration=exploration_schedule,
		input_size=INPUT_SIZE,
		num_actions=NUM_ACTIONS,
		steps_limit=STEPS_LIMITS,
		learning_rate=LEARNING_RATE,
		batch_size=BATCH_SIZE,
		target_update_freq=10000,
	)

	plot_loss(TrainReward)


if __name__ == '__main__':

	time_steps = 10001
	# Run training
	main(time_steps)
