from reinforcement_model import DQN, NUM_ACTIONS, INPUT_SIZE
from reinforcement_dqn_learn import dqn_learning
from dqn_utils import LinearSchedule, plot_loss

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
STEPS_LIMIT = 100
TARGET_UPDATE_FREQ = 10000


def main(time_steps):
	exploration_schedule = LinearSchedule(2500000, 0.05)

	TrainReward = dqn_learning(
		q_func=DQN,
		time_steps=time_steps,
		exploration=exploration_schedule,
		input_size=INPUT_SIZE,
		num_actions=NUM_ACTIONS,
		steps_limit=STEPS_LIMIT,
		learning_rate=LEARNING_RATE,
		batch_size=BATCH_SIZE,
		target_update_freq=TARGET_UPDATE_FREQ,
	)

	plot_loss(TrainReward)


if __name__ == '__main__':

	time_steps = 5000001
	# Run training
	main(time_steps)
