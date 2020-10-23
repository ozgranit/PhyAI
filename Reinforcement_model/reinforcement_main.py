from reinforcement_model import DQN
from reinforcement_dqn_learn import dqn_learning
from dqn_utils import LinearSchedule

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
STEPS_LIMITS = 7
NUM_ACTIONS = 20*20
INPUT_SIZE = 20*20


def main(time_steps):
	exploration_schedule = LinearSchedule(100000, 0.1)

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
	naive_loss = test_naive_model()
	# we don't always bother filling TestLoss
	try:
		print("LNN Best Loss: %f" % min(TestLoss[1]))
		print("Naive Model Loss: %f" % naive_loss)
		if naive_loss < min(TestLoss[1]):
			print("Naive Model did better.")
		else:
			print("LNN did better.")

		idx = TestLoss[1].index(min(TestLoss[1]))
		print("LNN Best Loss after %d Steps" % TestLoss[0][idx])
	except ValueError:
		print("No TestLoss, Moving on")
	# plot_loss(TrainLoss, TestLoss)


def test_LinearSchedule():
	exploration = LinearSchedule(100000, 0.1)
	for i in range(0, 130000, 10000):
		v = exploration.value(i)
		print(i, v)


if __name__ == '__main__':

	time_steps = 200001
	# Run training
	# main(time_steps)
	test_LinearSchedule()
