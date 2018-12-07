import gym
import readchar # pip install readchar

# Start game
env = gym.make('Pong-v0')
env.reset()
env.render()

# Breakout-v0 actions
action_idx = env.unwrapped.get_action_meanings()
print(action_idx)
# The printing result shos "['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']"

# Key configuration
# You can change the keys by your taste.
dict_keys = {'r': 0, 'f': 1, 'd': 2, 'a': 3} # 0 = NOOP / 1 = FIRE / 2 = RIGHT / 3 = LEFT
# RIGHTFIRE and LEFTFIRE maybe useless in the testing

total_reward = 0 
while True:
	key = readchar.readkey()
	if key not in dict_keys.keys():
		print("Not allowed keys!")
		break
	action = dict_keys[key]
	frame, reward, terminal, _ = env.step(action)
	total_reward += reward
	env.render()
	
	if terminal:
		print("Finish (reward: %d)" % total_reward)
		break
