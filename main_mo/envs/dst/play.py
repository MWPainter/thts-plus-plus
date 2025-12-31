import pygame
import numpy as np
import time

import deep_sea_treasure
from deep_sea_treasure import DeepSeaTreasureV0
from deep_sea_treasure import VamplewWrapper

# Make sure experiment are reproducible, so people can use the exact same versions
print(f"Using DST {deep_sea_treasure.__version__.VERSION} ({deep_sea_treasure.__version__.COMMIT_HASH})")

# dst = DeepSeaTreasureV0.new(
# 	max_steps=1000,
# 	max_velocity=2.0,
# 	swept_by_current_prob=0.2,
# 	render_treasure_values=True
# )

# dst.render()

# stop: bool = False
# time_reward: int = 0

# while not stop:
# 	events = pygame.event.get()

# 	action = (np.asarray([0, 0, 0, 1, 0, 0, 0]), np.asarray([0, 0, 0, 1, 0, 0, 0]))

# 	for event in events:
# 		if event.type == pygame.KEYDOWN:
# 			if event.key == pygame.K_LEFT:
# 				action = (2,3)
# 				# action = (np.asarray([0, 0, 1, 0, 0, 0, 0]), np.asarray([0, 0, 0, 1, 0, 0, 0]))
# 			elif event.key == pygame.K_RIGHT:
# 				action = (4,3)
# 				# action = (np.asarray([0, 0, 0, 0, 1, 0, 0]), np.asarray([0, 0, 0, 1, 0, 0, 0]))
# 			if event.key == pygame.K_UP:
# 				action = (3,2)
# 				# action = (np.asarray([0, 0, 0, 1, 0, 0, 0]), np.asarray([0, 0, 1, 0, 0, 0, 0]))
# 			elif event.key == pygame.K_DOWN:
# 				action = (3,4)
# 				# action = (np.asarray([0, 0, 0, 1, 0, 0, 0]), np.asarray([0, 0, 0, 0, 1, 0, 0]))

# 			if event.key in {pygame.K_ESCAPE}:
# 				stop = True

# 		if event.type == pygame.QUIT:
# 			stop = True

# 	_, reward, done, debug_info = dst.step(action)
# 	time_reward += int(reward[1])

# 	if done:
# 		print(f"Found treasure worth {float(reward[0]):5.2f} after {abs(time_reward)} timesteps!")
# 		time_reward = 0

# 	if not stop:
# 		dst.render()
# 		time.sleep(0.25)

# 	if done:
# 		dst.reset()
		


dst = VamplewWrapper.new(DeepSeaTreasureV0.new(
	max_steps=1000,
	max_velocity=2.0,
	swept_by_current_prob=0.2,
	render_treasure_values=True
), True)

dst.render()

stop: bool = False
time_reward: int = 0

while not stop:
	events = pygame.event.get()

	action = 4

	for event in events:
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_LEFT:
				action = 3
			elif event.key == pygame.K_RIGHT:
				action = 1
			if event.key == pygame.K_UP:
				action = 0
			elif event.key == pygame.K_DOWN:
				action = 2

			if event.key in {pygame.K_ESCAPE}:
				stop = True

		if event.type == pygame.QUIT:
			stop = True

	_, reward, done, debug_info = dst.step(action)
	time_reward += int(reward[1])

	if done:
		print(f"Found treasure worth {float(reward[0]):5.2f} after {abs(time_reward)} timesteps!")
		time_reward = 0

	if not stop:
		dst.render()
		time.sleep(0.25)

	if done:
		dst.reset()