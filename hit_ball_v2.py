import pygame
from Player import Player
from Mob import Mob
from Agent import DQNAgent

import numpy as np
import tensorflow as tf
from collections import deque
import random
import os


import time


print(tf.VERSION)
WIDTH = 400
HEIGHT = 500
FPS = 60

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


pygame.init()
clock = pygame.time.Clock()
display = pygame.display.set_mode((WIDTH, HEIGHT))
all_sprites = pygame.sprite.Group()
all_mobs = pygame.sprite.Group()

player = Player()
mob = Mob()

all_sprites.add(player)
all_sprites.add(mob)
all_mobs.add(mob)

################################################################################

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5

EPISODES = 2_000


epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

ACTION_SPACE_SIZE = 3

dqn_path = os.path.dirname(__file__)
save_model_path = os.path.join(dqn_path, "save_model")


new_observation = deque(maxlen=3)
observation = deque(maxlen=3)

agent = DQNAgent(REPLAY_MEMORY_SIZE, ACTION_SPACE_SIZE, MIN_REPLAY_MEMORY_SIZE, MINIBATCH_SIZE,
                 UPDATE_TARGET_EVERY, DISCOUNT)


#####################################################################################


def take_action(c_obs):

    if np.random.random() > epsilon:
        action_num = np.argmax(agent.get_qs(c_obs))

        if action_num == 0:
            action = pygame.K_LEFT
        elif action_num == 1:
            action = pygame.K_RIGHT
        else:
            action = None
    else:
        action = random.choice([pygame.K_LEFT, pygame.K_RIGHT, None])
        if action == pygame.K_LEFT:
            action_num = 0
        elif action == pygame.K_RIGHT:
            action_num = 1
        else:
            action_num = 2

    player.update(key_pressed=action)
    return action_num


def draw_text(surf, text, size, x, y):
    font_name = pygame.font.match_font('arial')
    font = pygame.font.Font(font_name, size)
    text_surface = font.render(text, True, WHITE)  # true antialis text
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    surf.blit(text_surface, text_rect)


def get_observation():
    global temp_observations
    global new_observations
    global current_observations
    global first_observations
    global action_num

    if len(temp_observations) == 3:  # If you have all the frames
        action_num = take_action(temp_observations)

        if len(current_observations) != 3:  # Check to see
            # first_observations = temp_observations
            current_observations = temp_observations
            temp_observations = []  # return first_observation. First Obs is saved

        elif len(new_observations) != 3:
            new_observations = temp_observations

        elif len(current_observations) == 3:
            # print()
            # print("Current observation :", current_observations)
            # print("New observation: ", new_observations)
            # print("Action: ", action_num)
            # print()
            #
            # current_observations = new_observations
            # new_observations = []
            # temp_observations = []

            return True

    else:
        temp_observations.append((mob.get_mob_position() + player.get_player_position()))


running = True
score = 0
start_time = time.time()

temp_observations = []
current_observations = []
new_observations = []
first_observations = []
action_num = 0
done = False
reward = 0
episodes = 0
update_model = True

while running:
    clock.tick(FPS)

    if time.time() - start_time > 0.3:  # 0.3
        start_time = time.time()
        all_observations = get_observation()

        hits = pygame.sprite.spritecollide(player, all_mobs, False)

        if not done:
            if hits:
                score += 1
                reward = 3
                mob.hit_player()

            elif abs(player.rect.x - mob.rect.x) > 200:
                reward = -0.2
            elif 200 > abs(player.rect.x - mob.rect.x) > 100:
                reward = -0.1
            elif 100 > abs(player.rect.x - mob.rect.x) > 50:
                reward = 0.1
            elif abs(player.rect.x - mob.rect.x) < 50:
                reward = 0.2

        if all_observations:

            # Even though screen height (y value) is 500 the ball resets before reaching 500. The highest value is
            # 480 which is achieved by the Y axis of the agent so 480 was used to normalize the data.

            current_observation_normalize = np.round([x / 480.0 for x in sum(current_observations, [])], decimals=3)
            new_observation_normalize = np.round([x / 480.0 for x in sum(new_observations, [])], decimals=3)

            # print("current obs: ", current_observations)
            # print("\nCurrent observation: ", current_observation_normalize)
            # print("Action: ", action_num)
            # print("New State: ", new_observation_normalize)
            # print("Reward: ", reward)
            # print("Done: ", done, '\n')

            agent.update_replay_memory((current_observation_normalize, action_num, reward, new_observation_normalize, done)) # three deciamls places
            agent.train(done)


            # Current, new, action, reward, done
            # check if falls reset_ball if == true then done == True
            # Once training is done with the observations call to reset it
            current_observations = new_observations
            new_observations = []
            temp_observations = []
            done = False
            if reward != 0:
                reward = 0

        else:  # Update every frame
            mob.update()
            reset_ball = mob.check_reset()

            if reset_ball:
                print("EPISODE: ", episodes, " SCORE: ", score)
                done = True
                score = 0
                reward = -3
                episodes += 1
                start_time = time.time()
                update_model = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # reset_ball = mob.check_reset()
    #
    # if reset_ball:
    #     score = 0
    #     start_time = time.time()

    if episodes % 50 == 0 and update_model:
        save_model = save_model_path + "/dqn_model_episode{}.h5".format(episodes)
        agent.model.save(save_model)
        update_model = False

    if episodes == EPISODES:
        running = False


    ############# DISPLAY ENV ######

    display.fill(BLACK)
    all_sprites.draw(display)

    draw_text(display, str(score), 30, WIDTH / 2, 3)
    pygame.display.flip()
    ##############################

pygame.quit()
