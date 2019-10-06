import pygame
import random
# firs take video of the game then add the effects, pizza tossing
# SCORE: +1 hit -1 Miss
# RESET: if -1, if score == -1: reset ball

WIDTH = 400
HEIGHT = 500
FPS = 60
# decimalz
# ch?v=ztnfN1ONcos
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

move_it = 0


class Player(pygame.sprite.Sprite):

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.Surface((50, 10))
        self.image.fill(GREEN)

        self.rect = self.image.get_rect()

        self.rect.bottom = HEIGHT - 10
        self.rect.right = WIDTH / 2

    def update(self, key_pressed):
        self.hit_wall()
        speed = 0

        if key_pressed == pygame.K_RIGHT:
            speed = 7
        if key_pressed == pygame.K_LEFT:
            speed = -7

        self.rect.x += speed


    def get_player_position(self):
        position = [self.rect.x, self.rect.y]
        return position


    def hit_wall(self):

        if self.rect.right > WIDTH:
            self.rect.right = WIDTH

        if self.rect.left < 0:
            self.rect.left = 0
