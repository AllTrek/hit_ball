import pygame
import random
from Player import Player


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


class Mob(pygame.sprite.Sprite):

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.Surface((20, 20))
        self.image.fill(RED)

        self.rect = self.image.get_rect()
        self.rect.x = random.randrange(10, WIDTH - 10)

        self.speedy = 5
        self.speedx = random.randrange(-3, 3)

        self.last_update = pygame.time.get_ticks()

    def update(self):

        self.hit_wall()
        self.rect.y += self.speedy
        self.rect.x += self.speedx

    def get_mob_position(self):
        position = [self.rect.x, self.rect.y]
        return position

    def hit_wall(self):

        if self.rect.right > WIDTH:
            self.rect.right = WIDTH
            self.speedx = -3

        elif self.rect.left < 0:
            self.rect.left = 0
            self.speedx = 3

        elif self.rect.top == 0:
            self.speedy = 5

        elif self.rect.bottom == HEIGHT:
            self.reset_ball()

    def hit_bottom(self):
        if self.rect.bottom == HEIGHT:
            return True
        else:
            return False

    def hit_player(self):
        self.speedy = -5
        self.speedx = random.randrange(-3, 3)

    def reset_ball(self):
        self.rect.y = 0
        self.speedy = 5
        self.speedx = random.randrange(-3, 3)

    def check_reset(self):
        if self.rect.bottom == HEIGHT:
            self.reset_ball()
            return True
        else:
            return False
