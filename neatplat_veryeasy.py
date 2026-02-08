# from time import sleep
import os
import pygame
from random import random
from math import sqrt
# Only create GUI/Display resources when not running headless
HEADLESS = os.environ.get('HEADLESS') == '1'
if not HEADLESS:
    from tkinter import Tk
    root = Tk()
    pygame.display.init()
# from keyboard import is_pressed
# from ctypes import windll

# Screen size
SCwidth = 1920*0.8
SChight = 1080*0.8
# SCREEN = pygame.display.set_mode((round(SCwidth*0.9),round(SChight*0.9)))
running = True
playerXL = (SCwidth*0.033,SCwidth*0.033)
player = pygame.Surface(playerXL)
player.fill((255,255,255))
floor1S = (SCwidth*0.85,SChight*0.1)
floor1 = pygame.Surface(floor1S)
floor1.fill((0,0,0))
# enemy size should be relative to screen width and height
enemy1S = (SCwidth*0.033, SCwidth*0.033)
enemy1 = pygame.Surface(enemy1S)
enemy1.fill((200,0,0))
winS = (SCwidth*0.1,SChight*0.2)
win = pygame.Surface(winS)
win.fill((0,200,0))
colpla = player.get_rect()
colfl1 = floor1.get_rect()
colem1 = enemy1.get_rect()
colwin = win.get_rect()
# compute floor/top coordinates consistently
floor_y = round(SChight * 0.82)
# place floor and win relative to floor_y
colfl1 = floor1.get_rect()
colfl1.topleft = (round(SCwidth * 0.01), floor_y)
colem1 = enemy1.get_rect()
# place win so its bottom sits on the floor
colwin.topleft = (round(SCwidth * 0.40), floor_y - win.get_height())  # EASY: Goal much closer
# hwnd = pygame.display.get_wm_info()['window']
# windll.user32.SetForegroundWindow(hwnd)
X = 20
Y = 300
cooldown = 0
gravity = SChight*0.0000907407
acsellaration = 1.01
enemyX = int(SCwidth*0.25)  # EASY: Fixed enemy position instead of random
# place enemy on the floor (top = floor_y - enemy height)
enemyY = floor_y - enemy1.get_height()
enemyISR = 0
enemyISL = 0
fly = 0
prev_Y = 300
prev_enemyX=enemyX
# def blit(X,Y,enemyX,enemyY):
#     pygame.display.flip()
#     SCREEN.fill((200, 200, 200))
#     SCREEN.blit(player, (round(X), round(Y)))
#     SCREEN.blit(floor1, (30, 800))
#     SCREEN.blit(enemy1, (enemyX, enemyY))
#     SCREEN.blit(win, (SCwidth * 0.7, (800 - SChight * 0.1)))
def enemy_move():
    global enemyX,enemyISR,enemyISL
    if X > enemyX:
        enemyX += SCwidth*((0.1603 + ((random()*0.01)-0.005))/1536)
        enemyISR = 1
        enemyISL = 0
    if X < enemyX:
        enemyX -= SCwidth*((0.1603 + ((random()*0.01)-0.005))/1536)
        enemyISR = 0
        enemyISL = 1
        # decrease speed till good point found.
def sim(A,D,W):
    global acsellaration,cooldown,gravity,X,Y,enemyX,enemyY,fly,prev_enemyX,prev_Y
    fitness: float = 0
    death: bool = False
    # enemy_move()  # EASY: Stationary enemy - no chasing AI
    colpla.topleft = (round(X), round(Y))
    # keep floor and enemy rects in sync with positions
    colfl1.topleft = (round(SCwidth * 0.01), floor_y)
    colem1.topleft = (round(enemyX), round(enemyY))
    if W and cooldown <= 0 or not fly == 0:
        fly += 1
        Y -= SChight * 0.00069444
        if fly == 180:  # EASY: Shorter jump duration
            fly = 0
    if not colpla.colliderect(colfl1):
        Y += gravity * acsellaration
        acsellaration *= 1.005
    else:
        cooldown -= 3
        acsellaration = 1.1
        Y -= 0.3
    if colpla.top > SChight * 0.9:
        # player fell off bottom - reset and penalize like other deaths
        X = 20
        Y = 300
        cooldown = 0
        gravity = 9.8 * 0.008
        acsellaration = 1.01
        enemyX = int(SCwidth*0.25)  # EASY: Fixed enemy position
        enemyY = floor_y - enemy1.get_height()
        fitness -= 50
        death = True
    if colpla.colliderect(colem1):
        X = 20
        Y = 300
        cooldown = 0
        gravity = 9.8 * 0.008
        acsellaration = 1.01
        enemyX = int(SCwidth*0.25)  # EASY: Fixed enemy position
        enemyY = floor_y - enemy1.get_height()
        fitness -= 100
        death = True
    if colpla.colliderect(colwin):
        X = 20
        Y = 300
        cooldown = 0
        gravity = 9.8 * 0.008
        acsellaration = 1.01
        enemyX = int(SCwidth*0.25)  # EASY: Fixed enemy position
        enemyY = floor_y - enemy1.get_height()
        fitness += 600
        death = True
    if D:
        X += SCwidth*0.0001953
    if A:
        X-=SCwidth*0.0001953
    # keep X within reasonable screen bounds so it doesn't run away
    player_width = player.get_width() if hasattr(player, 'get_width') else round(SCwidth * 0.02)
    X = max(0, min(X, SCwidth - player_width))
    Ae = Y - enemyY
    Be = X - enemyX
    # compute goal vector relative to the win rect
    Ag = Y - colwin.top
    Bg = X - colwin.left
    emenmydis = sqrt((Ae ** 2) + (Be ** 2))
    goaldis = sqrt((Ag ** 2) + (Bg ** 2))
    Y_vel = Y - prev_Y
    on_ground = 1.0 if colpla.colliderect(colfl1) else 0.0
    enemy_dx = enemyX - prev_enemyX
    fitness +=0.01
    fitness = float(fitness)
    prev_Y = Y
    prev_enemyX=enemyX

    return (
    fitness,
    enemyX,
    enemyY,
    X,
    Y,
    emenmydis,
    goaldis,
    Y_vel,
    on_ground,
    enemy_dx,
    death
)

def reset():
    """Reset the environment globals to their starting values.

    Call this before evaluating a new genome so each genome starts from
    the same initial state.
    """
    global X, Y, cooldown, gravity, acsellaration, enemyX, enemyY, enemyISR, enemyISL, fly, prev_Y, prev_enemyX
    X = 40
    Y = 300
    cooldown = 0
    gravity = 9.8 * 0.008
    acsellaration = 1.01
    # place enemy on the floor (top = floor_y - enemy height)
    enemyX = int(SCwidth*0.25)  # EASY: Fixed enemy position
    enemyY = floor_y - enemy1.get_height()
    enemyISR = 0
    enemyISL = 0
    fly = 0
    prev_Y = Y
    prev_enemyX = enemyX
fitness = 0
if __name__ == "__main__":
    for _ in range(1000):
        # pygame.display.flip()
        # blit(X,Y,enemyX,enemyY)
        fitness,enemyX,enemyY,X,Y,eemnydis,gdiss = sim(random(),random(),random(),fitness)
        #print(random())
    # print(round(fitness))
