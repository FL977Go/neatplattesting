"""Manual play script for the EASY version of NEAT platformer.
Controls: A / Left = move left, D / Right = move right, W / Up / Space = jump

Run with your venv active:
    python play_manual_easy.py
"""
import os
# ensure headless mode is disabled for manual play
os.environ.pop('HEADLESS', None)

import pygame
from pygame.locals import K_a, K_d, K_w, K_LEFT, K_RIGHT, K_UP, K_SPACE, QUIT
import neatplat_easy as neatplat  # Use easy version

pygame.init()
# use neatplat's screen size values
SCW, SCH = int(neatplat.SCwidth), int(neatplat.SChight)
screen = pygame.display.set_mode((SCW, SCH))
pygame.display.set_caption('Manual Play - EASY MODE - NEAT Platformer')
font = pygame.font.SysFont(None, 28)
clock = pygame.time.Clock()

# Start from clean state
neatplat.reset()

running = True
step = 0
total_fitness = 0.0

while running:
    for ev in pygame.event.get():
        if ev.type == QUIT:
            running = False

    keys = pygame.key.get_pressed()
    A = keys[K_a] or keys[K_LEFT]
    D = keys[K_d] or keys[K_RIGHT]
    W = keys[K_w] or keys[K_UP] or keys[K_SPACE]

    # advance simulation by one step using user actions
    (fitness, enemyX, enemyY, X, Y, emenmydis, goaldis, Y_vel, on_ground, enemy_dx, death) = neatplat.sim(A, D, W)
    total_fitness += fitness
    step += 1

    # render
    screen.fill((200, 200, 200))
    try:
        screen.blit(neatplat.floor1, (round(SCW * 0.01), neatplat.floor_y))
    except Exception:
        screen.blit(neatplat.floor1, (round(SCW * 0.01), int(neatplat.SChight * 0.82)))
    screen.blit(neatplat.enemy1, (round(enemyX), round(enemyY)))
    # EASY: Goal is at 50% (must match collision box in neatplat_easy.py line 46)
    screen.blit(neatplat.win, (round(SCW * 0.50), round(neatplat.floor_y - neatplat.win.get_height())))
    screen.blit(neatplat.player, (round(X), round(Y)))

    # HUD
    hud_lines = [
        '🎮 EASY MODE',
        f'Step: {step}',
        f'Fitness (last step): {fitness:.2f}',
        f'Total fitness: {total_fitness:.2f}',
        f'Enemy: ({enemyX:.1f}, {enemyY:.1f}) [STATIONARY]',
        f'Player: ({X:.1f}, {Y:.1f})',
        f'Goal distance: {goaldis:.1f}',
        f'Death: {death}',  # Debug: show death status
    ]
    for i, line in enumerate(hud_lines):
        surf = font.render(line, True, (0, 0, 0))
        screen.blit(surf, (10, 8 + i * 22))

    pygame.display.flip()
    clock.tick(60)

    if death:
        # show message briefly then reset
        if fitness >= 600:  # Won! (matches the +600 reward in neatplat_easy.py)
            # Create larger, more visible win message
            win_font = pygame.font.SysFont(None, 72)
            msg = win_font.render('🎉 YOU WON! 🎉', True, (0, 255, 0))
            msg2 = font.render('Reached the goal! Resetting in 2 seconds...', True, (0, 200, 0))
            # Draw semi-transparent background for better visibility
            overlay = pygame.Surface((SCW, SCH))
            overlay.set_alpha(128)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))
            screen.blit(msg, (SCW//2 - msg.get_width()//2, SCH//2 - 50))
            screen.blit(msg2, (SCW//2 - msg2.get_width()//2, SCH//2 + 30))
            pygame.display.flip()
            pygame.time.delay(2000)
        else:  # Died
            msg = font.render('💀 You Died - resetting...', True, (255, 0, 0))
            screen.blit(msg, (SCW//2 - 200, SCH//2))
            pygame.display.flip()
            pygame.time.delay(1000)
        neatplat.reset()
        total_fitness = 0.0
        step = 0

pygame.quit()
