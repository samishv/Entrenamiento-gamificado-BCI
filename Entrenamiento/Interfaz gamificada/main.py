from game import Game
import pygame

g = Game()

while g.running:
    g.curr_menu.display_menu()

    if g.playing:
        g.run()
        g.playing = False
        g.curr_menu = g.main_menu

pygame.quit()