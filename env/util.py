import numpy as np
from vizdoom import *

def attemptUse(game, action_len = 4):
    noop_action = [0 for _ in range(action_len)]
    use_action = [0 for _ in range(action_len - 1)]+[1]

    old_screen_buffer = game.get_state().screen_buffer[:,:402,:]
    while True:
        game.make_action(noop_action)
        if game.is_episode_finished():
            return False
        screen_buffer = game.get_state().screen_buffer[:,:402,:]
        if len(np.where(screen_buffer - old_screen_buffer != 0)[0]) == 0:
            break
        old_screen_buffer = screen_buffer
    game.make_action(use_action)

    saved_buffer = []
    for i in range(7):
        saved_buffer.append(screen_buffer)
        if i < 6:
            game.make_action(noop_action)
            if game.is_episode_finished():
                return False
            screen_buffer = game.get_state().screen_buffer[:, :402, :]

    first_vary = saved_buffer[0][1, :, :] - saved_buffer[1][1, :, :] != 0
    first_green1 = saved_buffer[0][1, :, :] > saved_buffer[0][0, :, :]
    first_green2 = saved_buffer[0][1, :, :] > saved_buffer[0][2, :, :]
    second_same = saved_buffer[1][0, :, :] - saved_buffer[2][0, :, :] == 0
    third_same = saved_buffer[2][1, :, :] - saved_buffer[3][1, :, :] == 0
    forth_same = saved_buffer[4][2, :, :] - saved_buffer[3][2, :, :] == 0
    fifth_same = saved_buffer[5][2, :, :] - saved_buffer[4][2, :, :] == 0
    sixth_same = saved_buffer[6][2, :, :] - saved_buffer[5][2, :, :] == 0

    bright_enough = saved_buffer[1][1, :, :] > 158

    doorraise_conditions = True
    for i in range(6):
        for j in range(i + 1, 7):
            if len(np.where(saved_buffer[i] - saved_buffer[j] != 0)[0]) < 250000:
                doorraise_conditions = False
    panel_conditions = first_green1 * first_green2 * first_vary * second_same * third_same * forth_same * fifth_same * sixth_same * bright_enough

    valid_panel = len(np.where(panel_conditions == True)[0]) > 1200

    return doorraise_conditions or valid_panel

def checkTeleported(image):
    #3*w*h
    image_rgbmax = np.max(image, axis=0)
    if len(np.where(image_rgbmax > 250)[0]) > 90000:
        return True
    else:
        return False

def checkKey(image, color):
    '''
    :param image:
    :param color:r0 b1 y2
    :return: True if Has the key
    '''
    return (color == 0 and image[0, 469, 491] > 200)\
        or (color == 2 and image[0, 445, 491] > 190)\
        or (color == 1 and image[2, 421, 491] > 200)
