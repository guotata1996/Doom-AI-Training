import math
import numpy as np
import cv2
from vizdoom import *
from oblige import *
import threading
import time
import os

'''
None: not in sight
angle: 0~360
'''
def mapPos2cameraPix(x, y, ai_x, ai_y, ai_angle, resolution_x = 640):
    relative_x = x - ai_x
    relative_y = y - ai_y
    raid_angle = ai_angle * math.pi / 180.0

    #+/- calculation
    target_vector = [relative_x, relative_y]
    if raid_angle <= math.pi / 2 or raid_angle > math.pi * 1.5:
        ai_vector = [1, math.tan(raid_angle)]
    else:
        ai_vector = [-1, -math.tan(raid_angle)]

    pixel_value = depth = None
    if relative_x != 0 and relative_y != 0 and target_vector[0]*ai_vector[0] + target_vector[1]*ai_vector[1] > 0:
        slope = math.tan(raid_angle)
        #line function: (-slope) * x + 1 * y = 0
        offset = abs(-slope * relative_x + relative_y) / math.sqrt(slope **2 + 1)
        depth = math.sqrt(relative_x ** 2 + relative_y ** 2 - offset ** 2)

        #if offset <= depth:
            #using: 90 degree sight so depth = max_offset
        pixel_absvalue = offset*0.5 / depth * resolution_x

        if target_vector[0]*ai_vector[1] > target_vector[1]*ai_vector[0]:
            pixel_value = resolution_x / 2 + pixel_absvalue
        else:
            pixel_value = resolution_x / 2 - pixel_absvalue

        pixel_value = min(pixel_value, resolution_x - 1)
        pixel_value = max(pixel_value, 0)

    return pixel_value, depth

def mapHeight2cameraPix(z, ai_z, depth):
    pix_pos = (- (z - 41 - ai_z) / depth * 383.59 + 201)
    pix_pos = min(pix_pos, 402)
    pix_pos = max(pix_pos, 0)
    return pix_pos

def issamebox(bbox1, bbox2):
    locationsame = inside(bbox1[0], bbox1[1], bbox2) or inside(bbox1[0] + bbox1[2], bbox1[1], bbox2) \
        or inside(bbox1[0], bbox1[1] + bbox1[3], bbox2) or inside(bbox1[0] + bbox1[2], bbox1[1] + bbox1[3], bbox2) \
        or inside(bbox2[0], bbox2[1], bbox1) or inside(bbox2[0] + bbox2[2], bbox2[1], bbox1) \
        or inside(bbox2[0], bbox2[1] + bbox2[3], bbox1) or inside(bbox2[0] + bbox2[2], bbox2[1] + bbox2[3], bbox1)
    #sizesame = 0.75 < bbox1[2] * bbox1[3] / (bbox2[2] * bbox2[3] + 0.01) < 1.25
    return locationsame

def inside(x, y, bbox):
    return bbox[0] <= x <= bbox[0] + bbox[2] and bbox[1] <= y <= bbox[1] + bbox[3]


#type2name = ['reddoor','bluedoor','yellowdoor','dooropen','teleport','exit','redkey','bluekey','yellowkey']

def genPos():
    seed = 110
    global_index = 0
    end_seed = 150
    capture_freq = 1

    global_index = global_index * capture_freq
    game = DoomGame()
    game.load_config('O:\\Doom\\viz_2018\\maps\\finddoor\\finddoor_human.cfg')
    game.add_available_game_variable(GameVariable.POSITION_X)
    game.add_available_game_variable(GameVariable.POSITION_Y)
    game.add_available_game_variable(GameVariable.ANGLE)
    game.add_available_game_variable(GameVariable.USER1)
    game.add_available_game_variable(GameVariable.USER2)
    game.add_available_game_variable(GameVariable.USER3)
    game.add_available_game_variable(GameVariable.USER4)
    game.add_available_game_variable(GameVariable.USER5)
    game.add_available_game_variable(GameVariable.USER6)
    game.add_available_game_variable(GameVariable.USER7)
    game.add_available_game_variable(GameVariable.USER8)
    game.add_available_game_variable(GameVariable.USER9)
    game.add_available_game_variable(GameVariable.USER10)
    game.add_available_game_variable(GameVariable.USER11)
    game.add_available_game_variable(GameVariable.USER12)
    game.add_available_game_variable(GameVariable.USER13)
    game.add_available_game_variable(GameVariable.USER14)
    game.add_available_game_variable(GameVariable.USER15)
    game.add_available_game_variable(GameVariable.USER16)
    game.add_available_game_variable(GameVariable.USER17)
    game.add_available_game_variable(GameVariable.USER18)
    game.add_available_game_variable(GameVariable.USER19)
    game.add_available_game_variable(GameVariable.USER20)
    game.add_available_game_variable(GameVariable.USER21)
    game.add_available_game_variable(GameVariable.USER22)
    game.add_available_game_variable(GameVariable.USER23)
    game.add_available_game_variable(GameVariable.USER24)
    game.add_available_game_variable(GameVariable.USER25)
    game.add_available_game_variable(GameVariable.USER26)
    game.add_available_game_variable(GameVariable.USER27)
    game.add_available_game_variable(GameVariable.USER28)
    game.add_available_game_variable(GameVariable.USER29)
    game.add_available_game_variable(GameVariable.USER30)
    game.add_available_game_variable(GameVariable.USER31)
    game.add_available_game_variable(GameVariable.USER32)
    game.add_available_game_variable(GameVariable.USER33)
    game.add_available_game_variable(GameVariable.USER34)
    game.add_available_game_variable(GameVariable.USER35)
    game.add_available_game_variable(GameVariable.USER36)
    game.add_available_game_variable(GameVariable.USER37)
    game.add_available_game_variable(GameVariable.USER38)
    game.add_available_game_variable(GameVariable.USER39)
    game.add_available_game_variable(GameVariable.USER40)
    game.set_labels_buffer_enabled(True)

    while True:
        seed += 1
        if seed > end_seed:
            break
        if False:
            gen = DoomLevelGenerator()
            gen.generate("{}.wad".format(seed))

        game.set_doom_scenario_path("O:\\Doom\\scenarios\\autogen\\{}_m.wad".format(seed))
        mapinfo = open('O:\\Doom\\scenarios\\autogen\\{}_m.txt'.format(seed))
        lined = mapinfo.readlines()
        class Spot:
            def __init__(self, type, x1, y1, x2, y2, zl, zh):
                self.type = type
                self.x1 = x1
                self.y1 = y1
                self.x2 = x2
                self.y2 = y2
                self.floor_height = zl
                self.ceiling_height = zh
        mapspots = []
        for line in lined:
            digits = line.split(' ')
            mapspots.append(Spot(int(digits[1]), int(digits[2]), int(digits[3]), int(digits[4]), int(digits[5]), int(digits[6]), int(digits[7])))
        game.set_mode(Mode.SPECTATOR)
        game.init()
        cv2.namedWindow('bbox')

        last_x, last_y = -100, -100
        while True:
            screen = game.get_state().screen_buffer
            vars = [game.get_game_variable(USER1), game.get_game_variable(USER2), game.get_game_variable(USER3), game.get_game_variable(USER4),
            game.get_game_variable(USER5), game.get_game_variable(USER6), game.get_game_variable(USER7), game.get_game_variable(USER8),
            game.get_game_variable(USER9), game.get_game_variable(USER10),game.get_game_variable(USER11),game.get_game_variable(USER12),
            game.get_game_variable(USER13),game.get_game_variable(USER14),game.get_game_variable(USER15),game.get_game_variable(USER16),
            game.get_game_variable(USER17),game.get_game_variable(USER18),game.get_game_variable(USER19),game.get_game_variable(USER20),
            game.get_game_variable(USER21),game.get_game_variable(USER22),game.get_game_variable(USER23),game.get_game_variable(USER24),
            game.get_game_variable(USER25),game.get_game_variable(USER26),game.get_game_variable(USER27),game.get_game_variable(USER28),
            game.get_game_variable(USER29),game.get_game_variable(USER30),game.get_game_variable(USER31),game.get_game_variable(USER32),
            game.get_game_variable(USER33),game.get_game_variable(USER34),game.get_game_variable(USER35),game.get_game_variable(USER36),
            game.get_game_variable(USER37),game.get_game_variable(USER38),game.get_game_variable(USER39),game.get_game_variable(USER40)]
            for i in range(40):
                vars[i] = int(vars[i])

            all_sights = np.zeros([200])
            for i in range(40):
                bi_code = str(bin(vars[i]))
                for j in range(-1, -6, -1):
                    if bi_code[j] == '1':
                        all_sights[i * 5 - j - 1] = 1
                    elif bi_code[j] == 'b':
                        break
            pos_x = game.get_game_variable(GameVariable.POSITION_X)
            pos_y = game.get_game_variable(GameVariable.POSITION_Y)
            pos_z = game.get_game_variable(GameVariable.POSITION_Z)
            angle = game.get_game_variable(GameVariable.ANGLE)
            fordisp = np.zeros([640, 480], dtype = np.uint8)
            forwrite = []

            bboxes = []

            for i in np.where(all_sights > 0)[0]:
                if mapspots[i] is None:
                    continue
                if (pos_x - last_x)**2 + (pos_y - last_y)**2 < 5:
                    continue

                a_x, a_depth = mapPos2cameraPix(mapspots[i].x1, mapspots[i].y1, pos_x, pos_y, angle)
                b_x, b_depth = mapPos2cameraPix(mapspots[i].x2, mapspots[i].y2, pos_x, pos_y, angle)
                if a_x is None or b_x is None or abs(a_x - b_x) < 10:
                    continue
                a_y = max(mapHeight2cameraPix(mapspots[i].floor_height, pos_z, a_depth), mapHeight2cameraPix(mapspots[i].floor_height, pos_z, b_depth))
                b_y = min(mapHeight2cameraPix(mapspots[i].ceiling_height, pos_z, b_depth), mapHeight2cameraPix(mapspots[i].ceiling_height, pos_z, a_depth))
                if a_y - b_y <= 0:
                    continue

                fordisp[min(int(a_x), int(b_x)):max(int(a_x), int(b_x)), int(b_y) : int(a_y)] = 255

                bbox_for_check = [min(a_x, b_x), abs(a_x - b_x), b_y, a_y - b_y]
                bbox = [(a_x+b_x)/2.0/640.0, (a_y + b_y)/2.0/480.0, abs(a_x - b_x)/640.0, (a_y - b_y)/480.0]
                close_check = [issamebox(bbox_for_check, oldbox) for oldbox in bboxes]

                type = mapspots[i].type
                if type >= 3:
                    type -= 3

                #print(close_check)
                if not np.any(close_check):
                    forwrite.append("{} {} {} {} {}".format(type, *bbox))
                    bboxes.append(bbox_for_check)

            labels = game.get_state().labels
            label_buffer = game.get_state().labels_buffer

            for l in labels:
                if l.object_name == 'BlueCard' or l.object_name == 'YellowCard' or l.object_name == 'RedCard':
                    y_pos, x_pos = np.where(label_buffer[:402,:] == l.value)
                    if len(x_pos) > 0 and np.max(y_pos) - np.min(y_pos) <= 130 and np.max(x_pos) - np.min(x_pos) <= 130: #label buffer bug ?
                        fordisp[np.min(x_pos):np.max(x_pos), np.min(y_pos):np.max(y_pos)] = 255
                        if l.object_name == 'BlueCard':
                            kind = 7
                        elif l.object_name == 'YellowCard':
                            kind = 8
                        else:
                            kind = 6

                        forwrite.append(
                            "{} {} {} {} {}".format(kind, (np.min(x_pos) + np.max(x_pos))/2.0/640.0,
                                                    (np.min(y_pos) + np.max(y_pos))/2.0/480.0,
                                                    (np.max(x_pos) - np.min(x_pos))/640.0,
                                                    (np.max(y_pos) - np.min(y_pos))/480.0))


            cv2.imshow('bbox',np.transpose(fordisp))
            if len(forwrite) > 0:
                if global_index % capture_freq == 0:
                    f = open('O:\\Doom\\detection\\doom_dataset\\labels\\{}.txt'.format(global_index // capture_freq), 'w')
                    for l in forwrite:
                        f.write(l + '\n')
                    f.close()
                    screen_trans = screen.transpose(1,2,0)
                    screen_trans = cv2.cvtColor(screen_trans, cv2.COLOR_BGR2RGB)
                    cv2.imwrite('O:\\Doom\\detection\\doom_dataset\\image\\{}.jpg'.format(global_index // capture_freq), screen_trans)
                global_index += 1

            last_x = pos_x
            last_y = pos_y

            game.advance_action()
            ac = game.get_last_action()
            if game.is_episode_finished():
                game.new_episode()
            if ac[5] > 0:
                game.close()
                print('last seed = {}'.format(seed))
                break
            elif ac[4] > 0:
                for _ in range(70):
                    game.advance_action()
                for i in range(len(mapspots)):
                    if mapspots[i] is not None and mapspots[i].type < 3 and (mapspots[i].x1 - pos_x)**2 + (mapspots[i].y1 - pos_y)**2 + (mapspots[i].x2 - pos_x)**2 + (mapspots[i].y2 - pos_y)**2 < 50000:
                        mapspots[i] = None

            if game.is_episode_finished():
                game.new_episode()
            cv2.waitKey(3)
            
def genNeg():
    game = DoomGame()
    game.load_config('O:\\Doom\\scenarios\\human.cfg')
    game.set_doom_scenario_path("O:\\Doom\\viz_2018\\maps\\navigation\\115_m.wad")
    game.init()
    global_index = 0
    caputure_freq = 4
    global_index = global_index*caputure_freq
    while True:
        game.advance_action()
        ac = game.get_last_action()
        if global_index % caputure_freq == 0:
            screen = game.get_state().screen_buffer
            screen = screen.transpose(1,2,0)
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            cv2.imwrite('O:\\Doom\\detection\\neg_dataset\\JPEGImages\\{}.jpg'.format(global_index // caputure_freq), screen)
        
        global_index += 1

def genVali():
    game = DoomGame()
    game.load_config('O:\\Doom\\scenarios\\human.cfg')
    game.set_doom_scenario_path("O:\\Doom\\scenarios\\autogen\\100_m.wad")
    game.init()
    global_index = 0
    caputure_freq = 4
    global_index = global_index*caputure_freq
    while True:
        game.advance_action()
        ac = game.get_last_action()
        if global_index % caputure_freq == 0:
            screen = game.get_state().screen_buffer
            screen = screen.transpose(1,2,0)
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            cv2.imwrite('O:\\Doom\\detection\\validate\\{}.jpg'.format(global_index // caputure_freq), screen)
        
        global_index += 1        
        
if __name__ == '__main__':
    genVali()