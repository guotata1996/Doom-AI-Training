import numpy as np
import cv2
from baselines.common.vec_env import VecEnv
from skimage.measure import label

norm_factor = 24

def get_observation(s, real_frame = False, get_rid_of_value=[], resolution = 84):
    #returns: 84 x 84 x 1
    if not real_frame:
        if s is None:
            return np.zeros([resolution, resolution, 1], dtype=np.uint8)
        else:
            texture = depth2texture(s, get_rid_of_value)
            return np.expand_dims(cv2.resize(texture, (resolution, resolution)) * norm_factor, -1)
    else:
        if s is None:
            return np.zeros([resolution, resolution, 3], dtype=np.uint8)
        else:
            screen = s.screen_buffer
            screen = screen.transpose((1,2,0))
            return cv2.resize(screen, (resolution, resolution))


def depth2texture(state, get_rid_of_value=[]):
    '''
    for texture buffer:
    0 - ceiling
    1 - wall
    2 - floor
    3 - ammo
    4 - health
    5 - explosive box
    6 - armor
    7 - enemy
    8 - rocket
    '''
    depth_buffer = state.depth_buffer.astype(np.int32)[:202, :]
    texture_buffer = np.array(depth_buffer, dtype=np.uint8)
    h, w = depth_buffer.shape
    for i in range(w):
        max_val = np.max(depth_buffer[:, i])
        while np.sum(depth_buffer[:, i] == max_val) < 8:
            max_val -= 1
        max_args = np.where(depth_buffer[:, i] == max_val)[0]
        if len(max_args) == 0:
            wall_start_idx = np.argmax(depth_buffer[:, i])
        else:
            wall_start_idx = max_args[0]
        wall_end_idx = wall_start_idx
        while depth_buffer[wall_start_idx, i] == depth_buffer[wall_end_idx, i] and wall_end_idx < h - 1:
            wall_end_idx += 1
        # expand wall at resolution 1
        real_max = depth_buffer[wall_start_idx, i]
        if wall_end_idx - wall_start_idx < 10:
            while abs(depth_buffer[wall_start_idx, i] - real_max) <= 3 and wall_start_idx > 0:
                wall_start_idx -= 1
            while abs(depth_buffer[wall_end_idx, i] - real_max) <= 3 and wall_end_idx < h - 1:
                wall_end_idx += 1

        texture_buffer[:wall_start_idx, i] = 0
        texture_buffer[wall_start_idx:wall_end_idx, i] = 1
        texture_buffer[wall_end_idx:, i] = 2
    # label buffer
    for l in state.labels:
        if l.value in state.labels_buffer[:202, :]:
            pos = np.where(state.labels_buffer[:202, :] == l.value)
            if l.object_name == 'RocketAmmo' and 3 not in get_rid_of_value:
                texture_buffer[pos] = 3
            if l.object_name == 'Medikit' and 4 not in get_rid_of_value:
                texture_buffer[pos] = 4
            if l.object_name == 'ExplosiveBarrel' and 5 not in get_rid_of_value:
                texture_buffer[pos] = 5
            if l.object_name.endswith('Armor') and 6 not in get_rid_of_value:
                texture_buffer[pos] = 6
            if l.object_name == "DoomPlayer" and 7 not in get_rid_of_value:
                bi_value_buffer = np.zeros_like(depth_buffer, dtype=np.uint8)
                bi_value_buffer[pos] = 1
                connection_areas, num_labels = label(bi_value_buffer, return_num=True, connectivity=2)
                for area_label in range(1, num_labels+1):
                    label_pos_x, label_pos_y = np.where(connection_areas == area_label)
                    x_min = np.min(label_pos_x)
                    x_max = np.max(label_pos_x)
                    y_min = np.min(label_pos_y)
                    y_max = np.max(label_pos_y)
                    texture_buffer[x_min:x_max+1, y_min: y_max+1] = 7
            if l.object_name == "Rocket" and 8 not in get_rid_of_value:
                texture_buffer[pos] = 8

    return texture_buffer
'''
def make_one_hot_image(data, depth):
    return (np.arange(depth) == data[:,:,None]).astype(np.uint8)
'''
def check_enemy_enter(obs):
    #returns true only when enemy appears in the middle of sight
    enemy_pixel = np.where(obs == 7 * norm_factor)
    return len(enemy_pixel[0]) >= 60

def check_enemy_leave(obs):
    enemy_pixel = np.where(obs == 7 * norm_factor)
    return len(enemy_pixel[0]) < 40

def check_bullet(obs):
    bullet_pixel = np.where(obs == 8 * norm_factor)
    return len(bullet_pixel) > 20

class VecEnvAdapter(VecEnv):
    def __init__(self, ordinary_env):
        self.env = ordinary_env
        self.nenv = len(ordinary_env)
        self.action_space = ordinary_env[0].action_space
        self.observation_space = ordinary_env[0].observation_space
        VecEnv.__init__(self, num_envs=self.nenv, observation_space=ordinary_env[0].observation_space, action_space=ordinary_env[0].action_space)

    def reset(self):
        a = np.stack([e.reset() for e in self.env])
        return np.stack([e.reset() for e in self.env])

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, rews, news, infos = [],[],[],[]
        for i in range(self.nenv):
            ob, rew, new, info = self.env[i].step(self._actions[i])
            obs.append(ob)
            news.append(new)
            rews.append(rew)
            infos.append(info)
        a = np.stack(obs)
        return np.stack(obs), np.stack(rews), np.stack(news), np.stack(infos)

    def close(self):
        for e in self.env:
            e.close()
