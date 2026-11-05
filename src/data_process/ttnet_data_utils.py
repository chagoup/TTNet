"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.05.21
# email: nguyenmaudung93.kstn@gmail.com
# project repo: https://github.com/maudzung/TTNet-Realtime-for-Table-Tennis-Pytorch
-----------------------------------------------------------------------------------
# Description: utils for TTNet dataset
"""

import os
import json
import sys
from collections import Counter

import cv2
from sklearn.model_selection import train_test_split
import torch
import numpy as np

sys.path.append('../')


def load_raw_img(img_path):
    """Load raw image based on the path to the image"""
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)  # BGR --> RGB
    return img


def gaussian_1d(pos, muy, sigma):
    """Create 1D Gaussian distribution based on ball position (muy), and std (sigma)"""
    target = torch.exp(- (((pos - muy) / sigma) ** 2) / 2)
    return target


def create_target_ball(ball_position_xy, sigma, w, h, thresh_mask, device):
    """Create target for the ball detection stages

    :param ball_position_xy: Position of the ball (x,y)
    :param sigma: standard deviation (a hyperparameter)
    :param w: width of the resize image
    :param h: height of the resize image
    :param thresh_mask: if values of 1D Gaussian < thresh_mask --> set to 0 to reduce computation
    :param device: cuda() or cpu()
    :return:
    """
    w, h = int(w), int(h)
    target_ball_position = torch.zeros((w + h,), device=device)
    # Only do the next step if the ball is existed
    if (w > ball_position_xy[0] > 0) and (h > ball_position_xy[1] > 0):
        # For x
        x_pos = torch.arange(0, w, device=device)
        target_ball_position[:w] = gaussian_1d(x_pos, ball_position_xy[0], sigma=sigma)
        # For y
        y_pos = torch.arange(0, h, device=device)
        target_ball_position[w:] = gaussian_1d(y_pos, ball_position_xy[1], sigma=sigma)

        target_ball_position[target_ball_position < thresh_mask] = 0.

    return target_ball_position


def create_target_ball_right(ball_position_xy, sigma, w, h, thresh_mask, device):
    """Create target for the ball detection stages, returned as an (x, y) tuple."""
    w, h = int(w), int(h)
    target_ball_position_x = torch.zeros(w, device=device)
    target_ball_position_y = torch.zeros(h, device=device)
    # Only do the next step if the ball is existed
    if (w > ball_position_xy[0] > 0) and (h > ball_position_xy[1] > 0):
        # For x
        x_pos = torch.arange(0, w, device=device)
        target_ball_position_x = gaussian_1d(x_pos, ball_position_xy[0], sigma=sigma)
        # For y
        y_pos = torch.arange(0, h, device=device)
        target_ball_position_y = gaussian_1d(y_pos, ball_position_xy[1], sigma=sigma)

        target_ball_position_x[target_ball_position_x < thresh_mask] = 0.
        target_ball_position_y[target_ball_position_y < thresh_mask] = 0.

    return target_ball_position_x, target_ball_position_y


def smooth_event_labelling(event_class, smooth_idx, event_frameidx):
    target_events = np.zeros((2,))
    if event_class < 2:
        n = smooth_idx - event_frameidx
        target_events[event_class] = np.cos(n * np.pi / 8)
        target_events[target_events < 0.01] = 0.
    return target_events


def get_events_infor(game_list, configs, dataset_type):
    """Get information of sequences of images based on events

    :param game_list: List of games (video names)
    :return:
    [
        each event: [[img_path_list], ball_position, target_events, segmentation_path]
    ]
    """
    # the paper mentioned 25, but used 9 frames only
    num_frames_from_event = int((configs.num_frames_sequence - 1) / 2)

    annos_dir = os.path.join(configs.dataset_dir, dataset_type, 'annotations')
    images_dir = os.path.join(configs.dataset_dir, dataset_type, 'images')
    events_infor = []
    events_labels = []
    for game_name in game_list:
        ball_annos_path = os.path.join(annos_dir, game_name, 'ball_markup.json')
        events_annos_path = os.path.join(annos_dir, game_name, 'events_markup.json')
        # Load ball annotations
        json_ball = open(ball_annos_path)
        ball_annos = json.load(json_ball)

        # Load events annotations
        json_events = open(events_annos_path)
        events_annos = json.load(json_events)
        for event_frameidx, event_name in events_annos.items():
            event_frameidx = int(event_frameidx)
            smooth_frame_indices = [event_frameidx]  # By default
            if (event_name != 'empty_event') and (configs.smooth_labelling):
                smooth_frame_indices = [idx for idx in range(event_frameidx - num_frames_from_event,
                                                             event_frameidx + num_frames_from_event + 1)]

            for smooth_idx in smooth_frame_indices:
                sub_smooth_frame_indices = [idx for idx in range(smooth_idx - num_frames_from_event,
                                                                 smooth_idx + num_frames_from_event + 1)]
                img_path_list = []
                for sub_smooth_idx in sub_smooth_frame_indices:
                    img_path = os.path.join(images_dir, game_name, 'img_{:06d}.jpg'.format(sub_smooth_idx))
                    img_path_list.append(img_path)
                last_f_idx = smooth_idx + num_frames_from_event
                # Get ball position for the last frame in the sequence
                if '{}'.format(last_f_idx) not in ball_annos.keys():
                    print('smooth_idx: {} - no ball position for the frame idx {}'.format(smooth_idx, last_f_idx))
                    continue
                ball_position_xy = ball_annos['{}'.format(last_f_idx)]
                ball_position_xy = np.array([ball_position_xy['x'], ball_position_xy['y']], dtype=np.int64)
                # Ignore the event without ball information
                if (ball_position_xy[0] < 0) or (ball_position_xy[1] < 0):
                    continue

                # Get segmentation path for the last frame in the sequence
                seg_path = os.path.join(annos_dir, game_name, 'segmentation_masks', '{}.png'.format(last_f_idx))
                if not os.path.isfile(seg_path):
                    print("smooth_idx: {} - The segmentation path {} is invalid".format(smooth_idx, seg_path))
                    continue
                event_class = configs.events_dict[event_name]

                target_events = smooth_event_labelling(event_class, smooth_idx, event_frameidx)
                events_infor.append([img_path_list, ball_position_xy, target_events, seg_path])
                # Re-label if the event is neither bounce nor net hit
                if (target_events[0] == 0) and (target_events[1] == 0):
                    event_class = 2
                events_labels.append(event_class)
    return events_infor, events_labels


def train_val_data_separation(configs):
    """Seperate data to training and validation sets.

    默认按【视频】留出验证集，而不是随机切分事件样本。

    为什么必须按视频：原先用 train_test_split 随机切分，会让同一个视频的相邻帧
    同时进入训练集和验证集。TTNet 的输入是连续 9 帧窗口，相邻样本几乎相同，
    验证 loss 会因此虚低，早停和选模型都失去意义（等于在训练集上验证）。

    配置：--val-games 显式指定验证视频；
         未指定时按 --val-size 从 train_game_list 末尾留出（默认留 1 个视频）。
    """
    dataset_type = 'training'

    if configs.no_val:
        events_infor, events_labels = get_events_infor(
            configs.train_game_list, configs, dataset_type)
        return events_infor, None, events_labels, None

    train_games = list(configs.train_game_list)
    val_games = list(getattr(configs, 'val_games', None) or [])

    if val_games:
        # 显式指定：从训练列表里剔除这些视频
        train_games = [g for g in train_games if g not in val_games]
    else:
        # 未显式指定：留出最后 n 个视频作验证集
        n_val = max(1, int(round(len(train_games) * getattr(configs, 'val_size', 0.2))))
        # 至少保留 2 个视频用于训练
        n_val = min(n_val, max(1, len(train_games) - 2))
        val_games = train_games[-n_val:]
        train_games = train_games[:-n_val]

    if not train_games:
        raise ValueError('留出验证集后训练集为空，请减少 --val-size 或调整 --val-games')

    print('[划分] 训练视频: {}'.format(train_games))
    print('[划分] 验证视频: {}'.format(val_games))

    train_events_infor, train_events_labels = get_events_infor(
        train_games, configs, dataset_type)
    val_events_infor, val_events_labels = get_events_infor(
        val_games, configs, dataset_type)

    return train_events_infor, val_events_infor, train_events_labels, val_events_labels


if __name__ == '__main__':
    from config.config import parse_configs

    configs = parse_configs()
    train_events_infor, val_events_infor, train_events_labels, val_events_labels = train_val_data_separation(configs)
    print('Counter train_events_labels: {}'.format(Counter(train_events_labels)))
    if val_events_labels is not None:
        print('Counter val_events_labels: {}'.format(Counter(val_events_labels)))
    event_name = 'net'
    event_class = configs.events_dict[event_name]
    configs.device = torch.device('cpu')
    ball_position_xy = np.array([100, 50])
    target_ball_position = create_target_ball(ball_position_xy, sigma=0.5, w=320, h=128, thresh_mask=0.01,
                                              device=configs.device)

    max_val_x = (target_ball_position[:320]).max()
    max_val_y = (target_ball_position[320:]).max()
    target_ball_g_x = np.argmax(target_ball_position[:320])
    target_ball_g_y = np.argmax(target_ball_position[320:])
    print('max_val_x: {}, max_val_y: {}'.format(max_val_x, max_val_y))
    print('target_ball_g_x: {}, target_ball_g_x: {}'.format(target_ball_g_x, target_ball_g_y))
