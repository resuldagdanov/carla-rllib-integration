#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


import collections.abc
import os
import shutil

import math
import cv2
import numpy as np
np.random.seed(0)

from tensorboard import program
from collections import deque


def post_process_image(image, normalized=True, grayscale=True):
    """
    Convert image to gray scale and normalize between -1 and 1 if required
    :param image:
    :param normalized:
    :param grayscale
    :return: image
    """
    if isinstance(image, list):
        image = image[0]
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image[:, :, np.newaxis]

    if normalized:
        return (image.astype(np.float32) - 128) / 128
    else:
        return image.astype(np.uint8)


def join_dicts(d, u):
    """
    Recursively updates a dictionary
    """
    result = d.copy()

    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            result[k] = join_dicts(d.get(k, {}), v)
        else:
            result[k] = v
    return result


def find_latest_checkpoint(directory):
    """
    Finds the latest checkpoint, based on how RLLib creates and names them.
    """
    start = directory
    max_checkpoint_int = -1
    checkpoint_path = ""

    # 1st layer: Check for the different run folders
    for f in os.listdir(start):
        if os.path.isdir(start + "/" + f):
            temp = start + "/" + f

            # 2nd layer: Check all the checkpoint folders
            for c in os.listdir(temp):
                if "checkpoint_" in c:

                    # 3rd layer: Get the most recent checkpoint
                    checkpoint_int = int(''.join([n for n in c
                                                  if n.isdigit()]))
                    if checkpoint_int > max_checkpoint_int:
                        max_checkpoint_int = checkpoint_int
                        checkpoint_path = temp + "/" + c + "/" + c.replace(
                            "_", "-")

    if not checkpoint_path:
        raise FileNotFoundError(
            "Could not find any checkpoint, make sure that you have selected the correct folder path"
        )

    return checkpoint_path


def get_checkpoint(name, directory, restore=False, overwrite=False):
    training_directory = os.path.join(directory, name)

    if overwrite and restore:
        raise RuntimeError(
            "Both 'overwrite' and 'restore' cannot be True at the same time")

    if overwrite:
        if os.path.isdir(training_directory):
            shutil.rmtree(training_directory)
            print("Removing all contents inside '" + training_directory + "'")
        return None


    if restore:
        return find_latest_checkpoint(training_directory)

    if os.path.isdir(training_directory) and len(os.listdir(training_directory)) != 0:
        raise RuntimeError(
            "The directory where you are trying to train (" +
            training_directory + ") is not empty. "
            "To start a new training instance, make sure this folder is either empty, non-existing "
            "or use the '--overwrite' argument to remove all the contents inside"
        )

    return None


def launch_tensorboard(logdir, host="localhost", port="6006"):
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", logdir, "--host", host, "--port", port])
    url = tb.launch()


def get_position(gps, route_planner):
    # Gets global latitude and longitude coordinates
    converted_gps = (gps - route_planner.mean) * route_planner.scale
    return converted_gps


def get_speed(hero):
    # Computes the speed of the hero vehicle in Km/h
    vel = hero.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def get_angle_to(pos, theta, target):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
        ])

    aim = R.T.dot(target - pos)
    angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
    angle = 0.0 if np.isnan(angle) else angle 

    return angle


def get_control(turn_controller, speed_controller, target, gps, theta, speed):
    # steering
    angle_unnorm = get_angle_to(pos=gps, theta=theta, target=target)
    angle = angle_unnorm / 90

    steer = turn_controller.step(angle)
    steer = np.clip(steer, -1.0, 1.0)
    steer = round(steer, 3)

    # acceleration
    angle_far_unnorm = get_angle_to(pos=gps, theta=theta, target=target)
    should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
    target_speed = 4.0 if should_slow else 7.0
    
    delta = np.clip(target_speed - speed, 0.0, 0.25)
    throttle = speed_controller.step(delta)
    throttle = np.clip(throttle, 0.0, 0.75)

    return steer, throttle, angle


def shift_point(ego_compass, ego_gps, near_node, offset_amount):
    # rotation matrix
    R = np.array([
        [np.cos(np.pi / 2 + ego_compass), -np.sin(np.pi / 2 + ego_compass)],
        [np.sin(np.pi / 2 + ego_compass), np.cos(np.pi / 2 + ego_compass)]
    ])

    # transpose of rotation matrix
    trans_R = R.T

    local_command_point = np.array([near_node[0] - ego_gps[0], near_node[1] - ego_gps[1]])
    local_command_point = trans_R.dot(local_command_point)

    # positive offset shifts near node to right; negative offset shifts near node to left
    local_command_point[0] += offset_amount
    local_command_point[1] += 0

    new_near_node = np.linalg.inv(trans_R).dot(local_command_point)

    new_near_node[0] += ego_gps[0]
    new_near_node[1] += ego_gps[1]

    return new_near_node


def calculate_high_level_action(turn_controller, speed_controller, high_level_action, gps, theta, speed, near_node):
    """
    0 -> brake
    1 -> no brake - go to left lane of next_waypoint
    2 -> no brake - keep lane (stay at next_waypoint's lane)
    3 -> no brake - go to right lane of next_waypoint
    """

    if high_level_action == 1: # left
        offset = -3.5
        new_near_node = shift_point(ego_compass=theta, ego_gps=gps, near_node=near_node, offset_amount=offset)
    
    elif high_level_action == 3: # right
        offset = 3.5
        new_near_node = shift_point(ego_compass=theta, ego_gps=gps, near_node=near_node, offset_amount=offset)
    
    else: # keep lane
        offset = 0.0
        new_near_node = near_node

    # get auto-pilot actions
    steer, throttle, angle = get_control(turn_controller, speed_controller, new_near_node, gps, theta, speed)

    if high_level_action == 0: # brake
        throttle = 0.0
        brake = 1.0
    else: # no brake
        throttle = throttle
        brake = 0.0

    return throttle, steer, brake


class RoutePlanner(object):
    def __init__(self, min_distance, max_distance, debug_size=256):
        self.route = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance

        # self.mean = np.array([49.0, 8.0]) # for carla 9.9
        # self.scale = np.array([111324.60662786, 73032.1570362]) # for carla 9.9
        self.mean = np.array([0.0, 0.0]) # for carla 9.10
        self.scale = np.array([111324.60662786, 111319.490945]) # for carla 9.10

    def set_route(self, global_plan, gps=False):
        self.route.clear()

        for pos, cmd in global_plan:
            if gps:
                pos = np.array([pos['lat'], pos['lon']])
                pos -= self.mean
                pos *= self.scale
            else:
                pos = np.array([pos.location.x, pos.location.y])
                pos -= self.mean

            self.route.append((pos, cmd))

    def run_step(self, gps):
        if len(self.route) == 1:
            return self.route[0]

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i-1][0])
            distance = np.linalg.norm(self.route[i][0] - gps)

            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        return self.route[1]


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative