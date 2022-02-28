#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import math
import numpy as np
from gym.spaces import Box, Discrete

import carla

from rllib_integration.base_experiment import BaseExperiment
from rllib_integration.helper import post_process_image, get_position, get_speed, calculate_high_level_action, PIDController


class DQNExperiment(BaseExperiment):
    def __init__(self, config={}):
        super().__init__(config)  # Creates a self.config with the experiment configuration

        self.frame_stack = self.config["others"]["framestack"]
        self.max_time_idle = self.config["others"]["max_time_idle"]
        self.max_time_episode = self.config["others"]["max_time_episode"]
        
        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]
        self.last_heading_deviation = 0
        self.last_action = None

        self.turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self.speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

        self.route_planner = None
        self.command_planner = None

        self.ego_gps = None
        self.compass = None
        self.speed_ms = None
        self.near_node = None

    def reset(self):
        """
        Called at the beginning and each time the simulation is reset
        """
        # Ending variables
        self.time_idle = 0
        self.time_episode = 0
        self.done_time_idle = False
        self.done_falling = False
        self.done_time_episode = False

        # hero variables
        self.last_location = None
        self.last_velocity = 0

        # Sensor stack
        self.prev_image_0 = None
        self.prev_image_1 = None
        self.prev_image_2 = None

        self.last_heading_deviation = 0

    def get_action_space(self):
        """
        Returns the action space, in this case, a discrete space
        """
        return Discrete(4)

    def get_observation_space(self):
        num_of_channels = 3
        image_space = Box(
            low=0.0,
            high=255.0,
            shape=(
                self.config["hero"]["sensors"]["birdview"]["size"],
                self.config["hero"]["sensors"]["birdview"]["size"],
                num_of_channels * self.frame_stack,
            ),
            dtype=np.uint8,
        )
        return image_space

    def compute_action(self, action):
        """
        Given the action, returns a carla.VehicleControl() which will be applied to the hero
        """
        throttle, steer, brake = calculate_high_level_action(turn_controller=self.turn_controller,
                                                             speed_controller=self.speed_controller,
                                                             high_level_action=action,
                                                             gps=self.ego_gps,
                                                             theta=self.compass,
                                                             speed=self.speed_ms,
                                                             near_node=self.near_node)

        action = carla.VehicleControl()
        action.throttle = float(throttle)
        action.steer = float(steer)
        action.brake = float(brake)
        action.reverse = False
        action.hand_brake = False

        self.last_action = action
        return action

    def get_observation(self, sensor_data, hero):
        """
        Function to do all the post processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """
        self.ego_gps = get_position(gps=sensor_data['gps'][1][:2], route_planner=self.route_planner)
        self.compass = sensor_data['imu'][1][-1]

        speed_kmph = get_speed(hero)
        self.speed_ms = 0.277778 * speed_kmph

        self.near_node, near_command = self.route_planner.run_step(gps=self.ego_gps)
        far_node, far_command = self.command_planner.run_step(gps=self.ego_gps)

        # image = post_process_image(sensor_data['birdview'][1], normalized = False, grayscale = False)
        image = post_process_image(sensor_data['front_camera'][1], normalized=False, grayscale=False) # TODO: give normalized input to the network

        if self.prev_image_0 is None:
            self.prev_image_0 = image
            self.prev_image_1 = self.prev_image_0
            self.prev_image_2 = self.prev_image_1

        images = image

        if self.frame_stack >= 2:
            images = np.concatenate([self.prev_image_0, images], axis=2)
        if self.frame_stack >= 3 and images is not None:
            images = np.concatenate([self.prev_image_1, images], axis=2)
        if self.frame_stack >= 4 and images is not None:
            images = np.concatenate([self.prev_image_2, images], axis=2)

        self.prev_image_2 = self.prev_image_1
        self.prev_image_1 = self.prev_image_0
        self.prev_image_0 = image

        return images, {}

    def get_done_status(self, observation, core):
        """
        Returns whether or not the experiment has to end
        """
        hero = core.hero
        speed_kmph = get_speed(hero)

        self.done_time_idle = self.max_time_idle < self.time_idle

        if speed_kmph > 1.0:
            self.time_idle = 0
        else:
            self.time_idle += 1

        self.time_episode += 1
        self.done_time_episode = self.max_time_episode < self.time_episode
        self.done_falling = hero.get_location().z < -0.5

        return self.done_time_idle or self.done_falling or self.done_time_episode

    def compute_reward(self, observation, core):
        """
        Computes the reward
        """
        def unit_vector(vector):
            return vector / np.linalg.norm(vector)
        def compute_angle(u, v):
            return -math.atan2(u[0]*v[1] - u[1]*v[0], u[0]*v[0] + u[1]*v[1])
        def find_current_waypoint(map_, hero):
            return map_.get_waypoint(hero.get_location(), project_to_road=False, lane_type=carla.LaneType.Any)
        def inside_lane(waypoint, allowed_types):
            if waypoint is not None:
                return waypoint.lane_type in allowed_types
            return False

        world = core.world
        hero = core.hero
        map_ = core.map

        # Hero-related variables
        hero_location = hero.get_location()
        hero_velocity = get_speed(hero)
        hero_heading = hero.get_transform().get_forward_vector()
        hero_heading = [hero_heading.x, hero_heading.y]

        # Initialize last location
        if self.last_location == None:
            self.last_location = hero_location

        # Compute deltas
        delta_distance = float(np.sqrt(np.square(hero_location.x - self.last_location.x) + \
                            np.square(hero_location.y - self.last_location.y)))
        delta_velocity = hero_velocity - self.last_velocity

        # Update variables
        self.last_location = hero_location
        self.last_velocity = hero_velocity

        # Reward if going forward
        reward = delta_distance

        # Reward if going faster than last step
        if hero_velocity < 20.0:
            reward += 0.05 * delta_velocity

        # La duracion de estas infracciones deberia ser 2 segundos?
        # Penalize if not inside the lane
        closest_waypoint = map_.get_waypoint(
            hero_location,
            project_to_road=False,
            lane_type=carla.LaneType.Any
        )
        if closest_waypoint is None or closest_waypoint.lane_type not in self.allowed_types:
            reward += -0.5
            self.last_heading_deviation = math.pi
        else:
            if not closest_waypoint.is_junction:
                wp_heading = closest_waypoint.transform.get_forward_vector()
                wp_heading = [wp_heading.x, wp_heading.y]
                angle = compute_angle(hero_heading, wp_heading)
                self.last_heading_deviation = abs(angle)

                if np.dot(hero_heading, wp_heading) < 0:
                    # We are going in the wrong direction
                    reward += -0.5

                else:
                    if abs(math.sin(angle)) > 0.4:
                        if self.last_action == None:
                            self.last_action = carla.VehicleControl()

                        if self.last_action.steer * math.sin(angle) >= 0:
                            reward -= 0.05
            else:
                self.last_heading_deviation = 0

        if self.done_falling:
            reward += -40
        if self.done_time_idle:
            print("Done idle")
            reward += -100
        if self.done_time_episode:
            print("Done max time")
            reward += 100

        return reward
