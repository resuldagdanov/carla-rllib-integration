import carla

from srunner.scenariomanager.carla_data_provider import *

from leaderboard.scenarios.route_scenario import RouteScenario, BasicScenario, convert_transform_to_location
from leaderboard.utils.route_parser import RouteParser
from leaderboard.utils.route_manipulation import interpolate_trajectory

class CustomRouteScenario(BasicScenario): #CustomBasicScenario

    category = "RouteScenario"

    def __init__(self, world, config, ego_vehicle_type, debug_mode=0, criteria_enable=True):
        self.config = config
        self.route = None
        self.sampled_scenarios_definitions = None

        self._update_route(world, config, debug_mode>0)

        self.ego_vehicle = self._update_ego_vehicle(ego_vehicle_type)

        self.list_scenarios = self._build_scenario_instances(world,
                                                             self.ego_vehicle,
                                                             self.sampled_scenarios_definitions,
                                                             scenarios_per_tick=10,
                                                             timeout=self.timeout,
                                                             debug_mode=debug_mode>1)

        print(f"self.list_scenarios {self.list_scenarios}")
        
        super(CustomRouteScenario, self).__init__(name=config.name,
                                            ego_vehicles=[self.ego_vehicle],
                                            config=config,
                                            world=world,
                                            debug_mode=debug_mode>1,
                                            terminate_on_failure=False,
                                            criteria_enable=criteria_enable)

    def _update_ego_vehicle(self, ego_vehicle_type):
        """
        Set/Update the start position of the ego_vehicle
        """
        # move ego to correct position
        elevate_transform = self.route[0][0]
        elevate_transform.location.z += 0.5

        ego_vehicle = CarlaDataProvider.request_new_actor(ego_vehicle_type,
                                                        elevate_transform,
                                                        rolename='hero')

        spectator = CarlaDataProvider.get_world().get_spectator()
        ego_trans = ego_vehicle.get_transform()
        ego_yaw = ego_trans.rotation.yaw
        spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(x=-15 * math.cos(ego_yaw * 3.1415 / 180), y=-15 * math.sin(ego_yaw * 3.1415 / 180), z=15),
                                carla.Rotation(yaw=ego_yaw, pitch=-30)))

        return ego_vehicle

    def _update_route(self, world, config, debug_mode):
        """
        Update the input route, i.e. refine waypoint list, and extract possible scenario locations

        Parameters:
        - world: CARLA world
        - config: Scenario configuration (RouteConfiguration)
        """

        # Transform the scenario file into a dictionary
        world_annotations = RouteParser.parse_annotations_file(config.scenario_file)

        # prepare route's trajectory (interpolate and add the GPS route)
        gps_route, route = interpolate_trajectory(world, config.trajectory)

        potential_scenarios_definitions, _ = RouteParser.scan_route_for_scenarios(
            config.town, route, world_annotations)

        self.route = route
        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(self.route))

        # Sample the scenarios to be used for this route instance.
        self.sampled_scenarios_definitions = self._scenario_sampling(potential_scenarios_definitions)

        # Timeout of scenario in seconds
        self.timeout = self._estimate_route_timeout()

        # Print route in debug mode
        if debug_mode:
            self._draw_waypoints(world, self.route, vertical_shift=1.0, persistency=50000.0)