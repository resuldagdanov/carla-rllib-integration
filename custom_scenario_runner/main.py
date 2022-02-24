import carla

from custom_scenario_manager import CustomScenarioManager
from custom_route_scenario import CustomRouteScenario
from leaderboard.utils.route_indexer import RouteIndexer

from srunner.scenariomanager.carla_data_provider import *

import argparse
from argparse import RawTextHelpFormatter

class CustomScenarioRunner():
    ego_vehicles = []

    # Tunable parameters
    client_timeout = 10.0  # in seconds
    wait_for_world = 20.0  # in seconds
    frame_rate = 20.0      # in Hz

    def __init__(self, args):
        self.args = args

    def _load_and_wait_for_world(self, args, town, ego_vehicles=None):
        """
        Load a new CARLA world and provide data to CarlaDataProvider
        """

        self.client = carla.Client(args.host, int(args.port))
        if args.timeout:
            self.client_timeout = float(args.timeout)
        self.client.set_timeout(self.client_timeout)

        self.traffic_manager = self.client.get_trafficmanager(int(args.trafficManagerPort))

        self.manager = CustomScenarioManager(args.timeout, args.debug > 1)

        self.world = self.client.load_world(town)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        self.world.reset_all_traffic_lights()

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(int(args.trafficManagerPort))

        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(int(args.trafficManagerSeed))

        # Wait for the world to be ready
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

        if CarlaDataProvider.get_map().name != town:
            raise Exception("The CARLA server uses the wrong map!"
                            "This scenario requires to use map {}".format(town))

    def _prepare_ego_vehicles(self, ego_vehicles, wait_for_ego_vehicles=False):
        """
        Spawn or update the ego vehicles
        """

        if not wait_for_ego_vehicles:
            for vehicle in ego_vehicles:
                self.ego_vehicles.append(CarlaDataProvider.request_new_actor(vehicle.model,
                                                                             vehicle.transform,
                                                                             vehicle.rolename,
                                                                             color=vehicle.color,
                                                                             vehicle_category=vehicle.category))

        else:
            ego_vehicle_missing = True
            while ego_vehicle_missing:
                self.ego_vehicles = []
                ego_vehicle_missing = False
                for ego_vehicle in ego_vehicles:
                    ego_vehicle_found = False
                    carla_vehicles = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')
                    for carla_vehicle in carla_vehicles:
                        if carla_vehicle.attributes['role_name'] == ego_vehicle.rolename:
                            ego_vehicle_found = True
                            self.ego_vehicles.append(carla_vehicle)
                            break
                    if not ego_vehicle_found:
                        ego_vehicle_missing = True
                        break

            for i, _ in enumerate(self.ego_vehicles):
                self.ego_vehicles[i].set_transform(ego_vehicles[i].transform)

        # sync state
        CarlaDataProvider.get_world().tick()

    def _load_and_run_scenario(self, args, config):
        self._load_and_wait_for_world(args, config.town, config.ego_vehicles)
        self._prepare_ego_vehicles(config.ego_vehicles, False)

        scenario = CustomRouteScenario(world=self.world, config=config, ego_vehicle_type=args.ego_vehicle_type, debug_mode=args.debug)
        print(f"scenario {scenario}")

        self.manager.load_scenario(scenario, config.repetition_index)
        self.manager.run_scenario()
        self.manager.stop_scenario()

    def run(self, args):
        route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)
        
        while route_indexer.peek():
            # setup
            config = route_indexer.next()

            print(f"config {config}")
            # run
            self._load_and_run_scenario(args, config)
            #self._cleanup()
    
    
def main():
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost', help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='2000', help='TCP port to listen to (default: 2000)')
    parser.add_argument('--trafficManagerPort', default='8000', help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--trafficManagerSeed', default='0', help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--debug', type=int, help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='', help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', default="60.0", help='Set the CARLA client timeout value in seconds')

    # simulation setup
    #parser.add_argument('--routes', help='Name of the route to be executed. Point to the route_xml_file to be executed.', required=True)
    #parser.add_argument('--scenarios', help='Name of the scenario annotation file to be mixed with the route.', required=True)
    parser.add_argument('--repetitions', type=int, default=1, help='Number of repetitions per route.')

    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str, default='./simulation_results.json', help="Path to checkpoint used for saving statistics and resuming")
    parser.add_argument('--evaluate', action="store_true", help='RL Model Evaluate(True) Train(False)')
    parser.add_argument('--imitation_learning', action="store_true", help='Imitation Learning Model (True), RL Model (False)')

    parser.add_argument("--ego_vehicle_type", type=str, default="vehicle.lincoln.mkz2017", help="Ego vehicle type in Carla")

    arguments = parser.parse_args()
    arguments.routes = "/home/feyza/depo/research/carla_rl/rllib-integration/custom_scenario_runner/routes/failed_routes/town05_long/traffic_light_1.xml" #"/home/feyza/depo/research/carla_rl/rllib-integration/custom_scenario_runner/routes/original_routes/routes_town05_long.xml"
    arguments.scenarios = "/home/feyza/depo/research/carla_rl/rllib-integration/custom_scenario_runner/scenarios/all_towns_traffic_scenarios_WOR.json"
    arguments.debug = 1
    arguments.ego_vehicle_type = "vehicle.lincoln.mkz2017"

    custom_scenario_runner = CustomScenarioRunner(arguments)
    custom_scenario_runner.run(arguments)

if __name__ == '__main__':
    main()