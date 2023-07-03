import glob
import os
import sys
import random

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


def main():
    actor_list = []

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('*vehicle*')
        spawn_points = world.get_map().get_spawn_points()

        # set synchorinized mode
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        for i in range(100):
            bp = random.choice(vehicle_bp)
            transform = random.choice(spawn_points)
            vehicle = world.try_spawn_actor(bp, transform)
            actor_list.append(vehicle)

        tm_port = traffic_manager.get_port()
        for vehicle in actor_list:
            vehicle.set_autopilot(True, tm_port)
            traffic_manager.auto_lane_change(vehicle, False)

        danger_car = actor_list[0]
        traffic_manager.ignore_lights_percentage(danger_car, 100)
        traffic_manager.ignore_signs_percentage(danger_car, 100)
        traffic_manager.ignore_vehicles_percentage(danger_car, 100)
        traffic_manager.distance_to_leading_vehicle(danger_car, 0)
        traffic_manager.vehicle_percentage_speed_difference(danger_car, -20)
        traffic_manager.auto_lane_change(danger_car, True)

        while (True):
            world.tick()
            spectator = world.get_spectator()
            transform = danger_car.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50),
                                                    carla.Rotation(pitch=-90)))

    finally:
        world.apply_settings(original_settings)
        print('destroying actors')
        actor_list = world.get_actors()
        vehicle_list = list(actor_list.filter('vehicle.*'))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
