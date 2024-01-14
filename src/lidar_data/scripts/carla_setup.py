import carla
import random

def setup_carla_environment():
    # Connect to the Carla server
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    # Get the Carla world and blueprint library
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Create the ego vehicle
    ego_vehicle_bp = blueprint_library.find('vehicle.volkswagen.t2_2021')
    # Get a random valid occupation in the world
    transform = random.choice(world.get_map().get_spawn_points())
    # Spawn the vehicle
    ego_vehicle = world.spawn_actor(ego_vehicle_bp, transform)
    # Set the vehicle autopilot mode
    ego_vehicle.set_autopilot(True)

    # Add a lidar on it
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', str(32))
    lidar_bp.set_attribute('points_per_second', str(90000))
    lidar_bp.set_attribute('rotation_frequency', str(40))
    lidar_bp.set_attribute('range', str(20))

    # Set the relative location
    lidar_location = carla.Location(0, 0, 2)
    lidar_rotation = carla.Rotation(0, 0, 0)
    lidar_transform = carla.Transform(lidar_location, lidar_rotation)

    # Spawn the lidar
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)

    return client, world, blueprint_library, ego_vehicle, lidar
