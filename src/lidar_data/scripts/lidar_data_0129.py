import carla
import random
import numpy as np
import rospy
import math
import pygame
from sensor_msgs.msg import PointCloud2, PointField
from configparser import ConfigParser


class LidarPublisher:
    def __init__(self, node_name):
        self.lidar_publisher = rospy.Publisher('/carla/ego_vehicle/lidar', PointCloud2, queue_size=10)
        rospy.init_node(node_name, anonymous=True)

    def publish_lidar_data(self, lidar_data):
        msg = PointCloud2()
        msg.header.stamp = rospy.Time().now()
        msg.header.frame_id = "lidar1"
        msg.height = 1
        msg.width = len(lidar_data)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * len(lidar_data)
        msg.is_dense = False
        msg.data = np.asarray(lidar_data, np.float32).tobytes()

        self.lidar_publisher.publish(msg)

    def destroy(self):
        rospy.signal_shutdown('Shutting down carla_publisher_node')


class CarlaSimulation:
    def __init__(self):
        self.actor_list = []
        self.sensor_list = []
        self.lidar_publisher = LidarPublisher('carla_publisher_node')
        self._control = carla.VehicleControl()
        self.world = None
        self.client = None
        self.ego_vehicle = None
        self.bp_lib = None
        self.traffic_manager = None
        self.renderObject = None
        self.display = None

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        self._parser = ConfigParser()
        self._parser.read('/home/bth/carla_data/src/lidar_data/scripts/wheel_config.ini')
        self._steer_idx = int(
            self._parser.get('G923 Racing Wheel', 'steering_wheel'))
        self._throttle_idx = int(
            self._parser.get('G923 Racing Wheel', 'throttle'))
        self._brake_idx = int(self._parser.get('G923 Racing Wheel', 'brake'))
        self._reverse_idx = int(self._parser.get('G923 Racing Wheel', 'reverse'))
        self._handbrake_idx = int(
            self._parser.get('G923 Racing Wheel', 'handbrake'))

    def setup_carla_world(self):
        # initialize ros node
        rospy.init_node('carla_publisher_node', anonymous=True)

        # initialize Carla client
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)

        # Get the Carla world and blueprint library
        self.world = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library()

    def spawn_pedestrians(self, num_walkers=50, ped_crossing_prob=0.9):
        ped_spawn_points = [carla.Transform(location=self.world.get_random_location_from_navigation()) for _ in
                            range(num_walkers)]

        # Get pedestrian blueprints
        ped_blueprints = self.bp_lib.filter('*pedestrian*')

        # Randomly generate pedestrians
        for spawn_point in random.sample(ped_spawn_points, num_walkers):
            self.world.try_spawn_actor(random.choice(ped_blueprints), spawn_point)

        walker_batch = []  # walker list
        walker_ai_batch = []  # walker controller list

        # Randomly generated pedestrian
        for j in range(num_walkers):
            walker_bp = random.choice(ped_blueprints)

            # Cancel pedestrian invincibility
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')

            # Randomly select and generate random pedestrians from the generation points that can generate pedestrians
            walker_batch.append(self.world.try_spawn_actor(walker_bp, random.choice(ped_spawn_points)))

        # walker controller blueprint
        walker_ai_blueprint = self.world.get_blueprint_library().find('controller.ai.walker')

        for walker in self.world.get_actors().filter('*pedestrian*'):
            walker_ai_batch.append(self.world.try_spawn_actor(walker_ai_blueprint, carla.Transform(), walker))

        # Start the pedestrian controller
        for i in range(len(walker_ai_batch)):
            walker_ai_batch[i].start()
            walker_ai_batch[i].go_to_location(self.world.get_random_location_from_navigation())
            walker_ai_batch[i].set_max_speed(1 + random.random())  # 1~2 m/s

        # Set pedestrian crossing
        self.world.set_pedestrians_cross_factor(ped_crossing_prob)

    def _parse_vehicle_wheel(self):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        # print (jsInputs)
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        K1 = 1.0  # 0.55
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

        K2 = 1.6  # 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        self._control.steer = steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd

        # toggle = jsButtons[self._reverse_idx]

        self._control.hand_brake = bool(jsButtons[self._handbrake_idx])

    def spawn_ego_vehicle(self):
        # Create the ego vehicle
        ego_vehicle_bp = self.bp_lib.find('vehicle.volkswagen.t2_2021')
        # Get a random valid occupation in the world
        transform = random.choice(self.world.get_map().get_spawn_points())
        # Spawn the vehicle
        self.ego_vehicle = self.world.spawn_actor(ego_vehicle_bp, transform)
        # Set the vehicle autopilot mode
        self.ego_vehicle.set_autopilot(False)

        # Add a camera on it
        camera_bp = self.bp_lib.find('sensor.camera.rgb')

        # Set the relative location
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

        # Spawn the camera
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)

        # Get camera dimensions
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        self.renderObject = RenderObject(image_w, image_h)
        # Start camera with PyGame callback
        camera.listen(lambda image: pygame_callback(image, self.renderObject))

        self.display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.display.fill((0, 0, 0))
        self.display.blit(self.renderObject.surface, (0, 0))
        pygame.display.flip()

        self.sensor_list.append(camera)

        # Add a lidar on it
        lidar_bp = self.bp_lib.find('sensor.lidar.ray_cast')

        lidar_attributes = {
            'channels': 160,
            'points_per_second': 384000,
            'rotation_frequency': 10,
            'range': 30.1,
            'horizontal_fov': 120,
            'upper_fov': 10,
            'lower_fov': -80
        }

        for attribute, value in lidar_attributes.items():
            lidar_bp.set_attribute(attribute, str(value))

        # Set the relative location
        lidar_location = carla.Location(0, 0, 2.5)
        lidar_rotation = carla.Rotation(0, 0, 0)
        lidar_transform = carla.Transform(lidar_location, lidar_rotation)

        # Spawn the lidar
        lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.ego_vehicle)

        lidar.listen(lambda data: lidar_callback(data, self.lidar_publisher))

        self.sensor_list.append(lidar)

    def set_synchronous_mode(self):
        # Set synchronous mode
        setting = self.world.get_settings()
        setting.synchronous_mode = True
        setting.fixed_delta_seconds = 0.05
        self.world.apply_settings(setting)

        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)

    def run_simulation_loop(self):
        # set the spectator to follow the ego vehicle
        spectator = self.world.get_spectator()
        transform = self.ego_vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20),
                                                carla.Rotation(pitch=-90)))

        for event in pygame.event.get():
            pass

        if isinstance(self._control, carla.VehicleControl):
            self._parse_vehicle_wheel()
            self._control.reverse = self._control.gear < 0
        self.ego_vehicle.apply_control(self._control)
        self.display.blit(self.renderObject.surface, (0, 0))
        pygame.display.flip()

        self.world.tick()

    def cleanup(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

        print('destroying actors.')
        for sensor in self.sensor_list:
            sensor.destroy()
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        for vehicle in self.world.get_actors().filter('*vehicle*'):
            vehicle.destroy()
        for walker in self.world.get_actors().filter('*pedestrian*'):
            walker.destroy()
        for controller in self.world.get_actors().filter('*controller*'):
            controller.stop()
        self.lidar_publisher.destroy()
        print('done!')


def main():
    # Initialize Pygame
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("CARLA Simulation")

    # initialize steering wheel
    pygame.joystick.init()

    joystick_count = pygame.joystick.get_count()
    if joystick_count > 1:
        raise ValueError("Please Connect Just One Joystick")

    try:
        carla_simulation = CarlaSimulation()
        carla_simulation.setup_carla_world()
        carla_simulation.spawn_pedestrians()
        carla_simulation.spawn_ego_vehicle()
        carla_simulation.set_synchronous_mode()
        while True:
            carla_simulation.run_simulation_loop()
            # for event in pygame.event.get():
            #     if event.type == pygame.JOYAXISMOTION:
            #         print(f"Axis {event.axis}: {event.value}")
            #     elif event.type == pygame.JOYBUTTONDOWN:
            #         print(f"Button {event.button} pressed.")
            #     elif event.type == pygame.JOYBUTTONUP:
            #         print(f"Button {event.button} released.")

    finally:
        carla_simulation.cleanup()
        pygame.quit()


# Convert lidar in carla to publish in pointcloud2 format in ros
def lidar_callback(point_cloud, lidar_publisher):
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))
    points = data[:, :-1]
    lidar_publisher.publish_lidar_data(points)


# Camera sensor callback, reshapes raw data from camera into 2D RGB and applies to PyGame surface
def pygame_callback(data, obj):
    img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
    img = img[:, :, :3]
    img = img[:, :, ::-1]
    obj.surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))


# Render object to keep and pass the PyGame surface
class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0, 255, (height, width, 3), dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0, 1))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
