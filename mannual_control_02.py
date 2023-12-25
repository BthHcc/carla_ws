import carla
import random
import pygame
import os
import numpy as np
import queue
import sys
import time
import open3d as o3d
from matplotlib import cm
from datetime import datetime

# 获取了名为 "plasma" 的颜色映射对象，并将颜色映射对象中的颜色转换为 NumPy 数组
VIDIDIS = np.array(cm.get_cmap("plasma").colors)

# 生成一个等差数列，用于表示颜色映射的范围
VID_RANGE = np.linspace(0.0, 1.0, VIDIDIS.shape[0])


# Render object to keep and pass the PyGame surface
class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0, 255, (height, width, 3), dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0, 1))


# Camera sensor callback, reshapes raw data from camera into 2D RGB and applies to PyGame surface
def pygame_callback(data, obj):
    img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
    img = img[:, :, :3]
    img = img[:, :, ::-1]
    obj.surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))


def generate_lidar_bp(blueprint_library, delta):
    """
    获取激光雷达蓝图
    :param blueprint_library: 世界蓝图库
    :param delta: 更新速率(s)
    :return: 激光雷达蓝图
    """

    # 一些参数设置
    lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("dropoff_general_rate", "0.0")
    lidar_bp.set_attribute("dropoff_intensity_limit", "1.0")
    lidar_bp.set_attribute("dropoff_zero_intensity", "0.0")

    lidar_bp.set_attribute("upper_fov", str(15.0))
    lidar_bp.set_attribute("lower_fov", str(-25.0))
    lidar_bp.set_attribute("channels", str(64.0))
    lidar_bp.set_attribute("range", str(100.0))
    lidar_bp.set_attribute("rotation_frequency", str(1.0 / delta))
    lidar_bp.set_attribute("points_per_second", str(500000))

    return lidar_bp


def lidar_callback(point_cloud, point_list):
    # 将点云（carla格式）转换为numpy.ndarray
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype("f4")))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    intensity = data[:, -1]
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIDIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIDIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIDIDIS[:, 2])]

    points = data[:, :-1]  # 只使用 x, y, z 坐标
    points[:, 1] = -points[:, 1]  # 这与官方脚本不同
    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)


# Control object to manage vehicle controls
class ControlObject(object):
    def __init__(self, veh):

        # Conrol parameters to store the control state
        self._vehicle = veh
        self._steer = 0
        self._throttle = False
        self._brake = False
        self._steer = None
        self._steer_cache = 0
        # A carla.VehicleControl object is needed to alter the
        # vehicle's control state
        self._control = carla.VehicleControl()

    # 检查Pygame窗口中的按键事件并定义控制状态
    def parse_control(self, event):
        # 检查是否有按键被按下
        if event.type == pygame.KEYDOWN:
            # 根据按下的键执行相应的操作
            if event.key == pygame.K_RETURN:
                # 当按下RETURN键时，将自动驾驶模式设置为False
                self._vehicle.set_autopilot(False)
            if event.key == pygame.K_UP:
                # 当按下UP键时，将油门状态设置为True
                self._throttle = True
            if event.key == pygame.K_DOWN:
                # 当按下DOWN键时，将刹车状态设置为True
                self._brake = True
            if event.key == pygame.K_RIGHT:
                # 当按下RIGHT键时，将方向状态设置为1
                self._steer = 1
            if event.key == pygame.K_LEFT:
                # 当按下LEFT键时，将方向状态设置为-1
                self._steer = -1
        # 检查是否有按键被释放
        if event.type == pygame.KEYUP:
            # 根据释放的键执行相应的操作
            if event.key == pygame.K_UP:
                # 当释放UP键时，将油门状态设置为False
                self._throttle = False
            if event.key == pygame.K_DOWN:
                # 当释放DOWN键时，将刹车状态设置为False，将后退状态设置为False
                self._brake = False
                self._control.reverse = False
            if event.key == pygame.K_RIGHT:
                # 当释放RIGHT键时，将方向状态设置为None
                self._steer = None
            if event.key == pygame.K_LEFT:
                # 当释放LEFT键时，将方向状态设置为None
                self._steer = None

    # 处理当前的控制状态，根据按键保持按下的情况修改控制参数
    def process_control(self):

        # 如果正在踩油门
        if self._throttle:
            # 逐渐增加油门，最大为1
            self._control.throttle = min(self._control.throttle + 0.01, 1)
            # 设置车辆档位为1
            self._control.gear = 1
            # 关闭刹车
            self._control.brake = False
        # 如果没有踩油门且没有刹车
        elif not self._brake:
            # 将油门设为0
            self._control.throttle = 0.0

        # 如果正在刹车
        if self._brake:
            # 如果车辆静止且没有倒车，刹车时切换到倒车档
            if self._vehicle.get_velocity().length() < 0.01 and not self._control.reverse:
                self._control.brake = 0.0
                self._control.gear = 1
                self._control.reverse = True
                self._control.throttle = min(self._control.throttle + 0.1, 1)
            # 如果已经在倒车，则继续增加油门
            elif self._control.reverse:
                self._control.throttle = min(self._control.throttle + 0.1, 1)
            # 否则，关闭油门，逐渐增加刹车
            else:
                self._control.throttle = 0.0
                self._control.brake = min(self._control.brake + 0.3, 1)
        # 如果没有刹车
        else:
            self._control.brake = 0.0

        # 如果正在转向
        if self._steer is not None:
            # 根据方向键的按下情况逐渐调整转向角度
            if self._steer == 1:
                self._steer_cache += 0.03
            if self._steer == -1:
                self._steer_cache -= 0.03
            # 限制转向角度在 -0.7 到 0.7 之间
            min(0.7, max(-0.7, self._steer_cache))
            # 将转向角度应用到控制参数，并四舍五入到一位小数
            self._control.steer = round(self._steer_cache, 1)
        # 如果没有转向
        else:
            # 根据当前转向角度逐渐减小转向
            if self._steer_cache > 0.0:
                self._steer_cache *= 0.2
            if self._steer_cache < 0.0:
                self._steer_cache *= 0.2
            # 如果转向角度很小，则将其设为0
            if 0.01 > self._steer_cache > -0.01:
                self._steer_cache = 0.0
            # 将转向角度应用到控制参数，并四舍五入到一位小数
            self._control.steer = round(self._steer_cache, 1)

        # 将控制参数应用到自动驾驶汽车上
        self._vehicle.apply_control(self._control)


def main():
    sensor_list = []
    try:

        # Connect to the client and retrieve the world object
        client = carla.Client('localhost', 2000)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        # 设置行人数量
        num_walkers = 100
        # 设置跑步的行人比例
        percentage_pedestrians_running = 0.15
        # 设置横穿马路的行人比例
        percentage_pedestrians_crossing = 1.0

        # Set up the TM in synchronous mode
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        # Set a seed so behaviour can be repeated if necessary
        traffic_manager.set_random_device_seed(0)
        random.seed(0)

        # We will aslo set up the spectator so we can see what we do
        spectator = world.get_spectator()

        # Retrieve the map's spawn points
        spawn_points = world.get_map().get_spawn_points()

        # Select some models from the blueprint library
        models = ['dodge', 'audi', 'model3', 'mini', 'mustang', 'lincoln', 'prius', 'nissan', 'crown', 'impala']
        blueprints = []
        for vehicle in world.get_blueprint_library().filter('*vehicle*'):
            if any(model in vehicle.id for model in models):
                blueprints.append(vehicle)

        # Set a max number of vehicles and prepare a list for those we spawn
        max_vehicles = 10
        max_vehicles = min([max_vehicles, len(spawn_points)]) + 1
        vehicles = []

        # Take a random sample of the spawn points and spawn some vehicles
        for i, spawn_point in enumerate(random.sample(spawn_points, max_vehicles)):
            # 生成主车
            if i == max_vehicles - 1:
                # 从蓝图库中挑选我们需要的主车蓝图
                ego_bp = world.get_blueprint_library().find('vehicle.volkswagen.t2_2021')
                ego_vehicle = world.try_spawn_actor(ego_bp, spawn_point)
                vehicles.append(ego_vehicle)
                break
            temp = world.try_spawn_actor(random.choice(blueprints), spawn_point)
            if temp is not None:
                vehicles.append(temp)

        # Parse the list of spawned vehicles and give control to the TM through set_autopilot()
        for vehicle in vehicles:
            vehicle.set_autopilot(True)
            # Randomly set the probability that a vehicle will ignore traffic lights
            traffic_manager.ignore_lights_percentage(vehicle, random.randint(0, 50))

        # 选择最后一辆车作为主车
        ego_vehicle = vehicles[-1]

        # 获得整个的blueprint库并从中筛选出行人
        ped_blueprints = world.get_blueprint_library().filter('*pedestrian*')

        # 通过world获得所有可以生成行人的地点并存储
        ped_spawn_points = []
        for i in range(num_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                ped_spawn_points.append(spawn_point)

        # 创建用来存储行人，行人速度设置和行人控制器的list
        walker_batch = []
        walker_speed = []
        walker_ai_batch = []

        # 在地图上随机生成num_walkers个行人，每个行人为行人蓝图库中的随机行人，并设定行人的移动速度
        for j in range(num_walkers):
            walker_bp = random.choice(ped_blueprints)

            # 取消行人无敌状态
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')

            # 设置行人的移动速度
            if walker_bp.has_attribute('speed'):
                if random.random() > percentage_pedestrians_running:
                    # 将对应行人速度设置为走路速度
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # 将对应行人速度设置为跑步速度
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            # 从可生成行人的生成点中随机选择并生成随机行人，之后将生成的行人添加到同一批中
            walker_batch.append(world.try_spawn_actor(walker_bp, random.choice(ped_spawn_points)))

        # 从蓝图库中寻找控制行人行动逻辑的控制器
        walker_ai_blueprint = world.get_blueprint_library().find('controller.ai.walker')

        # 为整批行人各自生成对应的控制器，并把控制器添加到代表批量控制器的列表中
        for walker in world.get_actors().filter('*pedestrian*'):
            walker_ai_batch.append(world.spawn_actor(walker_ai_blueprint, carla.Transform(), walker))

        # 批量启动行人控制器，并设置控制器参数
        for i in range(len(walker_ai_batch)):
            # 启动控制器
            walker_ai_batch[i].start()
            # 通过控制器设置行人的目标点
            walker_ai_batch[i].go_to_location(world.get_random_location_from_navigation())
            # 通过控制器设置行人的行走速度
            walker_ai_batch[i].set_max_speed(float(walker_speed[i]))

        # 设置行人横穿马路的参数
        world.set_pedestrians_cross_factor(percentage_pedestrians_crossing)

        # Set up the simulator in synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True  # Enables synchronous mode
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # 使用camera
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        # Initialise the camera floating behind the vehicle
        camera_init_trans = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)

        # 设置lidar的初始位置
        lidar_bp = generate_lidar_bp(blueprint_library, delta = 0.05)
        lidar_init_trans = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
        lidar = world.spawn_actor(lidar_bp, lidar_init_trans, attach_to=ego_vehicle)

        # 设置 point_list 来存储点云
        point_list = o3d.geometry.PointCloud()

        # 监听激光雷达以收集点云
        lidar.listen(lambda data: lidar_callback(data, point_list))

        # 为使用 open3d 进行显示设置一些基本设置
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="Display Point Cloud",
            width=960,
            height=540,
            left=480,
            top=270)

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1
        vis.get_render_option().show_coordinate_frame = True

        frame = 0
        dt0 = datetime.now()

        # Start camera with PyGame callback
        camera.listen(lambda image: pygame_callback(image, renderObject))

        # Get camera dimensions
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()

        # 渲染实例化对象
        renderObject = RenderObject(image_w, image_h)
        # 车辆控制实例化对象
        controlObject = ControlObject(ego_vehicle)

        # Initialise the display
        pygame.init()
        gameDisplay = pygame.display.set_mode((image_w, image_h), pygame.HWSURFACE | pygame.DOUBLEBUF)
        # Draw black to the display
        gameDisplay.fill((0, 0, 0))
        gameDisplay.blit(renderObject.surface, (0, 0))
        pygame.display.flip()

        # Game loop
        crashed = False

        while not crashed:
            if frame == 2:
                vis.add_geometry(point_list)

            vis.update_geometry(point_list)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.005)

            # Advance the simulation time
            world.tick()

            # 在这里添加一个旁观者来观察我们的自动驾驶车辆将如何移动
            spectator = world.get_spectator()
            transform = vehicle.get_transform()  # 获取车辆的变换
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50),
                                                    carla.Rotation(pitch=-90)))

            process_time = datetime.now() - dt0
            sys.stdout.write("\r" + "FPS: " + str(1.0 / process_time.total_seconds()) +
                             " Current Frame: " + str(frame))
            sys.stdout.flush()
            dt0 = datetime.now()

            frame += 1


            # Update the display
            gameDisplay.blit(renderObject.surface, (0, 0))
            pygame.display.flip()
            # Process the current control state
            controlObject.process_control()
            # Collect key press events
            for event in pygame.event.get():
                # If the window is closed, break the while loop
                if event.type == pygame.QUIT:
                    crashed = True

                # Parse effect of key press event on control state
                controlObject.parse_control(event)
                if event.type == pygame.KEYUP:
                    # TAB key switches vehicle
                    if event.key == pygame.K_TAB:
                        ego_vehicle.set_autopilot(True)
                        ego_vehicle = random.choice(vehicles)
                        # Ensure vehicle is still alive (might have been destroyed)
                        if ego_vehicle.is_alive:
                            # Stop and remove the camera
                            camera.stop()
                            camera.destroy()

                            # Spawn new camera and attach to new vehicle
                            controlObject = ControlObject(ego_vehicle)
                            camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)
                            camera.listen(lambda image: pygame_callback(image, renderObject))

                            # Update PyGame window
                            gameDisplay.fill((0, 0, 0))
                            gameDisplay.blit(renderObject.surface, (0, 0))
                            pygame.display.flip()
    finally:
        # Stop camera and quit PyGame after exiting game loop
        camera.stop()
        pygame.quit()
        lidar.destroy()
        vis.destroy_window()

        # 停止并销毁所有controller
        for controller in world.get_actors().filter('*controller*'):
            controller.stop()
        # 销毁所有车辆
        for vehicle in world.get_actors().filter('*vehicle*'):
            vehicle.destroy()
        # 销毁所有行人
        for walker in world.get_actors().filter('*walker*'):
            walker.destroy()
        # 销毁所有传感器
        for sensor in world.get_actors().filter('*sensor*'):
            sensor.destroy()

        # 恢复模拟世界设置
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
