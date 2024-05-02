import rospy
import carla
import random
import std_msgs
import struct
import sys
import ctypes
from sensor_msgs.msg import PointCloud2, PointField, Imu
from sensor_msgs import point_cloud2
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import Path
from std_msgs.msg import Float32, Int32
import numpy as np
from nav_msgs.msg import Path
import geometry_msgs.msg


import tf2_ros
import threading


from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__))))
import transforms as trans

from transforms3d.euler import quat2euler, euler2quat

_DATATYPES = {}
_DATATYPES[PointField.INT8] = ('b', 1)
_DATATYPES[PointField.UINT8] = ('B', 1)
_DATATYPES[PointField.INT16] = ('h', 2)
_DATATYPES[PointField.UINT16] = ('H', 2)
_DATATYPES[PointField.INT32] = ('i', 4)
_DATATYPES[PointField.UINT32] = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)


def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = '>' if is_bigendian else '<'

    offset = 0
    for field in (f for f in sorted(fields, key=lambda f: f.offset)
                  if field_names is None or f.name in field_names):
        if offset < field.offset:
            fmt += 'x' * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print('Skipping unknown PointField datatype [{}]' % field.datatype, file=sys.stderr)
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt


def quaternion_conjugate(quaternion):
    """Calculate the conjugate of a quaternion."""
    return np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])

class SubscriberNode:
    def __init__(self):
        rospy.init_node('subscriber_node', anonymous=True)
        self.lidar_publisher = rospy.Publisher('/carla/ego_vehicle/filtered_pointcloud', PointCloud2, queue_size=10)
        self.predict_path_publisher = rospy.Publisher('/carla/ego_vehicle/predict_path', Path, queue_size=10)
        self.marker_publisher = rospy.Publisher('/carla/ego_vehicle/car_marker', Marker, queue_size=10)

        self.carla_world = None
        self.ego_car_id = None
        self.latest_poses = [] # 最新姿态
        self.pre_path_length = 100 # 预测姿态  -> 5s

        self.current_speed = None # 车辆速度
        self.current_acceleration = None

        rospy.Subscriber('/carla/ego_vehicle/lidar', PointCloud2, self.pointcloud_callback)
        rospy.Subscriber('/carla/ego_vehicle/brake', Float32)
        rospy.Subscriber('/carla/ego_vehicle/id', Int32, self.id_callback)
        self.path_subscriber = rospy.Subscriber('/carla/ego_vehicle/path', Path, self.path_callback)
        self.imu_subscriber = rospy.Subscriber('/carla/ego_vehicle/imu', Imu, self.imu_callback)
        self.bbx_subscriber =rospy.Subscriber('/motion_detector/visualization/clusters', MarkerArray, self.bbx_callback)

        self.car_marker = None

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # self.tf_buffer = tf2_ros.Buffer()
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)


    def tf_listener_thread_func(self):
        rate = rospy.Rate(20)  # 设置监听器的频率
        while not rospy.is_shutdown():
            try:
                # 等待从 "lidar" 到 "map" 的变换
                (trans, rot) = self.tf_listener.lookupTransform("map", "lidar", rospy.Time(0))
                # 打印变换信息
                # rospy.loginfo("Transform from lidar to map: translation={}, rotation={}".format(trans, rot))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logerr("Failed to lookup transform from lidar to map")
            rate.sleep()


    def publish_lidar_transform(self):
        if self.ego_car_id is not None:
            # Get ego vehicle transform
            ego_car = self.carla_world.get_actor(self.ego_car_id)
            if ego_car is not None:
                ego_transform = ego_car.get_transform()

                # Publish transform
                map_origin = geometry_msgs.msg.TransformStamped()
                map_origin.header.stamp = rospy.Time()
                map_origin.header.frame_id = "map"
                map_origin.child_frame_id = "lidar"
                map_origin.transform.translation.x = ego_transform.location.x
                map_origin.transform.translation.y = ego_transform.location.y
                map_origin.transform.translation.z = ego_transform.location.z + 1.5
                roll, pitch, yaw = trans.carla_rotation_to_RPY(ego_transform.rotation)
                quat = euler2quat(roll, pitch, yaw)
                map_origin.transform.rotation.w = quat[0]
                map_origin.transform.rotation.x = quat[1]
                map_origin.transform.rotation.y = quat[2]
                map_origin.transform.rotation.z = quat[3]
                self.tf_broadcaster.sendTransform(map_origin)


    # def publish_sensor_transform(self):
    #     rate = rospy.Rate(20)
    #     while not rospy.is_shutdown():
    #         if self.ego_car_id is not None:
    #             # Get ego vehicle transform
    #             ego_car = self.carla_world.get_actor(self.ego_car_id)
    #             if ego_car is not None:
    #                 ego_transform = ego_car.get_transform()

    #                 # Publish transform
    #                 map_origin = geometry_msgs.msg.TransformStamped()
    #                 map_origin.header.stamp = rospy.Time()
    #                 map_origin.header.frame_id = "sensor"
    #                 map_origin.child_frame_id = "lidar"
    #                 map_origin.transform.translation.x = -0.04431
    #                 map_origin.transform.translation.y = -0.0056266
    #                 map_origin.transform.translation.z = -0.071236
    #                 roll, pitch, yaw = trans.carla_rotation_to_RPY(ego_transform.rotation)
    #                 quat = euler2quat(roll, pitch, yaw)
    #                 map_origin.transform.rotation.w = 0.0038566
    #                 map_origin.transform.rotation.x = 0.0025168
    #                 map_origin.transform.rotation.y = -0.99999
    #                 map_origin.transform.rotation.z = 0.00075326
    #                 self.tf_broadcaster2.sendTransform(map_origin)

    #         rate.sleep()


    def connect_to_carla(self):
        carla_client = carla.Client('localhost', port=2000)
        carla_client.set_timeout(10)  # second
        try:
            self.carla_world = carla_client.get_world()
        except RuntimeError as e:
            rospy.logerr('Can not connect to CARLA world.')
            raise e

    def id_callback(self, msg):
        if self.ego_car_id is None:
            self.ego_car_id = msg.data
        linear_velocity = self.get_vehicle_speed()
        if linear_velocity is not None:
            self.current_speed = linear_velocity


    def pointcloud_callback(self, msg):
        points = []  # List to hold points
        for point in point_cloud2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
            points.append(point)

        points = np.array(points)
        filtered_points = points[(points[:, 2] < -0.65) | (points[:, 2] > -0.45)]
        
        self.publish_lidar_data(filtered_points, self.lidar_publisher)
        self.publish_lidar_transform()


    def path_callback(self, msg):
        self.latest_poses = msg.poses[-self.pre_path_length:]  # 取出最新的姿态信息
        self.predict_future_poses(MAX_PATH_LENGTH = self.pre_path_length) # 预测未来十个姿态信息


    def imu_callback(self, imu_msg):
        # 解析 IMU 消息，提取车辆的速度和加速度
        linear_acceleration = imu_msg.linear_acceleration
        # angular_velocity = imu_msg.angular_velocity  
        if linear_acceleration is not None:
            # 根据车辆状态更新速度和加速度信息
            self.current_acceleration = linear_acceleration


    def bbx_callback(self, msg):
        for marker in msg.markers:
            if marker.type == Marker.LINE_LIST:
                points = marker.points
                self.check_distance_to_bounding_box(points)


    def check_distance_to_bounding_box(self, bbx_points):
        if self.ego_car_id is not None:
            ego_car = self.carla_world.get_actor(self.ego_car_id)
            if ego_car is not None:
                ego_transform = ego_car.get_transform()
                ego_location = ego_transform.location
                ego_front_location = ego_location + carla.Location(3.0, 0.0, 0.0)
                min_distance = float('inf')  # 初始最小距离设置为正无穷大
                min_bbx_location = None  # 最小距离对应的边界框点的位置
                for point in bbx_points:
                    bbx_location_rel = carla.Location(point.x, point.y, ego_location.z)  # 不考虑 Z 轴坐标
                    bbx_location = ego_transform.transform(bbx_location_rel)  # 将边界框位置从车辆坐标系转换为地图坐标系
                    distance = ego_front_location.distance(bbx_location)
                    if distance < min_distance:
                        min_distance = distance
                        # min_bbx_location = bbx_location
                        # rospy.loginfo("Minimum distance between ego car and bounding box: {} meters".format(min_distance))
                        # rospy.loginfo("Ego car location: x={}, y={}, z={}".format(ego_location.x, ego_location.y, ego_location.z))
                        # if min_bbx_location is not None:
                            # rospy.loginfo("Bounding box point location: x={}, y={}, z={}".format(min_bbx_location.x, min_bbx_location.y, min_bbx_location.z))
                    
                if min_distance < 3.0:
                    self.aeb()
            else:
                rospy.logwarn("Can't find or specified ID does not belong to a vehicle.")
        else:
            rospy.logwarn("ego_car_id has not been set.")


    def predict_future_poses(self, MAX_PATH_LENGTH):
        predict_path = Path()
        while len(self.latest_poses) < 2 or self.current_speed is None or self.current_acceleration is None: # 等一下姿态信息
            rospy.logwarn("Waiting...")
            rospy.sleep(1)

        future_poses = []
        for i in range(self.pre_path_length):  # 预测未来的pre_path_length帧姿态
            t = (i + 1) * 0.05  # 时间间隔为0.05秒 和世界一致
            future_pose = self.predict_pose(t, self.current_speed, self.current_acceleration)
            if len(future_poses) >= MAX_PATH_LENGTH:
                future_pose.pop(0)
                predict_path.poses.pop(0)
            future_poses.append(future_pose)
            predict_path.poses.append(future_pose)
        
        predict_path.header.frame_id = "map"
        predict_path.header.stamp = rospy.Time().now()
        self.predict_path_publisher.publish(predict_path)

    
    def predict_pose(self, t, speed, acceleration):
        # 使用速度和加速度预测未来姿态
        latest_pose = self.latest_poses[-1].pose

        future_pose = PoseStamped()
        future_pose.pose.position.x = latest_pose.position.x + speed.x * t + 0.5 * acceleration.x * t**2
        future_pose.pose.position.y = latest_pose.position.y + speed.y * t + 0.5 * acceleration.y * t**2
        future_pose.pose.position.z = latest_pose.position.z

        # 假设未来姿态的方向与最新姿态一致
        future_pose.pose.orientation = latest_pose.orientation

        ##############
        ##car marker##
        ##############
        self.car_marker = self.create_car_marker(latest_pose)
        self.marker_publisher.publish(self.car_marker)

        return future_pose
    

    def publish_lidar_data(self, lidar_data, publisher):
        msg = PointCloud2()
        msg.header.stamp = rospy.Time().now()
        msg.header.frame_id = "lidar" # 坐标系标识符就是map，仅仅是在原来的点云信号过滤得到的新点云

        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        cloud_struct = struct.Struct(_get_struct_fmt(False, msg.fields))
        buff = ctypes.create_string_buffer(cloud_struct.size * len(lidar_data))

        point_step, pack_into = cloud_struct.size, cloud_struct.pack_into
        offset = 0
        for p in lidar_data:
            pack_into(buff, offset, *p)
            offset += point_step

        msg = PointCloud2(header=msg.header,
                          height=1,
                          width=len(lidar_data),
                          is_dense=False,
                          is_bigendian=False,
                          fields=msg.fields,
                          point_step=cloud_struct.size,
                          row_step=cloud_struct.size * len(lidar_data),
                          data=buff.raw)

        publisher.publish(msg)


    def aeb(self):
        if self.ego_car_id is not None:
            ego_car = self.carla_world.get_actor(self.ego_car_id)
            brake_control = carla.VehicleControl(brake=1, steer=0, throttle=0)
            ego_car.apply_control(brake_control)
            rospy.loginfo('AEB triggered. Applying brake to ego car.')   


    def get_vehicle_speed(self):
        # 获取当前车辆速度
        if self.ego_car_id is not None:
            ego_car = self.carla_world.get_actor(self.ego_car_id)
            if ego_car is not None:
                velocity = ego_car.get_velocity()
                # speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])  # 计算速度的模长
                # if speed < 1e-5:
                #     speed = 0
                return velocity
                # rospy.loginfo("当前车辆速度：{} m/s".format(speed))
            else:
                rospy.logwarn("找不到或者ID指定的对象不是车辆")
        else:
            rospy.logwarn("ego_car_id尚未设置")   


    def create_car_marker(self, latest_pose):
        car_marker = Marker()
        car_marker.header.frame_id = "map"
        car_marker.type = Marker.CUBE
        car_marker.action = Marker.ADD
        car_marker.scale.x = 2.0
        car_marker.scale.y = 1.0
        car_marker.scale.z = 0.5
        car_marker.color.a = 1.0  # 完全不透明
        car_marker.color.r = 0.0
        car_marker.color.g = 1.0
        car_marker.color.b = 0.0
        car_marker.pose.orientation = latest_pose.orientation
        car_marker.pose.position = latest_pose.position
        return car_marker   
        

    def main(self):
        # lidar_transform_thread = threading.Thread(target=self.publish_lidar_transform)
        # lidar_transform_thread.start() 
        rospy.spin()


    def print_path(self):
        latest_poses = self.latest_poses
        if latest_poses:
            for i, pose in enumerate(latest_poses):
                rospy.loginfo("Latest pose {}: ".format(i+1))
                self.print_pose_information(pose)
        else:
            rospy.loginfo("No latest poses available.")  # 如果 latest_poses 为空，则打印消息


    def print_pose_information(self, pose):
        position = pose.pose.position
        orientation = pose.pose.orientation

        rospy.loginfo("Position: x={}, y={}, z={}".format(position.x, position.y, position.z))
        rospy.loginfo("Orientation: w={}, x={}, y={}, z={}".format(orientation.w, orientation.x, orientation.y, orientation.z))


if __name__ == '__main__':
    try:
        subscriber_node = SubscriberNode()
        subscriber_node.connect_to_carla()
        subscriber_node.main()
    except rospy.ROSInterruptException:
        pass

