#!/usr/bin/env python

import rospy
import carla
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Float32, Int32


class SubscriberNode:
    def __init__(self):
        rospy.init_node('subscriber_node', anonymous=True)

        rospy.Subscriber('/carla/ego_vehicle/lidar', PointCloud2, self.pointcloud_callback)
        rospy.Subscriber('/carla/ego_vehicle/brake', Float32)
        rospy.Subscriber('/carla/ego_vehicle/id', Int32, self.id_callback)

        self.carla_world = None
        self.connect_to_carla()
        self.ego_car_id = None

    def connect_to_carla(self):
        carla_client = carla.Client(host="192.168.1.8", port=2000)
        carla_client.set_timeout(10)  # second
        try:
            self.carla_world = carla_client.get_world()
        except RuntimeError as e:
            self.get_logger().error('can not connect to CARLA world.')
            raise e

    def id_callback(self, msg):
        if self.ego_car_id is None:
            self.ego_car_id = msg.data
        else:
            pass

    def pointcloud_callback(self, msg):
        rospy.loginfo("Received PointCloud2 message")
        for point in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = point
            # Exclude point clouds near z=-2.5
            if not ((z > -2.6 and z < -2.4) or (x > 2.0)):
                rospy.loginfo("Point: x={}, y={}, z={}".format(x, y, z))
                self.aeb()

    def aeb(self):
        if self.ego_car_id is not None:
            # Get ego car by id
            ego_car = self.carla_world.get_actor(self.ego_car_id)
            # Apply brake control to ego car
            brake_control = carla.VehicleControl(brake=1, steer=0, throttle=0)
            ego_car.apply_control(brake_control)
            rospy.loginfo('AEB triggered. Applying brake to ego car.')


    def main(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        subscriber_node = SubscriberNode()
        subscriber_node.main()
    except rospy.ROSInterruptException:
        pass
