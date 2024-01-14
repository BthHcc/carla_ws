#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2


def pointcloud_callback(msg):
    rospy.loginfo("Received PointCloud2 message")
    for point in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        x, y, z = point
        rospy.loginfo("Point: x={}, y={}, z={}".format(x, y, z))


def main():
    rospy.init_node('pointcloud_subscriber_node', anonymous=True)
    rospy.Subscriber('pointcloud_topic', PointCloud2, pointcloud_callback)
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
