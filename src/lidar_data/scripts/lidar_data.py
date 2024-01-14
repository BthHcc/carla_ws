import carla
import numpy as np
import sys
import rospy
from sensor_msgs.msg import PointCloud2, PointField

from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__))))
from carla_setup import setup_carla_environment

class LidarPublisher:
    def __init__(self):
        rospy.init_node('lidar_publisher_node', anonymous=True)
        self.lidar_publisher = rospy.Publisher('pointcloud_topic', PointCloud2, queue_size=10)

    def publish_lidar_data(self, lidar_data):
        msg = PointCloud2()
        msg.header.stamp = rospy.Time().now()
        msg.header.frame_id = "lidar1"

        msg.height = 1
        msg.width = len(lidar_data)

        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * len(lidar_data)
        msg.is_dense = False
        msg.data = np.asarray(lidar_data, np.float32).tostring()

        self.lidar_publisher.publish(msg)
        # print("published...")

    def destroy(self):
        rospy.signal_shutdown('Shutting down lidar_publisher_node')


def main():
    actor_list = []
    sensor_list = []

    lidar_publisher = LidarPublisher()

    try:
        client, world, blueprint_library, ego_vehicle, lidar = setup_carla_environment()

        lidar.listen(lambda data: lidar_callback(data, lidar_publisher))

        sensor_list.append(lidar)

        while True:
            # set the sectator to follow the ego vehicle
            spectator = world.get_spectator()
            transform = ego_vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20),
                                                    carla.Rotation(pitch=-90)))

    finally:
        print('destroying actors')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        for sensor in sensor_list:
            sensor.destroy()
        lidar_publisher.destroy()
        print('done.')


# Convert lidar in carla to publish in pointcloud2 format in ros
def lidar_callback(point_cloud, lidar_publisher):
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))
    points = data[:, :-1]
    lidar_publisher.publish_lidar_data(points)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
