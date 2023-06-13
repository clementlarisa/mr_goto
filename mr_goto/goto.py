from __future__ import print_function
import sys
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from geometry_msgs.msg import Point, Pose, PoseStamped
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry, Path

from scipy import spatial


class PointWrapper:
    def __init__(self, point):
        assert isinstance(point, Point)

        self.point = point
        self.min_speed = 0.5
        self.ttc = 0
        self.margin = 0.3
        self.velocity=0

class pure_pursuit(Node):
    def __init__(self):
        super().__init__('goto_node')
        self.get_logger().info("HEY from goto")
        #Topics & Subscriptions,Publishers
        lidarscan_topic = '/scan'
        drive_topic = '/pure_pursuit_nav'
        map_topic = '/map'
        odom_topic = '/odom'
        path_topic = '/path2'

        self.path = Path() 
        self.actual_path = Path()
        self.actual_path.header.frame_id="map"

        self.ground_pose = Pose()
        # self.L = 2
        self.L_factor = 2.
        self.odom_frame = ""
        self.odom_stamp = self.get_clock().now()
        self.velocity = 0.

        self.start_pose = (0, 0)
        self.goal_pose = Point()

        self.ttc_array = []
        self.ttc = 1000
        self.ttc_front = 1000
        self.speed = 0.
        self.steering_angle = 0.
        self.prev_error = 0.

        self.scan_msg = LaserScan()


        self.path_error = 0.

        self.prev_ground_path_index = None
        self.timestamp = 0
        self.n_log = 50

        self.tf_buffer = tf2_ros.Buffer(Duration(seconds=100.0))  # tf buffer length
        self.tf_listener = tf2_ros.TransformListener(buffer=self.tf_buffer, node=self)

        self.path_pub = self.create_subscription(Path, path_topic, self.path_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 1)
        self.lidar_sub = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 1)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 1)
        self.marker_pub = self.create_publisher(Marker, "/marker_goal", 1000)
        self.actual_path_pub = self.create_publisher(Path, "/actual_path", 1)

    def myScanIndex(self, scan_msg, angle):
        # expects an angle in degrees and outputs the respective index in the scan_ranges array.
        # angle: between -135 to 135 degrees, where 0 degrees is directly to the front

        rad_angle = angle * math.pi / 180.0

        if not (scan_msg.angle_min <= rad_angle <= scan_msg.angle_max):
            self.get_logger().info("ANGLE out of range: " + str(angle))
            return 540

        index = int((rad_angle - scan_msg.angle_min) / scan_msg.angle_increment)

        return index
    
    def myScanAngle(self, scan_msg, index):
        # returns angle in radians

        return scan_msg.angle_min + index * scan_msg.angle_increment

    def lidar_callback(self, data):
        self.scan_msg = data
        scan_ranges = np.array(data.ranges)
        scan_angles = np.linspace(data.angle_min, data.angle_max, len(scan_ranges))
        projected_speed_array = self.velocity * np.cos(scan_angles)
        projected_speed_array[projected_speed_array < 0.1] = 0.1
        self.ttc_array = (np.maximum(0,scan_ranges - 0.3)) / projected_speed_array
        self.ttc = np.amin(self.ttc_array[self.myScanIndex(self.scan_msg, math.degrees(self.steering_angle) - 30):self.myScanIndex(self.scan_msg, math.degrees(self.steering_angle) + 30)])
        # self.ttc = self.ttc_array[self.ttc_index]
        self.ttc_front = self.ttc_array[self.myScanIndex(data, math.degrees(self.steering_angle))]

    def odom_callback(self, data):
        """ Process each position update using the Pure Pursuit algorithm & publish an AckermannDriveStamped Message
        """
        self.ground_pose = data.pose.pose
        self.odom_frame = data.header.frame_id
        self.odom_stamp = data.header.stamp
        self.velocity = data.twist.twist.linear.x
        self.timestamp += 1

        if self.velocity < sys.float_info.min:
            self.prev_ground_path_index = None

        # self.L = 8. * self.velocity if self.velocity > sys.float_info.min else 2.
        # self.L  = 2
        # self.L = 0.59259259259259 * self.velocity + 1.8518518518519 if self.velocity > sys.float_info.min else 2.
        # self.L = 0.81481481481482 * self.speed + 0.2962962962963 if self.speed > sys.float_info.min else 1.5
        self.L = 0.35 * self.velocity + 0.18 if self.velocity > sys.float_info.min else 1.
        # self.L *= self.L_factor

        poses_stamped = self.path.poses
        if len(poses_stamped) == 0:
            return
        
        poses =  np.array([(p.pose.position.x, p.pose.position.y) for p in poses_stamped])
        tree = spatial.KDTree(poses)

        # find closest path point to current pose
        ground_pose = (self.ground_pose.position.x, self.ground_pose.position.y)
        self.append_pose(ground_pose[0], ground_pose[1])

        if self.prev_ground_path_index is None:
            dists = np.linalg.norm(poses - ground_pose, axis=1)
            # dx = [abs(ground_pose[1] - pose[0]) for pose in poses]
            # dy = [abs(ground_pose[0] - pose[1]) for pose in poses]
            # dists = np.hypot(dx, dy)
            self.prev_ground_path_index = np.argmin(dists)
            self.get_logger().info("start index.x: " + str(self.prev_ground_path_index), throttle_duration_sec=1)


        prev_dist = np.linalg.norm(poses[self.prev_ground_path_index] - ground_pose)
        i = self.prev_ground_path_index + 1
        while i < len(poses):
        # for i in range (0, len(poses)):
            dist = np.linalg.norm(poses[i] - ground_pose)
            if dist > prev_dist:
                break
            prev_dist = dist
            i += 1
        start_index = i - 1
        start_pose = poses[start_index]
        self.prev_ground_path_index = start_index
        self.start_pose = start_pose
        # start_index = tree.query((self.ground_pose.position.x, self.ground_pose.position.y))[1]
        # start_pose = poses[start_index

        # find goal point on path at lookahead_distance L
        dist = 0
        i = start_index + 1
        while i < len(poses):
            # dist += np.linalg.norm(poses[i] - start_pose)
            dist = np.linalg.norm(poses[i] - ground_pose)
            if dist > self.L:
                break
            i += 1
                
        goal = poses[i - 1]
        
        # path_points_in_range = tree.query_ball_point(start, self.L, return_sorted=True)
        # goal = poses[path_points_in_range[len(path_points_in_range) - 1]]


        # transform goal point to vehicle coordinate frame
        transform = self.tf_buffer.lookup_transform("base_link", self.path.header.frame_id, self.get_clock().now())
        goal_transformed = tf2_geometry_msgs.do_transform_point(PointWrapper(Point(goal[0], goal[1], 1)), transform).point

        self.goal_pose = goal_transformed
    
        self.path_error = goal_transformed.y

        curvature = 2 * goal_transformed.y / pow(self.L, 2)
        R = 1 / curvature

        # self.steering_angle = 1 / np.tan(curvature * 0.3302)
        self.steering_angle = np.arctan(0.3302 * curvature)
        # steering_angle = np.arctan(1 / R)
        self.steering_angle = np.clip(self.steering_angle, -0.4189, 0.4189)

        self.speed = self.compute_speed()
        
        # rclpy.loginfo_throttle(1, "goal_transformed.x: " + str(goal_transformed.y))
        if self.timestamp % self.n_log == 0:
            self.timestamp = 0
            # rclpy.loginfo(f"Ground-truth position: {ground_pose}")

        self.publish_drive(self.speed, self.steering_angle)
        self.actual_path_pub.publish(self.actual_path)
        self.visualize_point(goal[0], goal[1])
        # self.visualize_point(start_pose[0], start_pose[1])
    
    def path_callback(self, data):
        self.path = data
     

    def publish_drive(self, speed, angle):
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now()
        # drive_msg.header.frame_id = "laser"
        drive_msg.drive.steering_angle = angle
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

    def visualize_point(self,x,y,frame='map',r=0.0,g=1.0,b=0.0):
        marker = Marker()
        marker.header.frame_id = frame
        marker.header.stamp = self.get_clock().now()
        marker.id = 150
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 1
        marker.color.a = 1.0
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0
        marker.lifetime = Duration(seconds=0.1)
        self.marker_pub.publish(marker)

    def compute_speed(self):
        # choose front beam ttc if minimum ttc is not in fov [-45, 45]
        # if not -45 < math.degrees(self.myScanAngle(self.scan_msg, self.ttc_index)) < 45:
        #     # speed = 5 * ttc_array[self.myScanIndex(scan_msg, math.degrees(self.steering_angle))]
        #     speed = min(7, max(0.5, 7 * (1 - math.exp(-0.5*self.ttc_front))))
        # else:
        #     speed = min(7, max(0.5, 7 * (1 - math.exp(-0.75*self.ttc))))

        # alpha = np.arcsin(self.goal_pose.y / self.L)

        if self.goal_pose.x < 0:
            return 0.1

        alpha = self.steering_angle
        self.get_logger().info("alpha: " + str(alpha), throttle_duration_sec=1)

        # b = math.sqrt(pow(self.start_pose[0] - self.ground_pose.position.x, 2) + pow(self.start_pose[1] - self.ground_pose.position.y, 2))
        b = self.path_error
        dt = b * math.cos(alpha)

        projected_path_error = dt + self.L * math.sin(alpha)

        # ttc = self.ttc_array[self.myScanIndex(self.scan_msg, math.degrees(alpha))]
        ttc = self.ttc
        speed = min(7, max(0.25, 7 * (1 - math.exp(-0.75*ttc))))
        
        # clip speed by steering angle
        speed /= 10. * pow(self.steering_angle, 2) + 1


        speed *= max(1, 1 / (pow(10 * projected_path_error, 2))) if projected_path_error > sys.float_info.min else 1.

        # speed *= 5 * abs(path_error - self.prev_error)
        self.prev_error = projected_path_error

        # speed /= abs(path_error) + 1 if self.path_error > sys.float_info.min else 1.

        self.get_logger().info("path error: " + str(projected_path_error), throttle_duration_sec=1)
        # rclpy.loginfo_throttle(1, "velocity: " + str(self.velocity))

        # clip = math.exp(3*abs(self.steering_angle) - 2)
        # diff = speed - clip
        # if diff > 0.25:
        #     speed = diff
        # else:
        #     speed = 0.25
        

        return speed
    
    def append_pose(self, pos_x, pos_y):
        cur_pose = PoseStamped()
        cur_pose.header.stamp = self.get_clock().now()
        cur_pose.header.frame_id = self.odom_frame
        cur_pose.pose.position.x = pos_x
        cur_pose.pose.position.y = pos_y
        cur_pose.pose.position.z = 0
        self.actual_path.poses.append(cur_pose)

def main():
    print('Hi from mr_goto.')
    rclpy.init()
    rfgs = pure_pursuit()
    # rclpy.sleep(0.1)
    rclpy.spin(rfgs)
    rfgs.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
