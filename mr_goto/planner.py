#!/usr/bin/env python3
import sys
import re
import numpy as np
import skimage
import time
from collections import defaultdict
import heapq as heap
from functools import wraps
import yaml
from yaml.loader import SafeLoader

from mr_goto.bspline import approximate_b_spline_path

import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry, OccupancyGrid, Path, MapMetaData
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped, Point, Quaternion
from visualization_msgs.msg import Marker


def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


def timing(f):
    "A simple decorator for timing functions"
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        return result
    return wrap



class PathGenerator(Node):
    """ Creating a path for the car to drive using only data from the /map topic.
    """

    def __init__(self):
        super().__init__('planner')
        self.get_logger().info("HEY from planner")

        # Subscribers
        self.map_sub = self.create_subscription(OccupancyGrid, "map", self.map_callback, 1)
        self.initial_pose_sub = self.create_subscription(PoseWithCovarianceStamped, "initialpose", self.initial_pose_callback, 1)
        self.ground_truth_sub = self.create_subscription(Odometry, "ground_truth", self.ground_truth_callback, 1)
        self.goal_pose_sub = self.create_subscription(PoseStamped, "goal_pose", self.goal_pose_callback, 1)

        # Publishers

        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/nav", 1000)
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", qos_profile=QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        ))
        self.marker_pub = self.create_publisher( Marker, "/visualization_marker", 1000)
        self.path_pub = self.create_publisher( Path, '/path', 1)
        self.path2_pub = self.create_publisher(Path, '/path2', 1)
        self.map_pub_timer = self.create_timer(1.0, self.publish_map)
        self.map_seq_id = 0

        # Controller parameters
        self.sparsity = 5 
        self.scale = 1 # by which factor to downscale the map resolution before performing the path generation
        self.safety_margin = 0.2 # in meters
        self.occupancy_treshhold = 10 # pixel below this treshold (in percent) we consider free space

        self.declare_parameter('goal_point_x', 3.)
        self.declare_parameter('goal_point_y', 3.)
        # self.parameter_timer = self.create_timer(1, self.param_timer_callback)

        self.initial_pose = None
        self.goal_pose = rclpy
        self.map = OccupancyGrid()
        self.map_matrix = np.matrix([[]])

    
    def preprocess_map(self, map_msg):
        """
        Converts the map_msg.data array into a 2D numpy array.
        Current assumption: The map is centered at (0,0), which is also the robot's initial pose.
        WARNING: map_msg.info.origin represents the lower-right pixel of the map, not the starting position of the robot!
        Shouldn't we use /gt_pose messages to determine the inital pose of the robot, or can we assume it to always be at (0,0)?
        """

        # TODO: Calculate shortest path without downscaling the map
        self.map_width = int(np.ceil(map_msg.info.width/self.scale))
        self.map_height = int(np.ceil(map_msg.info.height/self.scale))
        self.map_res = map_msg.info.resolution*self.scale
        
        map_data = np.array(map_msg.data).reshape((map_msg.info.width, map_msg.info.height)).T
        map_data = skimage.measure.block_reduce(map_data, (self.scale,self.scale), np.min)

        # Set unknown values to be occupied
        map_data[map_data == - 1] = 100
        map_binary = (map_data < self.occupancy_treshhold).astype(int)


        return map_binary
    
    def publish_map(self):
        self.map.header.stamp = self.get_clock().now().to_msg()
        self.map.header.frame_id = 'map'
        self.map_pub.publish(self.map)

    def calculate_finish_line(self, driveable_area):
        # TODO: Currently, we assume the car is always facing straight forward at the start
        # Maybe adjust to calculate the finish line perpendicular to the inital orientation of the car?
        x = self.goal_point[0]
        y = self.goal_point[1]
        left_end = y
        right_end = y

        while driveable_area[x, right_end] == 1:
            right_end += 1

        while driveable_area[x, left_end] == 1:
            left_end -= 1  

        # self.get_logger().info(f"{right_end=}, {left_end=}")
        return (x, left_end), (x, right_end)

    def save_map_image(self, map, path):
        map_image = np.flip(np.flip(map, 1), 0)

        skimage.io.imsave(path, skimage.img_as_ubyte(255*map_image), check_contrast=False)
        self.get_logger().info(f"Saved map image to {path}")
    
    def erode_map(self, driveable_area):
        radius = int(self.safety_margin/self.map_res)

        tic = time.time()
        eroded_map = skimage.morphology.binary_erosion(driveable_area, footprint = np.ones((2*radius,2*radius)))
        toc = time.time()
        self.get_logger().info(f"Time for binary erosion: {toc - tic}")

        self.save_map_image(driveable_area, '/home/parallels/projects/mobile_robotics/ws02/src/mr_goto/maps/driveable_area.png')
        self.save_map_image(eroded_map, '/home/parallels/projects/mobile_robotics/ws02/src/mr_goto/maps/eroded_map.png')

        return eroded_map
    
    @timing
    def dijkstra(self, map_msg, safe_area, finish_line_start, finish_line_end, neighborhood):
        # Currently implemented with a 4-neighborhood - since Dijkstra is equivalent to breadth-first search
        # for uniform weights
        # Currently expects the finishline to always be horizontal
        # TODO: Make this into proper Dijkstra with an 8-neighborhood with diagonal weights of sqrt(2)
        x = finish_line_start[0]
        # finish_line = [(x,y) for y in range(finish_line_start[1], finish_line_end[1] + 1)]
        finish_line = [self.goal_point]

        visited = np.array(safe_area)
        visited.fill(False)

        priority_queue = []

        nodeCosts = defaultdict(lambda: float('inf'))
        for cell in finish_line:
            if cell == self.starting_point:
                visited[cell] = True
            else:
                nodeCosts[cell] = 0
                heap.heappush(priority_queue, (0, cell))

        previous_node = {}

        i = 0
        while priority_queue:
            i += 1
            dist, (x,y) = heap.heappop(priority_queue)

            if (x,y) == self.starting_point:
                return previous_node, dist

            visited[x,y] = True

            # Funky visualization of where the algorithm is currently exploring
            if i % 100 == 0:
                pos_x, pos_y = self.convert_grid_cell_to_position(x,y)
                marker = self.visualize_point(pos_x,pos_y)
                self.marker_pub.publish(marker)

            # Force the search to go down at the start in order to complete a whole lap
            if (x,y) in finish_line:
                new_x = x - 1
                new_y = y

                if safe_area[new_x, new_y] == 1:
                    new_costs = nodeCosts[(x,y)] + 1
                    if new_costs < nodeCosts[(new_x,new_y)]:
                        previous_node[(new_x,new_y)] = (x,y)
                        nodeCosts[(new_x,new_y)] = new_costs
                        heap.heappush(priority_queue, (new_costs, (new_x,new_y)))
            else:
                for delta_x, delta_y, weight in neighborhood:
                    try:
                        new_x = x + delta_x
                        new_y = y + delta_y 

                        # The loop is only completed if the start point is reached from above
                        if (new_x,new_y) == self.starting_point and new_x == x - 1:
                            new_costs = nodeCosts[(x,y)] + weight
                            if new_costs < nodeCosts[(new_x,new_y)]:
                                previous_node[(new_x,new_y)] = (x,y)
                                nodeCosts[(new_x,new_y)] = new_costs
                                heap.heappush(priority_queue, (new_costs, (new_x,new_y)))

                        if visited[new_x, new_y]:
                            continue

                        if safe_area[new_x, new_y] == 1:
                            new_costs = nodeCosts[(x,y)] + weight
                            if new_costs < nodeCosts[(new_x,new_y)]:
                                previous_node[(new_x,new_y)] = (x,y)
                                nodeCosts[(new_x,new_y)] = new_costs
                                heap.heappush(priority_queue, (new_costs, (new_x,new_y)))
                    except:
                        continue
    
    @timing
    def shortest_path(self, map_msg, safe_area, finish_line_start, finish_line_end, neighborhood):
        "Use Dijkstra with a 4-neighborhood or an 8-neighborhood"

        previous_node, dist = self.dijkstra(map_msg, safe_area, finish_line_start, finish_line_end, neighborhood)
        x = finish_line_start[0]
        one_before_finish_line = [(x-1,y) for y in range(finish_line_start[1], finish_line_end[1] + 1)]

        shortest_path = Path()
        pos_x, pos_y = self.convert_grid_cell_to_position(self.starting_point[0],self.starting_point[1])
        self.append_pose(map_msg, shortest_path, pos_x, pos_y)
        
        node = previous_node[self.starting_point]
        i = 0
        # the shortest path can move on the finish line at the start
        while node not in one_before_finish_line:
            i += 1
            node = previous_node[node]

            if (i % self.sparsity) == 0:
                pos_x, pos_y = self.convert_grid_cell_to_position(node[0],node[1])
                self.append_pose(map_msg, shortest_path, pos_x, pos_y)
        
        return shortest_path, dist
    
    @timing
    def optimize_raceline(self, map_msg, shortest_path):
        x_array = np.array([pose.pose.position.x for pose in shortest_path.poses])
        y_array = np.array([pose.pose.position.y for pose in shortest_path.poses])

        rax, ray, heading, curvature = approximate_b_spline_path(
        x_array, y_array, len(x_array), degree = 3, s=0.5)

        optimized_path = Path()
        distance = 0

        prev_x = rax[0]
        prev_y = ray[0]
        for (x,y) in zip(rax,ray):
            distance += np.sqrt((x-prev_x)**2 + (y-prev_y)**2)
            self.append_pose(map_msg, optimized_path, x, y)
            prev_x = x
            prev_y = y

        return optimized_path, distance
    
    def append_pose(self, map_msg, path, pos_x, pos_y):
        cur_pose = PoseStamped()
        cur_pose.header = map_msg.header
        cur_pose.pose.position.x = pos_x
        cur_pose.pose.position.y = pos_y
        cur_pose.pose.position.z = 0.0
        path.poses.append(cur_pose)

    
    def convert_position_to_grid_cell(self, pos_x, pos_y):
        "Takes a position in meters and converts it to the corresponding grid cell in the OccupancyGrid"

        index_x = int(pos_x/self.map_res + self.map_height/2)
        index_y = int(pos_y/self.map_res + self.map_width/2)

        return index_x, index_y
    
    def convert_grid_cell_to_position(self, index_x, index_y):
        "Takes a tuple (i,j) of indices on the grid and converts it to its coordinates in meters."
        
        pos_x = (index_x - self.map_height/2)*self.map_res
        pos_y = (index_y - self.map_width/2)*self.map_res

        return pos_x, pos_y


    @timing
    def fill4(self, map_binary, x, y):
        """Source: https://wikipedia.org/wiki/Floodfill
        0 is occupied
        1 is free space
        2 is driveable area
        """

        stack = []
        stack.append((x, y))
        while stack != []:
            (x, y) = stack.pop()
            if map_binary[x,y] == 1:
                map_binary[x,y] = 2
                if y + 1 < self.map_height:
                    stack.append((x, y + 1))
                if y - 1 >= 0:
                    stack.append((x, y - 1))
                if x + 1 < self.map_width:
                    stack.append((x + 1, y))
                if x - 1 >= 0:
                    stack.append((x - 1, y))
        
        return map_binary == 2
    
    def ground_truth_callback(self, data):
        self.initial_pose = data.pose.pose

    def initial_pose_callback(self, pose_msg):
        self.initial_pose = pose_msg.pose
        goal_point_x = self.get_parameter('goal_point_x').get_parameter_value().double_value
        goal_point_y = self.get_parameter('goal_point_x').get_parameter_value().double_value
        self.goal_pose = Pose(position=Point(x=float(goal_point_x), y=float(goal_point_y), z=1.), orientation=Quaternion(x=0., y=0., z=0., w=0.))
        self.get_logger().debug(f"Initial Pose: {pose_msg.pose}")
        self.plan()

    def goal_pose_callback(self, pose_msg):
        self.goal_pose = pose_msg.pose
        self.get_logger().debug(f"Goal Pose: {pose_msg.pose}")
        if self.initial_pose is not None:
            self.plan()

    def plan(self):
        if self.initial_pose is None or self.goal_pose is None:
            self.get_logger().info("something is not right")
            return
        
        map_binary = self.preprocess_map(self.map)
        self.get_logger().info(f"number of free grid cells: {np.sum(map_binary)}")

        self.starting_point = self.convert_position_to_grid_cell(self.initial_pose.pose.position.x, self.initial_pose.pose.position.y)
        self.goal_point = self.convert_position_to_grid_cell(self.goal_pose.position.x, self.goal_pose.position.y)
        driveable_area = self.fill4(map_binary, self.starting_point[0], self.starting_point[1])
        self.get_logger().info(f"number of driveable grid cells: {np.sum(driveable_area)}")

        # finish_line_start, finish_line_end = self.calculate_finish_line(driveable_area)    
        finish_line_start = self.goal_point
        finish_line_end = self.goal_point
        safe_area = self.erode_map(driveable_area)
        self.get_logger().info(f"number of safe grid cells: {np.sum(safe_area)}")

        
        # possible neighborhoods encoded in (delta_x, delta_y, weight) format
        neighborhood4 = [(0,1,1), (0,-1,1), (1,0,1), (-1,0,1)]
        neighborhood8 = [(0,1,1), (0,-1,1), (1,0,1), (-1,0,1), (1,1,np.sqrt(2)), (1,-1,np.sqrt(2)), (-1,1, np.sqrt(2)), (-1,1, np.sqrt(2))]

        shortest_path, distance = self.shortest_path(self.map, safe_area, finish_line_start, finish_line_end, neighborhood4)
        self.get_logger().info(f"Length of shortest path: {self.map_res * distance} meters")

        shortest_path, distance = self.shortest_path(self.map, safe_area, finish_line_start, finish_line_end, neighborhood8)
        self.get_logger().info(f"Length of shortest path (with diagonals): {self.map_res * distance} meters")

        optimized_path, distance = self.optimize_raceline(self.map, shortest_path)
        self.get_logger().info(f"Length of optimized path (with diagonals): {distance} meters")

        shortest_path.header = self.map.header
        self.path_pub.publish(shortest_path)

        optimized_path.header = self.map.header
        self.path2_pub.publish(optimized_path)


    def map_callback(self, map_msg):
        # self.get_logger().info(f"Map Header: {map_msg.header}")
        # self.get_logger().info(f"Map info: {map_msg.info}")

        self.map = map_msg

    def visualize_point(self,x,y,frame='map',r=0.0,g=1.0,b=0.0):
        marker = Marker()
        marker.header.frame_id = frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 150
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.a = 1.0
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.lifetime = Duration(seconds=0.25).to_msg()
        return marker



def main():
    rclpy.init()
    follow_the_gap = PathGenerator()
    map_matrix = read_pgm("/home/parallels/projects/mobile_robotics/ws02/src/mr_goto/maps/cave.pgm").copy()
    normalized = map_matrix * 100. / np.max(map_matrix)
    map_matrix = normalized.astype(float)
    with open("/home/parallels/projects/mobile_robotics/ws02/src/mr_goto/maps/cave.yaml") as f:
        map_yaml = yaml.load(f, Loader=SafeLoader)
        map = OccupancyGrid()
        origin = map_yaml['origin']
        origin_point = Point()
        origin_point.x = float(origin[0])
        origin_point.y = float(origin[1])
        origin_point.z = float(origin[2])
        origin_pose = Pose()
        origin_pose.position = origin_point
        origin_pose.orientation = Quaternion()
        map.info = MapMetaData()
        map.info.map_load_time = follow_the_gap.get_clock().now().to_msg()
        map.info.width = map_matrix.shape[0]
        map.info.height = map_matrix.shape[1]
        map.info.origin = origin_pose
        map.info.resolution = map_yaml['resolution']
        data = map_matrix.astype(int).flatten()
        free_thres = map_yaml['free_thresh']
        occupied_thres = map_yaml['occupied_thresh']
        # data[data > 1. - free_thres] = 0
        # data[data < 1. - occupied_thres] = 100
        data[data < free_thres] = 0
        data[data > occupied_thres] = 100
        data_copy = data.copy()
        data[data == 100] = 0
        data[data_copy == 0] = 100
        map.data = data.tolist()
        follow_the_gap.map = map
        follow_the_gap.map_matrix = map_matrix
        rclpy.spin(follow_the_gap)
        follow_the_gap.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
	main()
