# mr_goto
## Overview
This is a ROS2 package, developed in python by the collaborators: Mihai Stanusoiu, Larisa Clement.
The package is dependent on skimage. To install skimage, one can use pip:
```sh
pip install scikit-image
```
The package provides a parameter file params.yaml. Before starting the service, the parameters for the goto and ekf nodes need to be set accordingly. There, absolute paths must be provided for the mao files.

How to start the service:
```sh
ros2 launch launch/goto.launch.py params_file:=params.yaml localization:=pf
```

## Tasks

**Total points achieved:**

### 1. GoTo

- Your planner is using your self-localization (20 Points). This means listening to the `/estimated_pose` topic  <br>If you like to skip this point, you can use the **/ground_truth** data/msg in your planner.
- Your planner can be operated with rviz (20 Points) <br>If you like to skip this point, you can use command arguments or rqt to define a target pose where to drive to.

#### 1.1 New Node

- implemented the planner in a newly created node (50 Points)

#### 1.2 Simple, no Obstacle

- Your vehicle can drive to a goal location and stops there. (25 Points)
- Your vehicle can drive to a goal location, stops there and turns into the correct pose. (25 Points)

#### 1.3 Avoid obstacle

- Your vehicle can drive to a goal location even if there is obstacle with 1x1m (movable box) in size in between. (25 Points)
- Your vehicle can drive to a goal location even if there is a cave obstacle such the one [-5,-3] in between. (25 Points)

#### 1.4 Plan

- Write a node or modify the planner and/or self-localization to plan a path to the goal location using waypoints and publish it as ROS nav_msgs/Path message. (50 Points)
- Make the local planner to follow the nav_msgs/Path. (50 Points)

### 2. Publish the used map.

#### 2.1 Publish the used map. (45 Points)

To work with RViz you need the map published. Therefore, publish the map by yourself / in your node on the topic **/map**, once every second.

#### 2.2 Use a published map. (45 Points)

Instead of loading the map from a file, use the `OccupancyGrid` published on the `/map` topic.

### 3. Initialize self-localization and trigger driving using RViz (50 Points)
Assignee: Larisa
RViz should be used to initialize your self-localization. Therefore, you have to listen to **/initialpose** (25 Points) in your self localization node.
RViz should also be used to trigger your driving behaviour. Therefore, you have to listen to `/goal` (25 Points) in your planner node.

Print a ROS debug message if you receive a message on one of these topics.

### 4. Self-Localization tf (45 Points)

Use `/tf` to get the estimated robot position. Publish the estimated position to `/tf` in the self-locaization and listen to `/tf` in your planner.

**If your vehicle can reach a goal pose by planning a path and following the path, you are our hero.**

## 5. launch file

### 5.1 basic launch. (20 Points)

Write a launch file which starts all your nodes. (Stage, tuw_laserscan_features, mr_ekf or mr_pf, mr_move or goto or ...)
### 5.2 optional ekf or pf. (20 Points)

Add a argument named _localization_ to the launch file which allows starting eider the mr_ekf or the mr_pf node with _ekf_ or _pf_

### 5.3 Parameter file for nodes as parameter (20 Points)

The launch file should start nodes with a parameter file, the parameter file name must be an argument.

### 5.4 relative path for the parameter file name argument. (20 Points)

The parameter file name argument should be a relative file name.
