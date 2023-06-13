#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, TextSubstitution, PathJoinSubstitution
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetLaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def launch_setup(context, *args, **kwargs):
    this_directory = get_package_share_directory('mr_goto')

    localization_mode = LaunchConfiguration('localization').perform(context)
    params_file = LaunchConfiguration(
        "params_file")
    params_file_value = params_file.perform(context)
    params_file = PathJoinSubstitution(
        [
            this_directory,
            "config",
            params_file_value
        ]
    )
    if localization_mode == "ekf":
        print("Chose ekf")
        localization_node = Node(package='mr_ekf', executable='ekf_node',
                                 name='ekf', remappings=[('/scan', 'base_scan')], parameters=[params_file])
    if localization_mode == "pf":
        localization_node = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(
                get_package_share_directory('mr_pf'), 'launch'),
                '/pf_launch.py']),
            launch_arguments={
                'particle_filter_params_file': 'particle_filter.yaml',
                'particle_filter_map_file': 'cave.png',
                'level': '0'
            }.items()
        )

    goto_node = Node(
        package='mr_goto',
        executable='goto',
        name='goto',
        parameters=[params_file]
    )

    planner_node = Node(
        package='mr_goto',
        executable='planner',
        name='planner'
    )

    # stage_node = Node(
    #     package='stage_ros2',
    #     executable='stage_ros2',
    #     name='stage',
    #     output='screen'
    # )

    # rviz_node = Node(
    #     package='rviz2',
    #     executable='rviz2',
    #     output='screen')

    laser_scan_features_node = Node(
        package='tuw_laserscan_features',
        executable='composed_node',
        remappings=[('/scan', 'base_scan')],
        output='screen')

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(
                get_package_share_directory('mr_nav2'), 'launch', 'nav2'),
                '/rviz_launch.py']),
            launch_arguments={
                'particle_filter_params_file': 'particle_filter.yaml',
                'particle_filter_map_file': 'cave.png',
                'level': '0'
            }.items()
        ),
        laser_scan_features_node,
        localization_node,
        planner_node,
        goto_node
    ]


def generate_launch_description():

    return LaunchDescription([
        DeclareLaunchArgument(
            'localization',
            default_value='pf',
            description='Use ekf or pf'),
        DeclareLaunchArgument(
            'params_file',
            default_value=TextSubstitution(text='params.yaml'),
            description='ROS2 parameters'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(
                get_package_share_directory('stage_ros2'), 'launch'),
                '/stage.launch.py'])
        ),
        OpaqueFunction(function=launch_setup)
    ])
