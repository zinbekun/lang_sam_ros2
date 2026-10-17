#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # パラメータファイルのパスを取得
    lang_sam_executor_dir = get_package_share_directory('lang_sam_executor')
    params_file = os.path.join(lang_sam_executor_dir, 'config', 'params.yaml')

    # USBカメラノードの起動設定（use_usb_cam:=false で無効化）
    use_usb_cam_arg = DeclareLaunchArgument('use_usb_cam', default_value='true')
    usb_cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        output='screen',
        parameters=[params_file],
        condition=IfCondition(LaunchConfiguration('use_usb_cam'))
    )

    # LangSAM Trackerノードの起動設定
    lang_sam_tracker_node = Node(
        package='lang_sam_tracker',
        executable='lang_sam_tracker_node.py',
        name='lang_sam_tracker',
        output='screen',
        parameters=[params_file]
    )

    # LangSAM Person Followingノードの起動設定
    lang_sam_person_following_node = Node(
        package='lang_sam_person_following',
        executable='lang_sam_person_following_node',
        name='lang_sam_person_following',
        output='screen',
        parameters=[params_file]
    )

    return LaunchDescription([
        use_usb_cam_arg,
        usb_cam_node,
        lang_sam_tracker_node,
        lang_sam_person_following_node
    ])