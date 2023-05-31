---
layout: post
title:  "ROS Launch File"
date:   2023-05-31
desc: "ROS Launch File"
keywords: "Ros"
categories: [Ros]
tags: [Ros]
icon: icon-html
---


# ROS Learning Notes 
## _ROS Learning Notes 11_

[![image](https://static.wixstatic.com/media/3d5aae_1d1644f45a584ba7ac275771a4e00981~mv2.png/v1/fill/w_347,h_181,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/ros_feat.png)](https://www.bilibili.com/video/BV1zt411G7Vn?p=12&vd_source=d8d0bffc8e5266c19ad61d5b6c71609e)

> **_Keypoints_**
- Launch file syntax
- Examples

## launch file syntax
- launch file is a xml file that can launch multiple nodes at the same time.
- launch file can start ros master automatically.
- Example:
    ```xml
    <launch>
        <node pkg="turtlesim" type="turtlesim_node" name="turtlesim_node"/>
        <node pkg="turtlesim" type="turtle_teleop_key" name="turtle_teleop_key"/>
    </launch>
    ```
    - `pkg` is the package name
    - `type` is the node name
    - `name` is the name when the node is launched
    - `output` is the output type of the node to the terminal
    - `respawn` is to restart the node when it is killed
    - `required` is to kill all the nodes when one of them is killed
    - `ns` is to add a namespace to the node
    - `args` is to add arguments to the node

- Parameter: 
    - `<param>` is to set the parameter of one node. Stored in ROS server.
    - `<rosparam>` is to set the parameter of multiple nodes. Stored in ROS server.
      - `<param name="output_frame" value="odom"/>`
        - name: parameter name
        - value: parameter value
      - `<rosparam file="$(find my_package)/config/my_params.yaml" command="load" ns="params"/>`
        - file: the path of the parameter file
        - command: the command to load the parameter file
          - load: load the parameter file
          - dump: dump the parameter file
          - delete: delete the parameter file
        - ns: the namespace of the parameter file
    - `arg` is to set the argument of the launch file, only used in this launch file.
      - `<arg name="output_frame" default="arg-value"/>`
        - name: argument name
        - default: argument value
      - call the argument in the launch file:
        - `<param name="output_frame" value="$(arg output_frame)"/>` value must begin with `$(arg`. This value is decided by `default` in `<arg>`
        - `<node name="node" pkg="package" type="type" output="screen" respawn="true" required="true" ns="namespace" args="$(arg output_frame)"/>` 

- `<remap>` is to remap the ROS computation graph resource. Just to **rename** the topic name.
  - `<remap from="old_topic" to="new_topic"/>`
    - from: the old topic name
    - to: the new topic name
  
- `<include>` is to include another launch file, like C language.
  - `<include file="$(find my_package)/launch/my_launch_file.launch"/>`
    - file: the path of the launch file

## Examples
- simple launch file
  ```xml
  <launch>
    <node pkg="topic_learning" type="person_publisher" name="talker" output="screen"/>
    <node pkg="topic_learning" type="person_subscriber" name="listener" output="screen"/>
  </launch>
  ```

- turtlesim_parameter_config.launch
    ```xml
    <launch>
        <param name="/turtle_number" value="2"/>
        <node pkg="turtlesim" type="turtlesim_node" name="turtlesim_node">
            <param name="turtle_name1" value="turtle1"/>
            <param name="turtle_name2" value="turtle2"/>
            <rosparam file="$(find learning_launch)/config/param.yaml" command="load" />
        </node>

        <node pkg="turtlesim" type="turtle_teleop_key" name="turtle_teleop_key" output="screen"/>
    </launch>
    ```

- start_tf_demo_c++.launch
    ```xml
    <launch>
        <!-- Tuetlesim Node -->
        <node pkg="turtlesim" type="turtlesim_node" name="turtlesim_node"/>
        <node pkg="turtlesim" type="turtle_teleop_key" name="turtle_teleop_key" output="screen"/>

        <node pkg="learning_tf" type="turtle_tf_broadcaster" args="/turtle1" name="turtle1_tf_broadcaster"/>
        <node pkg="learning_tf" type="turtle_tf_broadcaster" args="/turtle2" name="turtle2_tf_broadcaster"/>

        <node pkg="learning_tf" type="turtle_tf_listener" name="turtle_tf_listener"/>
    </launch>
    ```

- start_tf_demo_python.launch
    ```xml
    <launch>
        <!-- Tuetlesim Node -->
        <node pkg="turtlesim" type="turtlesim_node" name="turtlesim_node"/>
        <node pkg="turtlesim" type="turtle_teleop_key" name="turtle_teleop_key" output="screen"/>

        <node name="turtle1_tf_broadcaster" pkg="learning_tf" type="turtle_tf_broadcaster.py" >
            <param name="turtle" type="string" value="turtle1"/>
        </node>

        <node name="turtle2_tf_broadcaster" pkg="learning_tf" type="turtle_tf_broadcaster.py" >
            <param name="turtle" type="string" value="turtle2"/>
        </node>

        <node pkg="learning_tf" type="turtle_tf_listener.py" name="turtle_tf_listener"/>

    </launch>
    ```

- turtlesim_remap.launch
    ```xml
    <launch>
        <include file="$(find learning_launch)/launch/simple.launch"/>

        <node pkg="turtlesim" type="turtlesim_node" name="turtlesim_node">
            <remap from="turtle1/cmd_vel" to="cmd_vel"/>
        </node>
    </launch>
    ```