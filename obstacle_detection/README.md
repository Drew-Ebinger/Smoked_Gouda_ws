Quick notes for mapping, navigation, and running the obstacle alert node.


1) FIRST TIME ONLY: CREATE AND SAVE MAP
Run these in separate terminals:

Terminal 1:
ros2 launch yahboomcar_nav map_gmapping_launch.py

Terminal 2:
ros2 launch yahboomcar_nav display_map_launch.py

Terminal 3:
ros2 run yahboomcar_ctrl yahboom_keyboard

Drive the car around until the map looks good in RViz.

Then save the map:
ros2 launch yahboomcar_nav save_map_launch.py

2) NORMAL RUN: NAVIGATION + ALERT NODE

Use this after the map has already been made.

Terminal 1:
ros2 launch yahboomcar_nav yahboomcar_nav_launch.py

Terminal 2:
ros2 launch yahboomcar_nav laser_bringup_launch.py

Terminal 3:
ros2 launch yahboomcar_nav navigation_dwa_launch.py

Terminal 4:
python3 ~/ros2_ws/src/my_nodes/x3_new_obstacle_alert.py (or whereever you store it)

3) RVIZ STEPS

In RViz:
1. Set Durability Policy to Transient Local to get saved map to show
2. Click 2D Pose Estimate
3. Set the robot's starting pose on the map
4. Send a 2D Goal Pose or waypoints
