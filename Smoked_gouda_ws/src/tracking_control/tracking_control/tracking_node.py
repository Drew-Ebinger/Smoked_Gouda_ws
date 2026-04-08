import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import TransformException, Buffer, TransformListener
import numpy as np
import math

## Functions for quaternion and rotation matrix conversion
## The code is adapted from the general_robotics_toolbox package
## Code reference: https://github.com/rpiRobotics/rpi_general_robotics_toolbox_py
def hat(k):
    """
    Returns a 3 x 3 cross product matrix for a 3 x 1 vector

             [  0 -k3  k2]
     khat =  [ k3   0 -k1]
             [-k2  k1   0]

    :type    k: numpy.array
    :param   k: 3 x 1 vector
    :rtype:  numpy.array
    :return: the 3 x 3 cross product matrix
    """
    khat = np.zeros((3, 3))
    khat[0, 1] = -k[2]
    khat[0, 2] = k[1]
    khat[1, 0] = k[2]
    khat[1, 2] = -k[0]
    khat[2, 0] = -k[1]
    khat[2, 1] = k[0]
    return khat


def q2R(q):
    """
    Converts a quaternion into a 3 x 3 rotation matrix according to the
    Euler-Rodrigues formula.

    :type    q: numpy.array
    :param   q: 4 x 1 vector representation of a quaternion q = [q0;qv]
    :rtype:  numpy.array
    :return: the 3x3 rotation matrix
    """
    I = np.identity(3)
    qhat = hat(q[1:4])
    qhat2 = qhat.dot(qhat)
    return I + 2 * q[0] * qhat + 2 * qhat2


def euler_from_quaternion(q):
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - z * x))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return [roll, pitch, yaw]


class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        self.get_logger().info('Tracking Node Started')

        # Current object pose
        self.obs_pose = None
        self.goal_pose = None
        self.home_pose = None
        self.state = 'to_goal'
        self.spin_start = None
        self.spin_duration = 6.28

        # ROS parameters
        self.declare_parameter('world_frame_id', 'odom')

        # Create a transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Create publisher for the control command
        self.pub_control_cmd = self.create_publisher(Twist, '/track_cmd_vel', 10)
        # Create subscribers to the detected object poses
        self.sub_detected_goal_pose = self.create_subscription(
            PoseStamped, 'detected_color_object_pose', self.detected_obs_pose_callback, 10)
        self.sub_detected_obs_pose = self.create_subscription(
            PoseStamped, 'detected_color_goal_pose', self.detected_goal_pose_callback, 10)

        # Create timer, running at 100Hz
        self.timer = self.create_timer(0.01, self.timer_update)

    def detected_obs_pose_callback(self, msg):
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

        if np.linalg.norm(center_points[:2]) > 3 or abs(center_points[2]) > 0.7:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                odom_id, msg.header.frame_id, rclpy.time.Time(), rclpy.duration.Duration(seconds=0.1))
            t_R = q2R(np.array([
                transform.transform.rotation.w,
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z
            ]))
            cp_world = t_R @ center_points + np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
        except TransformException as e:
            self.get_logger().error('Transform Error: {}'.format(e))
            return

        self.obs_pose = cp_world

    def detected_goal_pose_callback(self, msg):
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

        if np.linalg.norm(center_points[:2]) > 3 or abs(center_points[2]) > 0.7:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                odom_id, msg.header.frame_id, rclpy.time.Time(), rclpy.duration.Duration(seconds=0.1))
            t_R = q2R(np.array([
                transform.transform.rotation.w,
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z
            ]))
            cp_world = t_R @ center_points + np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
        except TransformException as e:
            self.get_logger().error('Transform Error: {}'.format(e))
            return

        self.goal_pose = cp_world

    def get_current_poses(self):
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_footprint', odom_id, rclpy.time.Time())
            robot_world_x = transform.transform.translation.x
            robot_world_y = transform.transform.translation.y
            robot_world_z = transform.transform.translation.z
            robot_world_R = q2R(np.array([
                transform.transform.rotation.w,
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z
            ]))

            obstacle_pose = None
            goal_pose = None

            if self.obs_pose is not None:
                obstacle_pose = robot_world_R @ self.obs_pose + np.array([
                    robot_world_x, robot_world_y, robot_world_z])

            if self.goal_pose is not None:
                goal_pose = robot_world_R @ self.goal_pose + np.array([
                    robot_world_x, robot_world_y, robot_world_z])

        except TransformException as e:
            self.get_logger().error('Transform error: ' + str(e))
            return None, None

        return obstacle_pose, goal_pose

    def timer_update(self):
        ################### Write your code here ###################
        # Save the starting pose once, in the world frame
        if self.home_pose is None:
            tf = self.tf_buffer.lookup_transform(
                self.get_parameter('world_frame_id').value,
                'base_footprint',
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            )
            p = tf.transform.translation
            self.home_pose = np.array([p.x, p.y, p.z])

        # Final state: stop
        if self.state == 'done':
            self.pub_control_cmd.publish(Twist())
            return

        current_obs_pose, current_goal_pose = self.get_current_poses()

        if self.state == 'to_goal':
            if current_goal_pose is None:
                self.pub_control_cmd.publish(Twist())
                #print("Going to goal!")
                return

            target_pose = current_goal_pose

            if np.linalg.norm(target_pose[:2]) < 0.18:
                # Start spin if not already spinning
                if self.spin_start is None:
                    self.spin_start = self.get_clock().now()
                    elapsed = (self.get_clock().now() - self.spin_start).nanoseconds / 1e9
                    print("Starting to Spin")

                if elapsed < self.spin_duration:
                    cmd = Twist()
                    cmd.angular.z = 0.5
                    self.pub_control_cmd.publish(cmd)
                    print("Spinning!")
                    return
                else:
                    self.state = 'returning_home'
                    self.pub_control_cmd.publish(Twist())
                    return

        elif self.state == 'returning_home':
            odom_id = self.get_parameter('world_frame_id').value

            tf = self.tf_buffer.lookup_transform(
                'base_footprint',
                odom_id,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            )
            t_R = q2R(np.array([
                tf.transform.rotation.w,
                tf.transform.rotation.x,
                tf.transform.rotation.y,
                tf.transform.rotation.z
            ]))
            t = np.array([
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z
            ])
            target_pose = t_R @ self.home_pose + t


            print("Going home!")
            return

            # Check if within 0.18 meters to home
        if np.linalg.norm(target_pose[:2]) < 0.18:
            self.state = 'done'
            self.pub_control_cmd.publish(Twist())
            print("Home!")
            return

        cmd_vel = self.controller(target_pose, current_obs_pose)
        self.pub_control_cmd.publish(cmd_vel)
        ################################################

   
    def controller(self, target_pose, obstacle_pose):
        ## Setup ##
        k_attract = 0.55
        k_repel = 0.22
        k_tangent = 0.12
        obs_rad_influence = 0.35
        obs_hard_stop = 0.15
        max_speed = 0.10

        F_repel = np.zeros(2)
        F_tangent = np.zeros(2)
        F_attract = np.zeros(2)

        ## Attraction ##
        F_attract[0] = k_attract * target_pose[0]
        F_attract[1] = k_attract * target_pose[1]

        ## Repulsion + tangential avoidance ##
        if obstacle_pose is not None:
            obs_xy = np.array([obstacle_pose[0], obstacle_pose[1]])
            d_obs = np.linalg.norm(obs_xy)

            if d_obs > 1e-6:
                away_from_obs = -obs_xy / d_obs

                tangent_dir = np.array([-away_from_obs[1], away_from_obs[0]])

                if d_obs < obs_hard_stop:
                    F_repel = 0.35 * away_from_obs
                    F_tangent = 0.20 * tangent_dir

                elif d_obs < obs_rad_influence:
                    mag = k_repel * (1.0 / d_obs - 1.0 / obs_rad_influence) / (d_obs ** 2)
                    F_repel = mag * away_from_obs

                    tangent_mag = k_tangent * (obs_rad_influence - d_obs) / obs_rad_influence
                    F_tangent = tangent_mag * tangent_dir

        ## Combination ##
        F_total = F_attract + F_repel + F_tangent

        ## Limit speed ##
        speed = np.linalg.norm(F_total)
        if speed > max_speed:
            F_total = (F_total / speed) * max_speed

        ## Translate to velocity command ##
        cmd_vel = Twist()
        cmd_vel.linear.x = float(F_total[0])
        cmd_vel.linear.y = float(F_total[1])
        cmd_vel.angular.z = 0.0

        return cmd_vel
        ############################################


def main(args=None):
    rclpy.init(args=args)
    tracking_node = TrackingNode()
    rclpy.spin(tracking_node)
    tracking_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
