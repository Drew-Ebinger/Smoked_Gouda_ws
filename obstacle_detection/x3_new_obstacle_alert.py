
import math
import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformException, TransformListener


class NewObstacleAlert(Node):
    def __init__(self):
        super().__init__('new_obstacle_alert')

        self.radius = 1.0
        self.cost_thresh = 100
        self.min_cells = 4
        self.cooldown = 5.0
        self.ignore_radius = 0.25
        self.persist_cycles = 2
        self.clear_cycles = 3

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, map_qos)
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.costmap_cb, 10)

        self.tf = Buffer(cache_time=Duration(seconds=10.0))
        TransformListener(self.tf, self)

        self.map_msg = None
        self.map_data = None
        self.costmap_msg = None
        self.costmap_data = None

        self.hit_streak = 0
        self.clear_streak = 0
        self.alert_active = False
        self.last_alert = self.get_clock().now() - Duration(seconds=self.cooldown + 1)

        self.create_timer(0.5, self.check)
        self.get_logger().info('obstacle alert node started')

    def map_cb(self, msg):
        self.map_msg = msg
        self.map_data = np.array(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))

    def costmap_cb(self, msg):
        self.costmap_msg = msg
        self.costmap_data = np.array(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))

    def check(self):
        if self.map_msg is None or self.costmap_msg is None:
            return

        try:
            tf = self.tf.lookup_transform('map', 'base_footprint', rclpy.time.Time(),
                                          timeout=Duration(seconds=0.1))
        except TransformException:
            return

        rx = tf.transform.translation.x
        ry = tf.transform.translation.y

        count, nearest_dist, nearest_bearing = self.count_dynamic_cells(rx, ry)
        detected = count >= self.min_cells

        if detected:
            self.hit_streak += 1
            self.clear_streak = 0
        else:
            self.hit_streak = 0
            self.clear_streak += 1

        now = self.get_clock().now()
        cooldown_ok = (now - self.last_alert).nanoseconds / 1e9 >= self.cooldown

        if detected and not self.alert_active and self.hit_streak >= self.persist_cycles and cooldown_ok:
            self.alert_active = True
            self.last_alert = now
            self.get_logger().warn(
                f'NEW OBSTACLE DETECTED | cells={count} | dist={nearest_dist:.2f} m '
                f'| bearing={nearest_bearing:.1f} deg | radius={self.radius:.2f} m'
            )
        elif self.alert_active and self.clear_streak >= self.clear_cycles:
            self.alert_active = False
            self.get_logger().info('obstacle cleared')

    def count_dynamic_cells(self, rx, ry):
        cm = self.costmap_msg
        res = cm.info.resolution
        ox = cm.info.origin.position.x
        oy = cm.info.origin.position.y

        cx = int((rx - ox) / res)
        cy = int((ry - oy) / res)
        r = int(math.ceil(self.radius / res))

        count = 0
        nearest_dist = float('inf')
        nearest_bearing = 0.0

        for gy in range(max(0, cy - r), min(cm.info.height, cy + r + 1)):
            for gx in range(max(0, cx - r), min(cm.info.width, cx + r + 1)):
                wx = ox + (gx + 0.5) * res
                wy = oy + (gy + 0.5) * res
                dx = wx - rx
                dy = wy - ry
                dist = math.hypot(dx, dy)

                if dist > self.radius or dist < self.ignore_radius:
                    continue
                if self.costmap_data[gy, gx] < self.cost_thresh:
                    continue

                m = self.map_msg
                sx = int((wx - m.info.origin.position.x) / m.info.resolution)
                sy = int((wy - m.info.origin.position.y) / m.info.resolution)
                if sx < 0 or sy < 0 or sx >= m.info.width or sy >= m.info.height:
                    continue
                static_val = int(self.map_data[sy, sx])
                if static_val == -1 or static_val > 20:
                    continue

                count += 1
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_bearing = math.degrees(math.atan2(dy, dx))

        return count, nearest_dist, nearest_bearing


def main(args=None):
    rclpy.init(args=args)
    node = NewObstacleAlert()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

