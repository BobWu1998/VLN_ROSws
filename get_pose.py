#! /usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from move_around import forward, rotate
import time
# def callback(msg):
#     print(msg.pose.pose.position)


# rospy.init_node('check_odometry')
# odom_sub = rospy.Subscriber('/odom', Odometry, callback)
# rospy.spin()



import numpy as np

class pose_obj:
    def __init__(self):
        # rospy.init_node('check_odometry')
        odom_sub = rospy.Subscriber('/odom', Odometry, self.callback)
        self.x, self.y, self.z = 0., 0., 0.
        self.angle_x, self.angle_y, self.angle_z = 0, 0, 0

    def quaternion_to_euler_angle(self, w, x, y, z):
        ysqr = y * y

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        X = np.degrees(np.arctan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = np.where(t2>+1.0,+1.0,t2)
        #t2 = +1.0 if t2 > +1.0 else t2

        t2 = np.where(t2<-1.0, -1.0, t2)
        #t2 = -1.0 if t2 < -1.0 else t2
        Y = np.degrees(np.arcsin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        Z = np.degrees(np.arctan2(t3, t4))

        return X, Y, Z 

    def callback(self, msg):
        print(msg.pose.pose.position)
        angle_x, angle_y, angle_z = self.quaternion_to_euler_angle(msg.pose.pose.orientation.w,\
                                                            msg.pose.pose.orientation.x,\
                                                            msg.pose.pose.orientation.y,\
                                                            msg.pose.pose.orientation.z)
        # print('True')
        # print("orientation:")
        # print("x:", angle_x)
        # print("y:", angle_y)
        # print("z:", angle_z)
        self.x = msg.pose.pose.position.x
        print(self.x)

    def get_pose(self):
        # rospy.init_node('check_odometry')
        # odom_sub = rospy.Subscriber('/odom', Odometry, self.callback)
        print('current_pose', self.x)
        return self.x


if __name__ == '__main__':
    # try:
    pose_ins = pose_obj()
    time.sleep(0.5)
    pose_ins.get_pose()

    # forward()
    # rotate(clockwise=0)
    # rotate(clockwise=0)
    pose_ins.get_pose()

    forward()
    forward()
    forward()
    pose_ins.get_pose()
    try:
        rospy.spin()
    except rospy.ROSInterruptException: pass
