#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import copy
PI = 3.1415926535897

SPEED = 0.2
ANGLE_SPEED = 30 # degree/s

class robot:
    def __init__(self):
        rospy.init_node('roomba', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        rospy.sleep(1)
        self.x, self.y, self.z = 0., 0., 0.
        self.angle_x, self.angle_y, self.angle_z = 0., 0., 0.
    
    def get_dist(self, init_x, init_y):
        return np.sqrt((self.x-init_x)**2 + (self.y-init_y)**2)

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

        # return X, Y, Z 
        return np.radians(X), np.radians(Y), np.radians(Z)

    def callback(self, msg):
        # print(msg.pose.pose.position)
        angle_x, angle_y, angle_z = self.quaternion_to_euler_angle(msg.pose.pose.orientation.w,\
                                                            msg.pose.pose.orientation.x,\
                                                            msg.pose.pose.orientation.y,\
                                                            msg.pose.pose.orientation.z)
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z
        self.angle_x, self.angle_y, self.angle_z = angle_x, angle_y, angle_z

    def rotate(self, angle_speed=ANGLE_SPEED, clockwise=0):
        
        vel_msg = Twist()

        angle = 15 # degrees

        # Converting from angles to radians
        angular_speed = ANGLE_SPEED*2*PI/360 # in radius
        relative_angle = angle*2*PI/360

        # We won't use linear components
        vel_msg.linear.x=0
        vel_msg.linear.y=0
        vel_msg.linear.z=0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0

        # Checking if our movement is CW or CCW
        if clockwise:
            vel_msg.angular.z = -abs(angular_speed)
        else:
            vel_msg.angular.z = abs(angular_speed)
        # Setting the current time for distance calculus
        t0 = rospy.Time.now().to_sec()
        current_angle = 0

        while(abs(current_angle) < relative_angle):
            self.velocity_publisher.publish(vel_msg)
            t1 = rospy.Time.now().to_sec()
            current_angle = angular_speed*(t1-t0)


        # Forcing our robot to stop
        vel_msg.angular.z = 0
        self.velocity_publisher.publish(vel_msg)
        # rospy.spin()

    def forward(self, speed=SPEED):

        vel_msg = Twist()
        vel_msg.linear.x = abs(speed)

        #Since we are moving just in x-axis
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0


        # Setting the current time for distance calculus
        t0 = rospy.Time.now().to_sec()
        current_distance = 0

        # Loop to move the turtle in an specified distance
        while(current_distance <= 0.25):
                #Publish the velocity
            self.velocity_publisher.publish(vel_msg)
                #Takes actual time to velocity calculus
            t1=rospy.Time.now().to_sec()
                #Calculates distancePoseStamped
            current_distance= speed*(t1-t0)
            # print('dist moved:', current_distance)
        # After the loop, stops the robot
        vel_msg.linear.x = 0
        # Force the robot to stop
        self.velocity_publisher.publish(vel_msg)

        

    def get_pose(self):
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.callback)
        print('current position:', self.x, self.y, self.z)
        print('current orientation:', self.angle_x, self.angle_y, self.angle_z)
        return self.x, self.y, self.z, self.angle_x, self.angle_y, self.angle_z


if __name__ == '__main__':
    try:
        robot_ins = robot()
        while True:
            cmd_list = input("Your cmds:")
            # robot_ins.forward()
            for cmd in cmd_list:
                if cmd == '0':
                    print('left')
                    robot_ins.rotate(clockwise=0)
                elif cmd == '1':
                    print('right')
                    robot_ins.rotate(clockwise=1)
                elif cmd == '2':
                    print('forward')
                    robot_ins.forward()
                    # rospy.spin()
                elif cmd == '3':
                    robot_ins.get_pose()
                else:
                    print('nothing happens')
        # rospy.spin()

    except rospy.ROSInterruptException: pass
