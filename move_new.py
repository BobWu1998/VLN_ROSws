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
        rospy.Rate(10)
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=3)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.callback)
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

        return X, Y, Z 

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
        angular_speed = angle_speed*2*PI/360 # in radius
        relative_angle = angle*2*PI/360

        # We won't use linear components
        vel_msg.linear.x=0.
        vel_msg.linear.y=0.
        vel_msg.linear.z=0.
        vel_msg.angular.x = 0.
        vel_msg.angular.y = 0.

        # Checking if our movement is CW or CCW
        if clockwise:
            vel_msg.angular.z = -abs(angular_speed)
        else:
            vel_msg.angular.z = abs(angular_speed)
        
        # save the current angle
        current_angle = self.angle_z#copy.deepcopy(self.angle_z)

        while(abs(self.angle_z-current_angle)*2*PI/360 < relative_angle):
            self.velocity_publisher.publish(vel_msg)


        # Forcing our robot to stop
        vel_msg.angular.z = 0
        self.velocity_publisher.publish(vel_msg)
        rospy.spin()

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
        curr_x, curr_y = self.x, self.y #copy.deepcopy(self.x), copy.deepcopy(self.y)#self.x.copy(), self.y.copy()

        # Loop to move the turtle in an specified distance
        while(current_distance <= 0.25):
            current_distance = self.get_dist(curr_x, curr_y)
            #Publish the velocity
            self.velocity_publisher.publish(vel_msg)

        
        print("forward stops")
        # After the loop, stops the robot
        vel_msg = Twist()
        vel_msg.linear.x = 0
        # Force the robot to stop
        self.velocity_publisher.publish(vel_msg)
        rospy.sleep(10.)
        # print(vel_msg)
        # rospy.spin()
        

    def get_pose(self):
        # # stop any movement
        # vel_msg = Twist()
        # vel_msg.linear.x = 0
        # vel_msg.linear.y = 0
        # vel_msg.linear.z = 0
        # vel_msg.angular.x = 0
        # vel_msg.angular.y = 0
        # vel_msg.angular.z = 0
        # # Force the robot to stop
        # self.velocity_publisher.publish(vel_msg)

        # rospy.init_node('check_odometry')
        # odom_sub = rospy.Subscriber('/odom', Odometry, self.callback)
        print('current position:', self.x, self.y, self.z)
        print('current orientation:', self.angle_x, self.angle_y, self.angle_z)
        return self.x, self.y, self.z, self.angle_x, self.angle_y, self.angle_z


if __name__ == '__main__':
    try:
        robot_ins = robot()
        
        cmd_list = input("Your cmds:")
        
        for cmd in cmd_list:
            if cmd == '0':
                print('left')
                robot_ins.get_pose()
                robot_ins.rotate(clockwise=0)
                robot_ins.get_pose()
            elif cmd == '1':
                print('right')
                robot_ins.get_pose()
                robot_ins.rotate(clockwise=1)
                robot_ins.get_pose()
            elif cmd == '2':
                print('forward')
                robot_ins.get_pose()
                robot_ins.forward()
                robot_ins.get_pose()
                # rospy.spin()
                
            else:
                print('nothing happens')
        rospy.spin()

    except rospy.ROSInterruptException: pass
