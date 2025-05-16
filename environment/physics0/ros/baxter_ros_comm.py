import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose2D

def read_pose():
    """Reads x, y, theta from pose.txt"""
    try:
        with open("environment/physics0/images/pose.txt", "r") as f:
            data = f.readline().strip().split(',')
            return float(data[0]), float(data[1]), float(data[2])
    except Exception as e:
        rospy.logerr("Error reading file: %s", e)
        return 0.0, 0.0, 0.0
    
def send_next_pose():

    pub = rospy.Publisher('/target_pose', Pose2D, queue_size=10)

    x, y, theta = read_pose()
    rospy.init_node('pc1_controller')

    pose = Pose2D()
    pose.x = x  # Example values
    pose.y = y
    pose.theta = theta
    pub.publish(pose)
    # Exit the publisher node after sending the message
    rospy.signal_shutdown("Message sent, shutting down.")

if __name__ == '__main__':
    try:
        send_next_pose()  # Start by sending first pose
    except rospy.ROSInterruptException:
        pass
