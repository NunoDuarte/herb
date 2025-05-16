#!/usr/bin/env python3
import rospy
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
import datetime

# Initialize global variables
bridge = CvBridge()
depth_image = None  
received_image = False  # Flag to track when we receive an image

def depth_image_callback(msg):
    """ Callback to capture one depth image and store it globally. """
    global depth_image, received_image
    try:
        # Convert ROS Image to OpenCV format
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        print(depth_image.mean())
        received_image = True  # Set flag to True after receiving one frame
        rospy.signal_shutdown("Received one image, exiting.")  # Shutdown ROS after getting one image
    except Exception as e:
        rospy.logerr(f"Error processing depth image: {e}")
        rospy.signal_shutdown("Error occurred, shutting down.")

if __name__ == "__main__":

    experiment = 'p88_s1'

    min_sec = datetime.datetime.now().strftime("%d%H%M%S")

    rospy.init_node("cropped_depth_image", anonymous=True)

    # Subscribe to depth topic (only capture one frame)
    rospy.Subscriber("/camera/depth/image_rect_raw", Image, depth_image_callback)

    # Wait for the callback to receive the image, then shutdown
    rospy.spin()

    # After ROS shuts down, plot the image in the main thread
    if received_image and depth_image is not None:
        height, width = depth_image.shape  

        # Define crop size (adjust as needed)
        crop_width = int(width * 0.2)  
        crop_height = int(height * 0.33)  

        # Compute center coordinates
        center_x , center_y = width // 2, height // 2

        # Compute crop boundaries
        x1, x2 = max(0, center_x - crop_width // 2), min(width, center_x + crop_width // 2)
        y1, y2 = max(0, center_y - crop_height // 2), min(height, center_y + crop_height // 2)

        # Crop the image
        cropped_depth = depth_image[y1-5:y2-5, x1-22:x2-25]

        # Find the minimum depth value (ignore zero values)
        max_depth = np.max(cropped_depth)  

        # Convert to height map (shift the lowest value to 0)
        height_map = max_depth - cropped_depth # Convert mm to meters

        np.save('/home/packbot/gojko-irbpp/environment/physics0/images/heightmap.npy', height_map)
        np.save('/home/packbot/gojko-irbpp/environment/physics0/images/heightmap_' + experiment + '_' + min_sec + '.npy', height_map)

        # Plot the cropped depth image (in the main thread)
        plt.figure(figsize=(8, 6))
        plt.imshow(height_map/1000, cmap="hot")
        cbar = plt.colorbar(label="Height (m)")  # Set colorbar label in meters
        plt.title("Normalized Height Map (Meters)")
        #plt.show()  # Blocking call, runs in the main thread

        # Save the figure as a PNG file
        plt.savefig("pack_height.png", dpi=300, bbox_inches="tight")
        plt.close()  # Close the figure to free memory

    else:
        print("No depth image received.")
