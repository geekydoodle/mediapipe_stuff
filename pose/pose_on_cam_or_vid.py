######## Pose Estimation Realtime using Mediapipe #########

# Author: George
# Youtube Channel : https://www.youtube.com/channel/UC-j647KCqjQKE50Npx9TZsg

# Description: 
# This program uses a library medipipe to perform Pose Estimation Detection.
# This Model Detetcts 33 different landmarks and also draw lines between them.

# Importing the libraries.

import time
import cv2
import mediapipe as mp

# Drawing utility makes it easy for us to render all the different landmarks on our pose.

mpdrawing = mp.solutions.drawing_utils

# Pulling out the pose model from mediapipe.

mppose= mp.solutions.pose

# Set the height and width.

height = 600
width = 900

# The Pose class's constructor
# has some optional parameters like
# static_image_mode,upper_body_only,
# smooth_landmarks,min_detection_confidence
# and min_tracking_confidence.

pose = mppose.Pose()

# Setting up the camara or video.

cam = cv2.VideoCapture(0)

# cam = cv2.VideoCapture('/home/george/george/coding_place/python/projects/pose/files/video.mp4')

# Initializing previous frame time.

prev_frame_time = 0

while cam.isOpened():
    
    # Reading a frame from the Camara.

    ret, frame = cam.read()

    # For video rotating

    # frame = cv2.rotate(frame, cv2.ROTATE_180) 

    # Converting BGR to RGB.

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resizeing the rgb img.

    resize = cv2.resize(rgb, (width, height))

    # Processing the RGB image and returning the landmarks of our pose.

    results = pose.process(resize)

    # Converting RGB to BGR.

    image = cv2.cvtColor(resize, cv2.COLOR_RGB2BGR)

    # Execute the code under the if statement
    # if something is there in pose_landmarks.

    if results.pose_landmarks:
            
        # Draw the landmarks and conections.

        mpdrawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mppose.POSE_CONNECTIONS, 
                
            # Changing color of the landmarks.
                
            mpdrawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                
            # Changing color of the connections.
                
            mpdrawing.DrawingSpec(color=(0,255,0), thickness=2))
            
    # Fps end time.
    
    current_frame_time = time.time()
    
    # Final fps.
    
    fps = 1 / (current_frame_time - prev_frame_time)
    
    # Setting current frame time to previous frame time.
    
    prev_frame_time = current_frame_time
    
    # Showing fps.
    
    cv2.putText(image, "FPS : " + str(int(fps)), (15, 44), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 214, 36), 2)
    
    # Show the final result.

    cv2.imshow('Pose', image)

    # If Esc button is clicked break out from the loop.

    if cv2.waitKey(10) & 0xFF == 27:
        break

# Releaseing the camara.

cam.release()

# Destroying all the windows.

cv2.destroyAllWindows()