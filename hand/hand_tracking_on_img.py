######## Hand Landmark Detection using Mediapipe #########

# Author: George
# Youtube Channel : https://www.youtube.com/channel/UC-j647KCqjQKE50Npx9TZsg

# Description: 
# This program uses a library called medipipe to perform Hand Landmark Detection.
# This Model Detetcts 21 Hand landmarks and also draw lines between them.

# Importing the libraries.

import cv2
import mediapipe as mp

# Put the number of hands you want.

numberhand = 2

# Drawing utility makes it easy for us to render all the different landmarks on our hands.

mpdrawing = mp.solutions.drawing_utils

# Pulling out the hands model from mediapipe.

mphands = mp.solutions.hands

# Set the height and width.

height = 600
width = 900

# The Hands class's constructor
# has some optional parameters like
# static_image_mode,max_num_hands,
# min_detection_confidence
# and min_tracking_confidence.

hands = mphands.Hands(max_num_hands=numberhand)

# Setting up img.

frame = cv2.imread('img.jpg')

# Converting BGR to RGB.

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Resizeing the rgb img.

resize = cv2.resize(rgb, (width, height))

# Processing the RGB image and returning the landmarks of our hands.

results = hands.process(resize)

# Converting RGB to BGR.

image = cv2.cvtColor(resize, cv2.COLOR_RGB2BGR)

# Execute the code under the if statement
# if something is there in results.multi_hand_landmarks.

if results.multi_hand_landmarks:

    # Pulling each hand from the list.

    for hand in results.multi_hand_landmarks:
            
        # Draw the landmarks and conections.

        mpdrawing.draw_landmarks(
            image,
            hand,
            mphands.HAND_CONNECTIONS, 
                
            # Changing color of the landmarks.
                
            mpdrawing.DrawingSpec(color = (0,0,255), thickness = 2, circle_radius = 2),
                
            # Changing color of the connections.
                
            mpdrawing.DrawingSpec(color = (0,255,0), thickness = 2))
    
# Show the final result.

cv2.imshow('Hands', image)

# Wait key

cv2.waitKey(0)

# Destroying all the windows.

cv2.destroyAllWindows()
