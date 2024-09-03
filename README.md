# Project Overview

**Objective:** To detect and track players, referees, and footballs in video footage using advanced AI techniques.

## Key Components

### Object Detection and Tracking
- Utilize YOLO (You Only Look Once), a state-of-the-art object detection model, to identify and track players, referees, and footballs in video frames.
- Train the YOLO model to enhance detection accuracy and performance.

### Player Team Assignment
- Implement K-means clustering for pixel segmentation to differentiate players based on the colors of their t-shirts.
- Use the segmented data to assign players to their respective teams.

### Ball Acquisition Measurement
- Calculate each team’s ball acquisition percentage throughout the match using the player and ball tracking data.

### Camera Movement Analysis
- Apply optical flow techniques to analyze and measure camera movement between frames for accurate player movement tracking.

### Perspective Transformation
- Implement perspective transformation to accurately represent the scene’s depth and perspective.
- Convert player movement measurements from pixels to meters for a more realistic assessment.

### Speed and Distance Calculation
- Determine players' speed and the distance covered based on the transformed perspective and movement data.

## Requirements
To run this project, you need to have the following requirements installed:
- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas

