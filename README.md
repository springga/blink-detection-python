# Blink detection with dlib

### Introduction
![blink detection animation](blink_detection_animation.gif)
(source: [pyimagesearch](https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/))

Thanks to the powerful [dlib](https://github.com/davisking/dlib), robust and efficient blink detection is achieved within 100 lines of code. Comparing to my other repo [blink-detection-js](https://github.com/springga/blink-detection-js), dlib handles dark condition and fast blinks quite reliably.

### Prerequisition
* [Python3](https://www.python.org/)
* [dlib](https://github.com/davisking/dlib)
* [opencv](https://github.com/opencv/opencv)

### Install dlib on Windows
Most google results for installing dlib on Windows require Visual Studio and CMake.

But I found a simple whl file can avoid all the pain.

Download [dlib-19.8.1-cp36-cp36m-win_amd64.whl](https://pypi.org/project/dlib/19.8.1/#files) and simply run:

`pip install dlib-19.8.1-cp36-cp36m-win_amd64.whl`

### Get Started
Download [dlib facical landmarks model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) (unzip to the project folder)

Run:

`python blink_detection.py`

### Reference
https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/

