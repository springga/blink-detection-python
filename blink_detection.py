import dlib
import cv2
import numpy as np
from scipy.spatial import distance


# face and facial landmarks detectors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# eye landmarks id of dlib output
left_eye_start, left_eye_end = 42, 48
right_eye_start, right_eye_end = 36, 42

# eye landmarks distribution
#   1  2
# 0      3
#   5  4
# measure eye opening with eye landmarks
def eye_opening(eye):
    vertical1 = distance.euclidean(eye[1], eye[5])
    vertical2 = distance.euclidean(eye[2], eye[4]) 
    horizontal = distance.euclidean(eye[0], eye[3])
    opening = (vertical1 + vertical2) / horizontal

    return opening

# convert dlib output to array
def landmarks_to_array(output):
    coords = []
    for i in range(output.num_parts):
        coords.append((output.part(i).x, output.part(i).y))

    return coords

# count blinks
THRESH = 0.5
eye_closed = False
blinks = 0

# start camera and counting
try:
    # get default camera
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        # convert to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # locate faces
        faces = detector(gray)

        for f in faces:
            # draw bounding box around face
            cv2.rectangle(frame, (f.left(), f.top()), (f.right(), f.bottom()), (255,0,0), 2)
            # detect facial landmarks
            face_landmarks = landmarks_to_array(predictor(gray, f))            
            # calculate eye opening
            left_eye = face_landmarks[left_eye_start:left_eye_end]
            right_eye = face_landmarks[right_eye_start:right_eye_end]
            left_opening = eye_opening(left_eye)
            right_opening = eye_opening(right_eye)
            opening = (left_opening + right_opening) / 2.0
            if opening <= THRESH:
                eye_closed = True
            elif eye_closed:
                blinks += 1
                eye_closed = False
                
            # visualize eyes shape
            left_eye_hull = cv2.convexHull(np.array(left_eye))
            right_eye_hull = cv2.convexHull(np.array(right_eye))
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
            
            # put measurement on video
            cv2.putText(frame, "Blinks: {}".format(blinks), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Opening: {:.2f}".format(opening), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # show result
        cv2.imshow('Video', frame)
        # press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # clean exit
    video_capture.release()
    cv2.destroyAllWindows()

# reference: https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
