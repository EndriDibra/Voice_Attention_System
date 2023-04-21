# Author: Endri Dibra

# importing the required libraries
import cv2
import dlib
import imutils
from imutils import face_utils
from gtts import gTTS
from playsound import playsound
from pygame import mixer
from scipy.spatial import distance


# downloading sound
mixer.init()
mixer.music.load("sound.wav")


# calculating eye ratio in order to detect eye's direction
def eye_ratio(eye):

    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])

    ear = (a + b) / (2.0 * c)

    return ear


# Initializing the type of audio and language
audio = 'speech.mp3'
language = "en"

# voice will speak if eye's direction is down
sp = gTTS(text="Attention, look straight!", lang=language, slow=False)
sp.save(audio)

# fixed numbers until the alarm sounds
thresh = 0.25
time_wall = 35

# detecting face and eyes
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# taking camera from device
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# this will be used as a time counter, until it surpass
# the fixed number for sound alarm
counter = 0

while True:

    # reading camera
    success, frame = camera.read()
    frame = imutils.resize(frame, width=450)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for s in subjects:

        # detecting and drawing eyes
        shape = predict(gray, s)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_ratio(leftEye)
        rightEAR = eye_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftEyeHull], -1, (255, 0, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 255), 1)

        # if eyes are down the counter will be increased
        if ear < thresh:

            counter += 1

            # if counter surpass the fixed number
            # alarm will be activated with voice machine
            if counter >= time_wall:

                mixer.music.play()
                playsound(audio)

        # re-initializing to zero if counter is
        # not equal or greater than time_wall
        else:

            counter = 0

    # displaying camera
    cv2.imshow("Camera", frame)

    # "t" goes for "terminate"
    if cv2.waitKey(1) & 0xFF == ord("t"):
        break

# terminating the program
cv2.destroyAllWindows()
camera.release()