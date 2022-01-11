# sign.py

# import
import cv2
import mediapipe as mp
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import sign
from distance import dist  # distance.py


prev_fun_name = "prev"
volume_name = "volume"

def prev_fun():
    # variable
    max_num_hands = 1

    # cam
    cap = cv2.VideoCapture(0)

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    
    # mediapipe soulutins
    mpHands = mp.solutions.hands
    my_hands = mpHands.Hands(max_num_hands=max_num_hands,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils

    # finger
    compareIndex = [[10, 4], [6, 8], [10, 12], [14, 16], [18, 20]]

    # thumb, index_finger, middle_finger, ring_figer, pinky (엄지 검지 중지 약지 소지)
    open = [False, False, False, False, False]
    gesture = [[False, False, False, False, True, 'Back']]

    # angle
    while True:
        success, img = cap.read()
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = my_hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for i in range(0, 5):
                    open[i] = dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[compareIndex[i][0]].x,
                                   handLms.landmark[compareIndex[i][0]].y) < \
                              dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[compareIndex[i][1]].x,
                                   handLms.landmark[compareIndex[i][1]].y)

                    open_f = dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[14].x,
                                handLms.landmark[14].y) < \
                           dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[16].x,
                                handLms.landmark[16].y)

                    if open_f == False:
                        curdist = -dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[8].x,
                                        handLms.landmark[8].y) / \
                                  (dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[5].x,
                                        handLms.landmark[5].y) * 2)
                        curdist = curdist * 100
                        curdist = -96 - curdist
                        curdist = min(0, curdist)
                        volume.SetMasterVolumeLevel(curdist, None)
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            print(open)
            text_x = (handLms.landmark[0].x * w)
            text_y = (handLms.landmark[0].y * h)
            for i in range(0, len(gesture)):
                flag = True
                for j in range(0, 5):
                    if (gesture[i][j] != open[j]):
                        flag = False

                if (flag == True):
                    # font
                    cv2.putText(img, gesture[i][5], (round(text_x) - 50, round(text_y) - 250),
                                cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        # reversal
        fimg = cv2.flip(img, 1)

        # show img(cam)
        cv2.imshow(volume_name, fimg)

        # gesture for motion
        if open == [False, False, False, False, True]:
            cv2.destroyWindow(volume_name)
            sign.sign_fun()

        exit
        if cv2.waitKey(1) == ord('q'):
            break
