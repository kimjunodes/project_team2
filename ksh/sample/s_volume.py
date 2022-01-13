import cv2
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from distance import dist

volume_name = "volume"

def volume():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    cap = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    my_hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    while True:
        success, img = cap.read()
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = my_hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:    # 관절 사이 계산
                open = dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[14].x, handLms.landmark[14].y) < \
                    dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[16].x, handLms.landmark[16].y)
                if open == False:
                    curdist = -dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[8].x, handLms.landmark[8].y) / \
                            (dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[5].x, handLms.landmark[5].y) * 2)
                    curdist = curdist * 100
                    curdist = -96 - curdist
                    curdist = min(0, curdist)
                    volume.SetMasterVolumeLevel(curdist, None)
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                


        cv2.waitKey(10)
        cv2.imshow(volume_name, cv2.flip(img, 1))
        cv2.waitKey(10)

volume()