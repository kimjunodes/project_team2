# sign_test

# import
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import sys
from cvzone.HandTrackingModule import HandDetector
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from distance import dist
from mouse import mouse
from video import video
from vol import vol

def screen():
    devices = AudioUtilities.GetSpeakers()  # 오디오 받아오기
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    screenWidth, screenHeight = pyautogui.size()
    pyautogui.FAILSAFE = False

    # variable
    max_num_hands = 1

    # cam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # mediapipe soulutins
    mpHands = mp.solutions.hands
    my_hands = mpHands.Hands(max_num_hands=max_num_hands,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils


    # finger
    compareIndex = [[5, 4], [6, 8], [10, 12], [14, 16], [18, 20]]

    # thumb, index_finger, middle_finger, ring_figer, pinky (엄지 검지 중지 약지 소지)
    open = [False, False, False, False, False]
    gesture = [[False, True, False, False, False, '1'],
            [False, True, True, False, False, '2'],
            [True, True, True, False, False, '3'],
            [False, True, True, True, True, '4'],
            [True, True, True, True, True, '5'],
            [True, True, False, False, True, 'mouse'],
            [False, False, False, False, True, 'quit']
            ]

    k = 0

    q = list(0 for i in range(20))
    qc = list(0 for i in range(20))
    reset = np.zeros(20)
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

            for i in range(0, len(gesture)):
                flag = True
                for j in range(0, 5):
                    if (gesture[i][j] != open[j]):
                        flag = False
                if (flag == True):
                    ges = gesture[i][5]

                    if k < 20:
                        q[k] = ges
                        print(q)
                        k += 1
                    else:
                        for h in range(19):
                            qc[h] = q[h + 1]
                        qc[19] = ges
                        q = qc
                        print(q)

                    if q.count('1') == 20:
                        cap.release()
                        cv2.destroyAllWindows()
                        q = reset
                        mouse()
                        cap = cv2.VideoCapture(0)
                    elif q.count('2') == 20:
                        cap.release()
                        cv2.destroyAllWindows()
                        q = reset
                        vol()
                        cap = cv2.VideoCapture(0)
                    elif q.count('3') == 20:
                        cap.release()
                        cv2.destroyAllWindows()
                        q = reset
                        video()
                        cap = cv2.VideoCapture(0)

                    elif q.count('quit') == 20:
                        sys.exit(0)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        # reversal
        fimg = cv2.flip(img, 1)

        # show img(cam)
        cv2.waitKey(1)
        cv2.imshow("MAIN", fimg)