#mouse.py

# import
import cv2
import mediapipe as mp
import math
import pyautogui
import time
import numpy as np
from distance import dist

mouse_name = "mouse"
max_num_hands = 1

# cam
cam = cv2.VideoCapture(0)

# screen size
screenWidth, screenHeight = pyautogui.size()
pyautogui.FAILSAFE = False

# mediapipe soultions
mpHands = mp.solutions.hands
my_hands = mpHands.Hands(max_num_hands = max_num_hands,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5)
mpDraw = mp.solutions.drawing_utils

compareIndex = [[10, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
open = [False, False, False, False, False]
gesture = [[False, True, False, False, False, '1'],
            [False, False, False, False, True, 'quit']]

k = 0

q = list(0 for i in range(20))
qc = list(0 for i in range(20))
reset = np.zeros(20)

cap = cv2.VideoCapture(0)
while True:


    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = my_hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            gumx = handLms.landmark[8].x
            gumy = handLms.landmark[8].y
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

                if gesture[i][5] == '1':
                    pyautogui.moveTo(gumx * screenWidth, gumy * screenHeight)
                elif gesture[i][5] == '2':
                    pyautogui.mouseDown()
                    pyautogui.mouseUp()
                    time.sleep(1)
                elif gesture[i][5] == 'mouse':
                    pyautogui.press('left')
                    time.sleep(1)

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

        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # reversal
    fimg = cv2.flip(img, 1)

    # show img(cam)
    cv2.imshow("mouse", fimg)
    cv2.waitKey(1)

    if q.count('quit') == 20:
        cv2.destroyWindow("mouse")