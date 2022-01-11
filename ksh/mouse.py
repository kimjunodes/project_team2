# mouse.py

# import
import cv2
import mediapipe as mp
import math
import pyautogui
import time
from distance import dist

mouse_name = "mouse"

def mouse():
    # cam
    cam = cv2.VideoCapture(0)

    # screen size
    screenWidth, screenHeight = pyautogui.size()

    pyautogui.FAILSAFE = False

    # mediapipe soultions
    mpHands = mp.solutions.hands
    my_hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    # distance
    

    # performance
    while True:
        success, img = cam.read()
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = my_hands.process(imgRGB)

        # cursor location
        curmX, curmY = pyautogui.position()
        print("Cursor : {0}, {1}".format(curmX,curmY))

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:  # 관절 사이 계산
                gumx = handLms.landmark[8].x
                gumy = handLms.landmark[8].y

                # location index finger
                print("Index finger: {0}, {1}".format(gumx,gumy))
                open = dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[14].x, handLms.landmark[14].y) < \
                    dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[16].x, handLms.landmark[16].y)

                # move to cursor
                if open == False:
                    pyautogui.moveTo(gumx * screenWidth, gumy * screenHeight)
                    if dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[10].x, handLms.landmark[10].y) < dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[12].x, handLms.landmark[12].y):
                        pyautogui.mouseDown()
                        pyautogui.mouseUp()
                        time.sleep(1)
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        # show img(cam)
        cv2.imshow(mouse_name, img)
        cv2.waitKey(1)

        #prev_fun
        # prev_fun()

        # exit
        if cv2.waitKey(1) == ord('q'):
                break