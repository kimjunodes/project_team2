# mouse.py

# import
import cv2
import mediapipe as mp
import math
import pyautogui
import time
from distance import dist
# from s_sign import smain

mouse_name = "mouse"
max_num_hands = 1

def mouse():
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
    gesture = [[False, True, False, False, False, '1']]

    # performance
    while True:
        success, img = cam.read()
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = my_hands.process(imgRGB)

        # cursor location
        curmX, curmY = pyautogui.position()
        print("Cursor : {0}, {1}".format(curmX, curmY))

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:  # 관절 사이 계산
                gumx = handLms.landmark[8].x
                gumy = handLms.landmark[8].y
                for i in range(0,5):
                    open[i] = dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[compareIndex[i][0]].x, handLms.landmark[compareIndex[i][0]].y) < \
                    dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[compareIndex[i][1]].x, handLms.landmark[compareIndex[i][1]].y)

                # location index finger
                print("Index finger: {0}, {1}".format(gumx, gumy))
                openm = dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[14].x,
                            handLms.landmark[14].y) < \
                       dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[16].x,
                            handLms.landmark[16].y)

                # move to cursor
                if openm == False:
                    pyautogui.moveTo(gumx * screenWidth, gumy * screenHeight)
                    if dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[10].x,
                            handLms.landmark[10].y) < dist(handLms.landmark[0].x, handLms.landmark[0].y,
                                                           handLms.landmark[12].x, handLms.landmark[12].y):
                        pyautogui.mouseDown()
                        pyautogui.mouseUp()
                        time.sleep(1)
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

        # show img(cam)
        cv2.imshow(mouse_name, img)
        cv2.waitKey(1)

        if open == [False,False,False,False,True]:
            cv2.destroyWindow(mouse_name)
            # smain()
            break

        # exit
        if cv2.waitKey(1) == ord('q'):
            break

        
mouse()