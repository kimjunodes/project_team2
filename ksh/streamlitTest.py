import cv2
import pyautogui
import time
import streamlit as st
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from distance import dist


max_num_hands = 1
k = 0
q = list(0 for i in range(20))
qc = list(0 for i in range(20))
reset = np.zeros(20)

st.title("Webcam Test")

rad = st.sidebar.radio("Solution",["MAIN", "A", "B", "C"])
FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

mpHands = mp.solutions.hands
my_hands = mpHands.Hands(max_num_hands = max_num_hands,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5)
mpDraw = mp.solutions.drawing_utils
if rad == "MAIN":
    mouse_name = "MAIN"
    st.write(mouse_name)

    run = st.checkbox(mouse_name)
    FRAME_WINDOW = st.image([])

    screenWidth, screenHeight = pyautogui.size()
    pyautogui.FAILSAFE = False

    compareIndex = [[5, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
    open = [False, False, False, False, False]
    gesture = [[False, True, False, False, False, '1'],
            [False, True, True, False, False, '2'],
            [True, True, True, False, False, '3'],
            [False, True, True, True, True, '4'],
            [True, True, True, True, True, '5'],
            [True, True, False, False, True, 'mouse'],
            [False, False, False, False, True, 'quit']
            ]
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
                        # cv2.destroyAllWindows()
                        q = reset
                        print("mouse")
                        cap = cv2.VideoCapture(0)
                    elif q.count('2') == 20:
                        cap.release()
                        # cv2.destroyAllWindows()
                        q = reset
                        print("volum")
                        cap = cv2.VideoCapture(0)
                    elif q.count('3') == 20:
                        cap.release()
                        # cv2.destroyAllWindows()
                        q = reset
                        print("video")
                        cap = cv2.VideoCapture(0)

                    elif q.count('quit') == 20:
                        # sys.exit(0)
                        print("quit")
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fimg = cv2.flip(img, 1)
        FRAME_WINDOW.image(fimg)
if rad == "A":
    mouse_name = "mouse"
    st.write(mouse_name)

    run = st.checkbox(mouse_name)
    FRAME_WINDOW = st.image([])

    screenWidth, screenHeight = pyautogui.size()
    pyautogui.FAILSAFE = False

    compareIndex = [[10, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
    open = [False, False, False, False, False]
    gesture = [[False, True, False, False, False, '1'],
                [False, True, True, False, False, '2'],
                [False, False, False, False, True, 'quit']]
    while run:
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

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fimg = cv2.flip(img, 1)
        FRAME_WINDOW.image(fimg)
    else:
        st.write('Stopped')

if rad == "B":
    vol_name = "volume"
    st.write(vol_name)

    run = st.checkbox(vol_name)
    FRAME_WINDOW = st.image([])

    devices = AudioUtilities.GetSpeakers()  # 오디오 받아오기
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    
    screenWidth, screenHeight = pyautogui.size()
    pyautogui.FAILSAFE = False

    compareIndex = [[10, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
    open = [False, False, False, False, False]
    gesture = [[False, True, False, False, False, '1'],
            [False, False, False, False, True, 'quit']]
    while run:
        success, img = cap.read()
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = my_hands.process(imgRGB)


        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:  # 관절 사이 계산
                open = dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[14].x,
                            handLms.landmark[14].y) < \
                        dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[16].x,
                            handLms.landmark[16].y)

                quit = dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[17].x,
                            handLms.landmark[17].y) < \
                        dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[20].x,
                            handLms.landmark[20].y)

                if open == False:
                    curdist = -dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[8].x,
                                    handLms.landmark[8].y) / \
                                (dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[5].x,
                                    handLms.landmark[5].y) * 2)
                    curdist = curdist * 100
                    curdist = -96 - curdist
                    curdist = min(0, curdist)
                    if curdist > -60 and curdist <0:
                        volume.SetMasterVolumeLevel(curdist, None)
                else:
                    continue

                if k<20:
                    q[k] = str(bool(quit))
                    print(q)
                    k+=1
                else:
                    for h in range(19):
                        qc[h] = q[h+1]
                    qc[19] = str(bool(quit))
                    q = qc
                    print(q)

                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        else:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fimg = cv2.flip(img, 1)
        FRAME_WINDOW.image(fimg)
    else:
        st.write('Stopped')
if rad == "C":

    st.write("C")