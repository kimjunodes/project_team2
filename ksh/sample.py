import cv2
import streamlit as st
import mediapipe as mp
import pyautogui
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from distance import dist

st.title("Webcam Test")

rad = st.sidebar.radio("Solution",["A", "B", "C"])
FRAME_WINDOW = st.image([])

if rad == "A":
    st.write("A")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    cam = cv2.VideoCapture(0)

    devices = AudioUtilities.GetSpeakers()  # 오디오 받아오기
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    video_name = "video"
    max_num_hands = 1

    # cam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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

        # ret, frame = cam.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(img)

    else:
        st.write('Stopped')

if rad == "B":
    st.write("B")

if rad == "C":
    st.write("C")