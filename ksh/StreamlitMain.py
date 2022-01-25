import cv2
import pyautogui
import time
import streamlit as st
import mediapipe as mp
import numpy as np
import tempfile
import tkinter as tk
import HandTrackingModule as htm
from cvzone.HandTrackingModule import HandDetector
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from distance import dist

max_num_hands = 1
k = 0
q = list(0 for i in range(30))
qc = list(0 for i in range(30))
reset = np.zeros(30)

fn = ''

st.title("Webcam Test")

rad = st.sidebar.radio("Solution", ["MAIN", "Explain", "etc"])
FRAME_WINDOW = st.image([])

mpHands = mp.solutions.hands
my_hands = mpHands.Hands(max_num_hands=max_num_hands,
                         min_detection_confidence=0.5,
                         min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

def alarm(text):
    root = tk.Tk()
    root.wm_overrideredirect(True)
    root.geometry("120x50+1+1")
    root.bind(root.destroy())

    l = tk.Label(text='', font=("Helvetica", 30))
    l.pack(expand=True)

    l.config(text=text, fg='green')
    root.mainloop()


def mouse():
    mouse_name = "mouse"
    st.write(mouse_name)

    k = 0
    q = list(0 for i in range(30))
    qc = list(0 for i in range(30))

    cap = cv2.VideoCapture(0)

    screenWidth, screenHeight = pyautogui.size()
    pyautogui.FAILSAFE = False

    compareIndex = [[10, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
    open = [False, False, False, False, False]
    gesture = [[False, True, False, False, False, '1'],
               [False, True, True, False, False, '2'],
               [False, True, False, False, True, 'mouse'],
               [False, False, False, False, True, 'quit']]
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

                    if k < 30:
                        q[k] = ges
                        k += 1
                    else:
                        for h in range(29):
                            qc[h] = q[h + 1]
                        qc[29] = ges
                        q = qc

                    if q.count('quit') == 30:
                        cap.release()
                        cv2.destroyAllWindows()
                        return

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fimg = cv2.flip(img, 1)
        FRAME_WINDOW.image(fimg)


def volume():
    vol_name = "volume"
    st.write(vol_name)

    k = 0
    q = list(0 for i in range(30))
    qc = list(0 for i in range(30))

    wCam, hCam = 640, 480
    

    pTime = 0
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, wCam)
    cap.set(4, hCam)

    detector = htm.handDetector(mode=False, maxHands=1, detectionCon=0.7, trackCon=0.5)
    

    devices = AudioUtilities.GetSpeakers()  # 오디오 받아오기
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volRange = volume.GetVolumeRange()
    minVol = volRange[0]
    maxVol = volRange[1]
    vol = 0
    volBar = 400
    volPer = 0
    area = 0
    colorVol = (255, 0, 0)

    pyautogui.FAILSAFE = False

    while True:
        success, img = cap.read()
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = my_hands.process(imgRGB)
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img, draw=True)


        
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:  # 관절 사이 계산

                if len(lmList) != 0:
                    # 손 사이즈
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
                    if 250 < area < 1000:
                    
                        # 엄지와 검지 사이
                        length, img, lineInfo = detector.findDistance(4, 8, img)

                        # 볼륨
                        volBar = np.interp(length, [50, 200], [400, 150])
                        volPer = np.interp(length, [50, 200], [0, 100])

                        # 볼륨 조절
                        smoothness = 10
                        volPer = smoothness * round(volPer / smoothness)

                        # 올리기
                        fingers = detector.fingersUp()

                        # 약지사용
                        if not fingers[2]:
                            volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                            cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                            colorVol = (0, 255, 0)
                        else:
                            colorVol = (255, 0, 0)

                if k < 30:
                    q[k] = str(bool(quit))
                    k += 1
                else:
                    for h in range(29):
                        qc[h] = q[h + 1]
                    qc[29] = str(bool(quit))
                    q = qc

                if q.count('True') == 30:
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        # 볼륨조절창
        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)
              


        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fimg = cv2.flip(img, 1)
        FRAME_WINDOW.image(fimg)

def video(fn):
    video_name = "video"
    st.write(video_name)

    k = 0

    q = list(0 for i in range(30))
    qc = list(0 for i in range(30))

    LENGTH_THRESHOLD = 50
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap_video = cv2.VideoCapture(fn)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    total_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
    _, video_img = cap_video.read()

    def draw_timeline(img, rel_x):
        img_h, img_w, _ = img.shape
        timeline_w = max(int(img_w * rel_x) - 50, 50)
        cv2.rectangle(img, pt1=(50, img_h - 50), pt2=(timeline_w, img_h - 49), color=(0, 0, 255), thickness=-1)

    rel_x = 0
    frame_idx = 0
    draw_timeline(video_img, 0)

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            _, cam_img = cap.read()
            cam_img = cv2.flip(cam_img, 1)

            hands, cam_img = detector.findHands(cam_img)

            if total_frames == frame_idx+1:
                cap.release()
                cv2.destroyAllWindows()
                return

            if hands:
                lm_list = hands[0]['lmList']
                fingers = detector.fingersUp(hands[0])
                length, info, cam_img = detector.findDistance(lm_list[4], lm_list[8], cam_img)  # 엄지, 검지를 이용하여 계산

                if fingers == [0, 0, 0, 0, 1]:  # 정지
                    if k<30:
                        q[k] = '1'
                        k += 1
                    else:
                        for h in range(29):
                            qc[h] = q[h+1]
                        qc[29] = '1'
                        q = qc
                    pass
                else:  # Play
                    if k<30:
                        q[k] = '0'
                        k += 1
                    else:
                        for h in range(29):
                            qc[h] = q[h+1]
                        qc[29] = '0'
                        q = qc

                    if length < LENGTH_THRESHOLD:  # Navigate
                        rel_x = lm_list[4][0] / w
                        frame_idx = int(rel_x * total_frames)
                        frame_idx = min(max(frame_idx, 0), total_frames)

                        cap_video.set(1, frame_idx)
                    else:
                        frame_idx += 1
                        rel_x = frame_idx / total_frames

                _, video_img = cap_video.read()
                draw_timeline(video_img, rel_x)

            if q.count('1') == 30:
                cap.release()
                cv2.destroyAllWindows()
                return

            video_cvt = cv2.cvtColor(video_img, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(video_cvt)

if rad == "MAIN":
    mouse_name = "MAIN"
    st.write(mouse_name)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    f = False

    uf = st.checkbox('Upload file')
    if uf == True:
        f = st.file_uploader("Upload file")
        if bool(f) == False:
            st.sidebar.write('등록된 동영상이 없습니다.')
        elif bool(f) == True:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(f.read())
            fn = tfile.name

    FRAME_WINDOW = st.image([])

    pyautogui.FAILSAFE = False

    compareIndex = [[5, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
    open = [False, False, False, False, False]
    gesture = [[False, True, False, False, False, '1'],
               [False, True, True, False, False, '2'],
               [True, True, True, False, False, '3'],
               [False, True, True, True, True, '4'],
               [True, True, True, True, True, '5'],
               [True, True, False, False, True, 'mouse'],
               [False, False, False, False, False, 'quit']]

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

                    if k < 30:
                        q[k] = ges
                        k += 1
                    else:
                        for h in range(29):
                            qc[h] = q[h + 1]
                        qc[29] = ges
                        q = qc

                    if q.count('1') == 30:
                        cap.release()
                        cv2.destroyAllWindows()
                        q = reset
                        mouse()
                        cap = cv2.VideoCapture(0)
                    elif q.count('2') == 30:
                        cap.release()
                        cv2.destroyAllWindows()
                        q = reset
                        volume()
                        cap = cv2.VideoCapture(0)

                    elif q.count('3') == 30 and bool(f) == False:
                        st.write("동영상 파일을 Upload 하셔야 됩니다.")
                        continue
                    elif q.count('3') == 30 and bool(f) == True:
                        cap.release()
                        # cv2.destroyAllWindows()
                        q = reset
                        video(fn)
                        cap = cv2.VideoCapture(0)

                    elif q.count('quit') == 30:
                        break
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fimg = cv2.flip(img, 1)
        FRAME_WINDOW.image(fimg)

#if rad == "Explain":
#if rad == "etc":
