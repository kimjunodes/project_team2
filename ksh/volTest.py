import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
import streamlit as st
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


################################
wCam, hCam = 640, 480
################################
st.title("Webcam Test")


cap = cv2.VideoCapture(0) # , cv2.CAP_DSHOW
FRAME_WINDOW = st.image([])
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(mode=False, maxHands=1, detectionCon=0.7, trackCon=0.5)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))


volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0
area = 0
colorVol = (255, 0, 0)


while True:
    success, img = cap.read()
    
    # 손
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True)
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

    # 볼륨조절창
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    # 프레임
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(cimg)
