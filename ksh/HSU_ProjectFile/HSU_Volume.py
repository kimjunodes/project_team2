import cv2
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()                  # 오디오 받아오기
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume)) # 오디오 크기 조절

cap = cv2.VideoCapture(0)   # 캠

mpHands = mp.solutions.hands
my_hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


def dist(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1-x2, 2)) + math.sqrt(math.pow(y1-y2, 2))

def sound():
    volume.SetMasterVolumeLevel(curdist, None)

with mpHands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

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


                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)



        cv2.imshow("Volume", img)
        cv2.waitKey(1)


