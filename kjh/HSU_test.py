# sign_test

# import
import cv2
import mediapipe as mp
import math
import time
import pyautogui
import os
from cvzone.HandTrackingModule import HandDetector
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

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

def spider():
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
                    if gesture[i][5] == '1':
                        pyautogui.moveTo(gumx * screenWidth, gumy * screenHeight)
                    elif gesture[i][5] == '2':
                        pyautogui.mouseDown()
                        pyautogui.mouseUp()
                        time.sleep(1)
                    elif gesture[i][5] == 'spider':
                        pyautogui.press('left')
                        time.sleep(1)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        # reversal
        fimg = cv2.flip(img, 1)

        # show img(cam)
        cv2.imshow("spider", fimg)
        cv2.waitKey(1)
        if open == [False,False,False,False,True]:
            cv2.destroyAllWindows()
            return

def vol():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
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
                if open == False:
                    curdist = -dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[8].x,
                                    handLms.landmark[8].y) / \
                              (dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[5].x,
                                    handLms.landmark[5].y) * 2)
                    curdist = curdist * 100
                    curdist = -96 - curdist
                    curdist = min(0, curdist)
                    volume.SetMasterVolumeLevel(curdist, None)
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        cv2.imshow("vol_control", img)
        cv2.waitKey(1)
        if dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[17].x,
                            handLms.landmark[17].y) < \
                       dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[20].x,
                            handLms.landmark[20].y):
            cv2.destroyAllWindows()
            return

def video():
    LENGTH_THRESHOLD = 50
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    dir = os.getcwd()
    rp_dir = dir.replace('\\', '/')
    cap_video = cv2.VideoCapture("{0}/test.{1}".format(rp_dir, 'mp4' or 'avi' or 'mkv'))
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

            if hands:

                lm_list = hands[0]['lmList']
                fingers = detector.fingersUp(hands[0])

                length, info, cam_img = detector.findDistance(lm_list[4], lm_list[8], cam_img)  # 엄지, 검지를 이용하여 계산

                if fingers == [0, 0, 0, 0, 1]:  # 정지
                    cv2.destroyAllWindows()
                    return
                else:  # Play
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
            fimg = cv2.flip(cam_img, 1)
            cv2.imshow('video', video_img)
            cv2.imshow('cam', fimg)
            cv2.waitKey(1)

# distance
def dist(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1 - x2, 2)) + math.sqrt(math.pow(y1 - y2, 2))

# finger
compareIndex = [[5, 4], [6, 8], [10, 12], [14, 16], [18, 20]]

# thumb, index_finger, middle_finger, ring_figer, pinky (엄지 검지 중지 약지 소지)
open = [False, False, False, False, False]
gesture = [[False, True, False, False, False, '1'],
           [False, True, True, False, False, '2'],
           [True, True, True, False, False, '3'],
           [False, True, True, True, True, '4'],
           [True, True, True, True, True, '5'],
           [True, True, False, False, True, 'spider']]

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
                if gesture[i][5] == '1':
                    time.sleep(2)
                    cap.release()
                    cv2.destroyAllWindows()
                    spider()
                    cap = cv2.VideoCapture(0)
                elif gesture[i][5] == '2':
                    time.sleep(2)
                    cap.release()
                    cv2.destroyAllWindows()
                    vol()
                    cap = cv2.VideoCapture(0)
                elif gesture[i][5] == '3':
                    time.sleep(2)
                    cap.release()
                    cv2.destroyAllWindows()
                    video()
                    cap = cv2.VideoCapture(0)

        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # reversal
    fimg = cv2.flip(img, 1)

    # show img(cam)
    cv2.waitKey(1)
    cv2.imshow("MAIN", fimg)