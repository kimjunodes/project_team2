# video.py

from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import cv2
import numpy as np
import os
from distance import dist

video_name = "video"

def draw_timeline(img, rel_x):
        img_h, img_w, _ = img.shape
        timeline_w = max(int(img_w * rel_x) - 50, 50)
        cv2.rectangle(img, pt1=(50, img_h - 50), pt2=(timeline_w, img_h - 49), color=(0, 0, 255), thickness=-1)

def video():
    k = 0
    q = list(0 for i in range(20))
    qc = list(0 for i in range(20))

    reset = np.zeros(20)

    LENGTH_THRESHOLD = 50
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    dir = os.getcwd()
    rp_dir = dir.replace('\\', '/')
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap_video = cv2.VideoCapture("{0}/test.{1}".format(rp_dir, 'mp4' or 'avi' or 'mkv'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    total_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
    _, video_img = cap_video.read()

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
                    if k<20:
                        q[k] = '1'
                        print(q)
                        k += 1
                    else:
                        for h in range(19):
                            qc[h] = q[h+1]
                        qc[19] = '1'
                        q = qc
                        print(q)
                else:  # Play
                    if k<20:
                        q[k] = '0'
                        print(q)
                        k += 1
                    else:
                        for h in range(19):
                            qc[h] = q[h+1]
                        qc[19] = '0'
                        q = qc
                        print(q)

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
            cv2.imshow(video_name, video_img)
            cv2.imshow('cam', fimg)
            cv2.waitKey(1)

            if q.count('1') == 20:
                cv2.destroyAllWindows()
                return
