# video.py
""" https://google.github.io/mediapipe/ """ # 참고 사이트

# import
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import cv2

def video():
    # variable
    LENGTH_THRESHOLD = 50
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    # cam
    cap_cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # video location
    cap_video = cv2.VideoCapture('D:\Pycharm\PyCharm Community Edition 2020.3.3\project_team2\ksh\sample_data\\03.mp4')

    # cam size
    cap_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 500)      
    cap_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)     
    w = int(cap_cam.get(cv2.CAP_PROP_FRAME_WIDTH))

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

        # hands soltions
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while cap_cam.isOpened():
            _, cam_img = cap_cam.read()
            cam_img = cv2.flip(cam_img, 1)

            hands, cam_img = detector.findHands(cam_img)

            if hands:
                lm_list = hands[0]['lmList']
                fingers = detector.fingersUp(hands[0])

                # distance between thumb and index finger
                length, info, cam_img = detector.findDistance(lm_list[4], lm_list[8], cam_img) 

                if fingers == [0, 0, 0, 0, 0]: # stop
                    pass
                else: # start
                    if length < LENGTH_THRESHOLD: # Navigate
                        rel_x = lm_list[4][0] / w
                        frame_idx = int(rel_x * total_frames)
                        frame_idx = min(max(frame_idx, 0), total_frames)

                        cap_video.set(1, frame_idx)
                    else:
                        frame_idx += 1
                        rel_x = frame_idx / total_frames

                    _, video_img = cap_video.read()
                    draw_timeline(video_img, rel_x)

            # show img(cam)
            cv2.imshow('video', video_img)
            cv2.imshow('cam', cam_img)
            if cv2.waitKey(1) == ord('q'):
                break