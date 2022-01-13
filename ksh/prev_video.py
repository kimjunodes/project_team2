# video.py
""" https://google.github.io/mediapipe/ """  # 참고 사이트

# import
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import cv2
from distance import dist
from s_sign import smain

video_name = "video"


def video():
    # variable
    LENGTH_THRESHOLD = 50
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    # cam
    cap_cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # video location
    cap_video = cv2.VideoCapture('D:\Pycharm\PyCharm Community Edition 2020.3.3\HSU_BIT_Project\sample_data\\03.mp4')

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

    compareIndex = [[10, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
    open = [False, False, False, False, False]
    gesture = [[False, True, False, False, False, '1']]

    # hands soltions
    mp_hands = mp.solutions.hands
    mpDraw = mp.solutions.drawing_utils
    with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        while cap_cam.isOpened():
            _, cam_img = cap_cam.read()
            cam_img = cv2.flip(cam_img, 1)

            hands, cam_img = detector.findHands(cam_img)

            success, img = cap_cam.read()
            h, w, c = img.shape
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(imgRGB)

            if hands:
                lm_list = hands[0]['lmList']
                fingers = detector.fingersUp(hands[0])

                # distance between thumb and index finger
                length, info, cam_img = detector.findDistance(lm_list[4], lm_list[8], cam_img)

                if fingers == [0, 0, 0, 0, 0]:  # stop
                    pass
                else:  # start
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
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    for i in range(0, 5):
                        open[i] = dist(handLms.landmark[0].x, handLms.landmark[0].y,
                                       handLms.landmark[compareIndex[i][0]].x, handLms.landmark[compareIndex[i][0]].y) < \
                                  dist(handLms.landmark[0].x, handLms.landmark[0].y,
                                       handLms.landmark[compareIndex[i][1]].x, handLms.landmark[compareIndex[i][1]].y)
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
                mpDraw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            if open == [False, True, False, False, False]:
                cv2.destroyWindow(video_name)
                smain()
            # show img(cam)
            cv2.imshow('vieo', video_img)
            cv2.imshow(video_name, cam_img)

            if cv2.waitKey(1) == ord('q'):
                break


video()