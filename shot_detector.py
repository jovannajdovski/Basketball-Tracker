import math
import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import sys

def score(ball_pos, hoop_pos):
    x = []
    y = []
    rim_height = hoop_pos[-1][0][1] - 0.5 * hoop_pos[-1][3]

    for i in reversed(range(len(ball_pos)-1)):
        if ball_pos[i][0][1] < rim_height:
            x.append(ball_pos[i][0][0])
            y.append(ball_pos[i][0][1])
            x.append(ball_pos[i+1][0][0])
            y.append(ball_pos[i+1][0][1])
            break

    if len(x) > 1:
        m, b = np.polyfit(x, y, 1)
        print(x, y)
        # Checks if projected line fits between the ends of the rim {x = (y-b)/m}
        predicted_x = ((hoop_pos[-1][0][1] - 0.5*hoop_pos[-1][3]) - b)/m
        rim_x1 = hoop_pos[-1][0][0] - 0.4 * hoop_pos[-1][2]
        rim_x2 = hoop_pos[-1][0][0] + 0.4 * hoop_pos[-1][2]
        if rim_x1 < predicted_x < rim_x2:
            return True


def detect_down(ball_pos, hoop_pos):
    y = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]
    if ball_pos[-1][0][1] > y:
        return True
    return False


def detect_up(ball_pos, hoop_pos):
    x1 = hoop_pos[-1][0][0] - 4 * hoop_pos[-1][2]
    x2 = hoop_pos[-1][0][0] + 4 * hoop_pos[-1][2]
    y1 = hoop_pos[-1][0][1] - 2 * hoop_pos[-1][3]
    y2 = hoop_pos[-1][0][1] - 0.5 * hoop_pos[-1][3]

    if x1 < ball_pos[-1][0][0] < x2 and y1 < ball_pos[-1][0][1] < y2:
        return True
    return False


def in_hoop_region(center, hoop_pos):
    if len(hoop_pos) < 1:
        return False
    x = center[0]
    y = center[1]

    x1 = hoop_pos[-1][0][0] - 1 * hoop_pos[-1][2]
    x2 = hoop_pos[-1][0][0] + 1 * hoop_pos[-1][2]
    y1 = hoop_pos[-1][0][1] - 1 * hoop_pos[-1][3]
    y2 = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]

    if x1 < x < x2 and y1 < y < y2:
        return True
    return False


def clean_ball_pos(ball_pos, frame_count):
    if len(ball_pos) > 1:
        w1 = ball_pos[-2][2]
        h1 = ball_pos[-2][3]
        w2 = ball_pos[-1][2]
        h2 = ball_pos[-1][3]

        x1 = ball_pos[-2][0][0]
        y1 = ball_pos[-2][0][1]
        x2 = ball_pos[-1][0][0]
        y2 = ball_pos[-1][0][1]

        f1 = ball_pos[-2][1]
        f2 = ball_pos[-1][1]
        f_dif = f2 - f1

        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        max_dist = 4 * math.sqrt((w1) ** 2 + (h1) ** 2)

        # Ball should not move a 4x its diameter within 5 frames
        if (dist > max_dist) and (f_dif < 5):
            ball_pos.pop()

        elif (w2*1.4 < h2) or (h2*1.4 < w2):
            ball_pos.pop()

    # Remove points older than 30 frames
    if len(ball_pos) > 0:
        if frame_count - ball_pos[0][1] > 30:
            ball_pos.pop(0)

    return ball_pos


def clean_hoop_pos(hoop_pos):
    if len(hoop_pos) > 1:
        x1 = hoop_pos[-2][0][0]
        y1 = hoop_pos[-2][0][1]
        x2 = hoop_pos[-1][0][0]
        y2 = hoop_pos[-1][0][1]

        w1 = hoop_pos[-2][2]
        h1 = hoop_pos[-2][3]
        w2 = hoop_pos[-1][2]
        h2 = hoop_pos[-1][3]

        f1 = hoop_pos[-2][1]
        f2 = hoop_pos[-1][1]

        f_dif = f2-f1

        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        max_dist = 0.5 * math.sqrt(w1 ** 2 + h1 ** 2)

        # Hoop should not move 0.5x its diameter within 5 frames
        if dist > max_dist and f_dif < 5:
            hoop_pos.pop()

        if (w2*1.3 < h2) or (h2*1.3 < w2):
            hoop_pos.pop()

    if len(hoop_pos) > 25:
        hoop_pos.pop(0)

    return hoop_pos


def run_shot_detector(video_path):
    # video_path = "./data/videos/test_video_7.mp4"
    model = YOLO("./models/best.pt")
    class_names = ['Basketball', 'Basketball Rim']
    cap = cv2.VideoCapture(video_path)

    ball_pos = []
    hoop_pos = []
    frame_count = 0
    makes = 0
    attempts = 0
    up = False
    down = False
    up_frame = 0
    down_frame = 0
    fade_frames = 20
    fade_counter = 0
    overlay_color = (0, 0, 0)

    while True:
        success, frame = cap.read()

        if not success:
            break

        # Tracker currently are not as good as detection frame by frame
        
        # results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
        # boxes = results[0].boxes
        # if boxes.id is not None:
        #     track_ids = results[0].boxes.id.int().tolist()

        # for box, track_id in zip(boxes, track_ids):
        #     x1, y1, x2, y2 = box.xyxy.tolist()[0]
        #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #     w, h = x2 - x1, y2 - y1

        #     # x1, y1, w, h = box.xywh.tolist()[0]
        #     # x1, y1, w, h = int(x1), int(y1), int(w), int(h)

        #     conf = math.ceil((box.conf.item() * 100)) / 100
        #     cls = int(box.cls.item())
        #     current_class = class_names[cls]
        #     center = (int(x1 + w / 2), int(y1 + h / 2))

        results = model(frame, stream=True, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                current_class = class_names[cls]
                center = (int(x1 + w / 2), int(y1 + h / 2))

                if (conf > .3 or (in_hoop_region(center, hoop_pos) and conf > 0.15)) and current_class == "Basketball":
                    ball_pos.append((center, frame_count, w, h, conf))
                    cvzone.cornerRect(frame, (x1, y1, w, h), colorR=(255,0,0), colorC=(255,0,0))

                if conf > .5 and current_class == "Basketball Rim":
                    hoop_pos.append((center, frame_count, w, h, conf))
                    cvzone.cornerRect(frame, (x1, y1, w, h))

        ball_pos = clean_ball_pos(ball_pos, frame_count)
        for i in range(len(ball_pos)):
            cv2.circle(frame, ball_pos[i][0], 2, (0, 0, 255), 2)

        if len(hoop_pos) > 1:
            hoop_pos = clean_hoop_pos(hoop_pos)
            cv2.circle(frame, hoop_pos[-1][0], 2, (128, 128, 0), 2)

        if len(hoop_pos) > 0 and len(ball_pos) > 0:
            if not up:
                up = detect_up(ball_pos, hoop_pos)
                if up:
                    up_frame = ball_pos[-1][1]

            if up and not down:
                down = detect_down(ball_pos, hoop_pos)
                if down:
                    down_frame = ball_pos[-1][1]

            if frame_count % 10 == 0:
                if up and down and up_frame < down_frame:
                    attempts += 1
                    up = False
                    down = False

                    if score(ball_pos, hoop_pos):
                        makes += 1
                        overlay_color = (0, 255, 0)
                        fade_counter = fade_frames
                    else:
                        overlay_color = (0, 0, 255)
                        fade_counter = fade_frames

        text = f"{makes} / {attempts}"
        cvzone.putTextRect(
            frame, text, (50, 100),
            scale=4, thickness=4,
            colorT=(255, 255, 255), colorR=(0, 0, 0),
            offset=12,
            border=3, colorB=(255, 255, 0)
        )

        if fade_counter > 0:
            alpha = 0.2 * (fade_counter / fade_frames)
            frame = cv2.addWeighted(frame, 1 - alpha, np.full_like(frame, overlay_color), alpha, 0)
            fade_counter -= 1

        cv2.imshow('Frame', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = sys.argv[1]
    run_shot_detector(video_path)