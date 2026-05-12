import cv2
import json
import math
import time
import threading
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, HTTPServer

import mediapipe as mp


HOST = "0.0.0.0"
PORT = 5000

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

GESTURE_MESSAGES = {
    "DRINK WATER": "Patient needs to drink water",
    "BATHROOM": "Patient needs to go to bathroom",
    "ASSISTANCE": "Patient needs assistance",
    "PAIN": "Patient may be in pain",
    "READY": "Patient is ready / all good",
    "DISCOMFORT": "Patient is uncomfortable",
    "OK": "Patient is OK / thank you",
    "NO HAND": "No hand detected",
    "UNKNOWN": "Gesture not recognized",
}

latest_cv_data = {
    "cv_alert": False,
    "gesture": "NO HAND",
    "message": GESTURE_MESSAGES["NO HAND"],
    "confidence": 0.0,
    "hand_detected": False,
    "timestamp": time.strftime("%H:%M:%S"),
}

latest_lock = threading.Lock()


def distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y


def finger_folded(lm, tip, pip):
    return lm[tip].y > lm[pip].y


def thumb_is_up(lm):
    thumb_tip = lm[4]
    thumb_ip = lm[3]
    thumb_mcp = lm[2]
    return thumb_tip.y < thumb_ip.y and thumb_tip.y < thumb_mcp.y


def thumb_is_down(lm):
    wrist = lm[0]
    thumb_tip = lm[4]
    thumb_mcp = lm[2]
    return thumb_tip.y > wrist.y and thumb_tip.y > thumb_mcp.y


def detect_gesture(lm):
    thumb_tip = lm[4]
    wrist = lm[0]

    index_up = finger_up(lm, 8, 6)
    middle_up = finger_up(lm, 12, 10)
    ring_up = finger_up(lm, 16, 14)
    pinky_up = finger_up(lm, 20, 18)

    index_folded = finger_folded(lm, 8, 6)
    middle_folded = finger_folded(lm, 12, 10)
    ring_folded = finger_folded(lm, 16, 14)
    pinky_folded = finger_folded(lm, 20, 18)

    folded_count = sum([index_folded, middle_folded, ring_folded, pinky_folded])
    up_count = sum([index_up, middle_up, ring_up, pinky_up])

    ok_distance = distance(thumb_tip, lm[8])
    palm_size = distance(wrist, lm[9])

    if thumb_is_down(lm) and folded_count >= 3:
        return "ASSISTANCE"

    if index_up and not middle_up and not ring_up and not pinky_up:
        return "BATHROOM"

    if index_up and middle_up and not ring_up and not pinky_up:
        return "READY"

    if up_count >= 4:
        return "PAIN"

    if ok_distance < palm_size * 0.28 and middle_up and ring_up and pinky_up:
        return "OK"

    if folded_count >= 4 and thumb_is_up(lm):
        return "DRINK WATER"

    if folded_count >= 4:
        return "DISCOMFORT"

    return "UNKNOWN"


def gesture_color_bgr(gesture):
    if gesture in ["ASSISTANCE", "PAIN"]:
        return (0, 0, 255)
    if gesture in ["DRINK WATER", "BATHROOM", "DISCOMFORT"]:
        return (0, 255, 255)
    if gesture in ["READY", "OK"]:
        return (0, 255, 0)
    return (180, 180, 180)


def gesture_is_alert(gesture):
    return gesture in [
        "DRINK WATER",
        "BATHROOM",
        "ASSISTANCE",
        "PAIN",
        "DISCOMFORT",
    ]


def update_latest_data(gesture, confidence, hand_detected):
    payload = {
        "cv_alert": gesture_is_alert(gesture),
        "gesture": gesture,
        "message": GESTURE_MESSAGES.get(gesture, GESTURE_MESSAGES["UNKNOWN"]),
        "confidence": round(float(confidence), 2),
        "hand_detected": bool(hand_detected),
        "timestamp": time.strftime("%H:%M:%S"),
    }

    with latest_lock:
        latest_cv_data.update(payload)


def draw_panel(frame, gesture, fps, confidence, hand_detected):
    h, w, _ = frame.shape
    panel_x = int(w * 0.68)

    cv2.rectangle(frame, (panel_x, 0), (w, h), (22, 22, 22), -1)
    cv2.putText(
        frame,
        "GESTURE ALERTS",
        (panel_x + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    y = 82
    visible_gestures = [
        "DRINK WATER",
        "BATHROOM",
        "ASSISTANCE",
        "PAIN",
        "READY",
        "DISCOMFORT",
        "OK",
    ]

    for item in visible_gestures:
        color = gesture_color_bgr(item)
        active = item == gesture
        thickness = 2 if active else 1
        rect_color = color if active else (90, 90, 90)

        cv2.rectangle(frame, (panel_x + 12, y - 24), (w - 12, y + 38), rect_color, thickness)
        cv2.putText(
            frame,
            item,
            (panel_x + 25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color if active else (190, 190, 190),
            2,
        )
        cv2.putText(
            frame,
            GESTURE_MESSAGES[item],
            (panel_x + 25, y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (230, 230, 230),
            1,
        )
        y += 70

    cv2.rectangle(frame, (0, h - 50), (w, h), (15, 15, 15), -1)
    status_text = "Detected" if hand_detected else "Not detected"
    status_color = (0, 255, 0) if hand_detected else (0, 0, 255)

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, "Hand:", (140, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, status_text, (200, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (345, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(frame, f"API: http://LAPTOP_IP:{PORT}/cv-data", (555, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)


class CvDataHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ["/cv-data", "/cv-data/"]:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Use /cv-data")
            return

        with latest_lock:
            body = json.dumps(latest_cv_data).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


def start_http_server():
    server = HTTPServer((HOST, PORT), CvDataHandler)
    print(f"CV API server running at http://{HOST}:{PORT}/cv-data")
    server.serve_forever()


def main():
    server_thread = threading.Thread(target=start_http_server, daemon=True)
    server_thread.start()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not detected. Try changing VideoCapture(0) to VideoCapture(1).")
        return

    gesture_history = deque(maxlen=8)
    prev_time = time.time()

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.70,
        min_tracking_confidence=0.70,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read camera frame.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            hand_detected = False
            raw_gesture = "NO HAND"
            confidence = 0.0

            if results.multi_hand_landmarks:
                hand_detected = True
                hand_landmarks = results.multi_hand_landmarks[0]
                lm = hand_landmarks.landmark

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                raw_gesture = detect_gesture(lm)

                if results.multi_handedness:
                    confidence = results.multi_handedness[0].classification[0].score
                else:
                    confidence = 0.80

                x_values = [p.x for p in lm]
                y_values = [p.y for p in lm]
                x1 = int(min(x_values) * w) - 20
                y1 = int(min(y_values) * h) - 20
                x2 = int(max(x_values) * w) + 20
                y2 = int(max(y_values) * h) + 20
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            gesture_history.append(raw_gesture)
            stable_gesture = Counter(gesture_history).most_common(1)[0][0]

            current_time = time.time()
            fps = 1 / max(current_time - prev_time, 0.001)
            prev_time = current_time

            message = GESTURE_MESSAGES.get(stable_gesture, GESTURE_MESSAGES["UNKNOWN"])
            main_color = gesture_color_bgr(stable_gesture)

            update_latest_data(stable_gesture, confidence, hand_detected)

            cv2.putText(frame, f"Gesture: {stable_gesture}", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, main_color, 3)
            cv2.putText(frame, f"Status: {message}", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
            cv2.putText(frame, time.strftime("Time: %H:%M:%S"), (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

            draw_panel(frame, stable_gesture, fps, confidence, hand_detected)
            cv2.imshow("Smart Bed - Patient Gesture Monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()