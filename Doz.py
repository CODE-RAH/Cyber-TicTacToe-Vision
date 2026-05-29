"""
این کد توسط امیرفرخانی موسس کدراه پیاده سازی شده
همکار و اساتید (تقلبی) قبل از برداشتن سورس کد
و تموم کردن کار به اسم خودتون ذکر منبع کنید
در غیر اینصورت رضایتی در کار نیست و اگه متوجه بشیم عواقب داره

"""

import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)


num_rows = 3
num_cols = 3
drawing_size = 40

rectangle_x1, rectangle_y1 = 170, 100
rectangle_x2, rectangle_y2 = 470, 400

cell_width = (rectangle_x2 - rectangle_x1) // num_cols
cell_height = (rectangle_y2 - rectangle_y1) // num_rows


drawing_states = np.zeros((num_rows, num_cols), dtype=int)
turn = 1
winner = 0
win_line = None


button_x1, button_y1 = 220, 420
button_x2, button_y2 = 420, 470
button_text = "Play Again"

cap = cv2.VideoCapture(0)
cv2.namedWindow('Combined', cv2.WINDOW_NORMAL)

finger_in_cell = False
start_time = 0
current_row, current_col = -1, -1


def check_win_and_line(states):

    for i in range(3):
        if states[i, 0] == states[i, 1] == states[i, 2] != 0:
            y = rectangle_y1 + i * cell_height + cell_height // 2
            return int(states[i, 0]), ((rectangle_x1, y), (rectangle_x2, y))

    for i in range(3):
        if states[0, i] == states[1, i] == states[2, i] != 0:
            x = rectangle_x1 + i * cell_width + cell_width // 2
            return int(states[0, i]), ((x, rectangle_y1), (x, rectangle_y2))


    if states[0, 0] == states[1, 1] == states[2, 2] != 0:
        return int(states[0, 0]), ((rectangle_x1, rectangle_y1), (rectangle_x2, rectangle_y2))


    if states[0, 2] == states[1, 1] == states[2, 0] != 0:
        return int(states[0, 2]), ((rectangle_x2, rectangle_y1), (rectangle_x1, rectangle_y2))

    return 0, None



while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)


    for r in range(num_rows):
        for c in range(num_cols):
            x1 = rectangle_x1 + c * cell_width
            y1 = rectangle_y1 + r * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if drawing_states[r, c] == 1:
                cv2.circle(frame, (cx, cy), drawing_size, (255, 0, 0), 5)

            elif drawing_states[r, c] == 2:
                cv2.line(frame, (cx - drawing_size, cy - drawing_size),
                         (cx + drawing_size, cy + drawing_size), (0, 0, 255), 5)
                cv2.line(frame, (cx - drawing_size, cy + drawing_size),
                         (cx + drawing_size, cy - drawing_size), (0, 0, 255), 5)

    if winner == 0:
        winner, win_line = check_win_and_line(drawing_states)

    is_tie = np.all(drawing_states != 0) and winner == 0


    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        index_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        h, w, _ = frame.shape
        fx, fy = int(index_tip.x * w), int(index_tip.y * h)


        if winner == 0 and not is_tie:
            fingertip_color = (255, 0, 0) if turn == 1 else (0, 0, 255)
        else:
            fingertip_color = (0, 255, 0)

        cv2.circle(frame, (fx, fy), 12, fingertip_color, -1)

        if winner != 0 or is_tie:
            if button_x1 < fx < button_x2 and button_y1 < fy < button_y2:
                cv2.rectangle(frame, (button_x1, button_y1), (button_x2, button_y2), (0, 255, 0), -1)

                if not finger_in_cell:
                    finger_in_cell = True
                    start_time = time.time()

                elif time.time() - start_time >= 0.7:
                    drawing_states = np.zeros((num_rows, num_cols), dtype=int)
                    turn = 1
                    winner = 0
                    win_line = None
                    finger_in_cell = False
                    current_row, current_col = -1, -1

            else:
                finger_in_cell = False

        else:

            found = False
            for r in range(3):
                for c in range(3):
                    x1 = rectangle_x1 + c * cell_width
                    y1 = rectangle_y1 + r * cell_height
                    x2 = x1 + cell_width
                    y2 = y1 + cell_height

                    if x1 < fx < x2 and y1 < fy < y2:
                        found = True

                        if drawing_states[r][c] == 0:
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2


                            if turn == 1:
                                cv2.circle(frame, (cx, cy), drawing_size, (255, 0, 0), 2)
                            else:
                                cv2.line(frame, (cx - drawing_size, cy - drawing_size),
                                         (cx + drawing_size, cy + drawing_size), (0, 0, 255), 2)
                                cv2.line(frame, (cx - drawing_size, cy + drawing_size),
                                         (cx + drawing_size, cy - drawing_size), (0, 0, 255), 2)

                            if not finger_in_cell or (r != current_row or c != current_col):
                                finger_in_cell = True
                                start_time = time.time()
                                current_row, current_col = r, c

                            else:
                                if time.time() - start_time >= 0.9:
                                    drawing_states[r][c] = turn
                                    turn = 3 - turn
                                    finger_in_cell = False

                        break
                if found:
                    break

            if not found:
                finger_in_cell = False

    else:
        finger_in_cell = False

   
    if winner != 0 and win_line is not None:
        color = (255, 0, 0) if winner == 1 else (0, 0, 255)
        cv2.line(frame, win_line[0], win_line[1], color, 10)


    if winner != 0:
        color = (255, 0, 0) if winner == 1 else (0, 0, 255)
        cv2.putText(frame, "Winner!", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

        cv2.rectangle(frame, (button_x1, button_y1), (button_x2, button_y2), (0, 200, 0), -1)
        cv2.putText(frame, button_text, (button_x1 + 10, button_y1 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    elif is_tie:
        cv2.putText(frame, "Tie!", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        cv2.rectangle(frame, (button_x1, button_y1), (button_x2, button_y2), (0, 200, 0), -1)
        cv2.putText(frame, button_text, (button_x1 + 10, button_y1 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Combined", frame)

    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
