import cv2
import numpy as np
import pandas as pd
import csv
import sys
from scipy.optimize import linear_sum_assignment

MAP_W, MAP_H = 800, 1200
CX, CY = MAP_W // 2, MAP_H // 2
R_PX = 110

FIXED_BALL_ID = ['BALL']
FIXED_RED_IDS = [f'A{i}' for i in range(1, 8)]
FIXED_BLUE_IDS = [f'B{i}' for i in range(1, 8)]
ALL_15_IDS = FIXED_BALL_ID + FIXED_RED_IDS + FIXED_BLUE_IDS

# 실행 시 인자로 영상명을 넣거나, 없으면 기본 sc.mp4로 동작
video_path = sys.argv[1] if len(sys.argv) > 1 else "sc.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"'{video_path}' 영상을 열 수 없습니다. 파일 이름이나 경로를 확인해주세요.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps): fps = 30.0

skip_interval = max(1, int(fps * 0.1))

vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

display_h = 900
display_w = int(display_h * (vw / vh)) if vh > 0 else 600

cv2.namedWindow("Tactical Analysis", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tactical Analysis", display_w, display_h)

cv2.namedWindow("2D Tactical Minimap", cv2.WINDOW_NORMAL)
cv2.resizeWindow("2D Tactical Minimap", 400, 600)

csv_filename = "tactical_positions.csv"

with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['Time(sec)', 'ID', 'Team', 'X_Coord', 'Y_Coord'])

def separate_ball_and_players(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red1, red2)
    blue_mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([135, 255, 255]))
    team_mask = cv2.bitwise_or(red_mask, blue_mask)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=22,
        param1=50, param2=22, minRadius=10, maxRadius=35
    )

    ball_info, player_list = None, []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cx, cy, r = i[0], i[1], i[2]
            if cx < 20 or cx > vw - 20 or cy < 20 or cy > vh - 20: continue

            y1, y2 = max(0, cy - r), min(vh, cy + r)
            x1, x2 = max(0, cx - r), min(vw, cx + r)
            crop_team = team_mask[y1:y2, x1:x2]
            crop_frame = frame[y1:y2, x1:x2]

            if crop_team.size == 0: continue

            if (np.count_nonzero(crop_team) / crop_team.size) < 0.05 and r <= 18:
                if ball_info is None or r < ball_info[2]:
                    ball_info = (cx, cy, r)
            else:
                player_list.append((cx, cy, r, crop_frame))

    return ball_info, player_list

def classify_team_color(crop):
    if crop.size == 0: return (0, 0, 220), "Red"
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_crop)
    non_grass = ~((h > 35) & (h < 85) & (s > 40))
    valid_h = h[non_grass]
    if len(valid_h) == 0:
        b, g, r = cv2.split(crop)
        return ((0, 0, 220), "Red") if np.mean(r) > np.mean(b) else ((220, 50, 50), "Blue")
    return ((0, 0, 220), "Red") if (np.mean(valid_h) < 20 or np.mean(valid_h) > 150) else ((220, 50, 50), "Blue")

prev_positions = {id_name: None for id_name in ALL_15_IDS}
velocities = {id_name: (0, 0) for id_name in ALL_15_IDS}

def match_team_robust(detected_players, team_ids, max_distance=60.0):
    if not detected_players:
        return []

    det_coords = [p[0] for p in detected_players]

    predicted_coords = []
    for lid in team_ids:
        pos = prev_positions[lid]
        vel = velocities[lid]
        if pos is not None:
            predicted_coords.append((pos[0] + vel[0], pos[1] + vel[1]))
        else:
            predicted_coords.append(None)

    valid_indices = [i for i, p in enumerate(predicted_coords) if p is not None]

    if not valid_indices:
        results = []
        for idx, (coord, crop) in enumerate(detected_players[:len(team_ids)]):
            lid = team_ids[idx]
            prev_positions[lid] = coord
            velocities[lid] = (0, 0)
            results.append((lid, coord, crop))
        return results

    cost_matrix = np.full((len(team_ids), len(det_coords)), 1e5)
    for i, lid in enumerate(team_ids):
        pred = predicted_coords[i]
        if pred is not None:
            for j, det in enumerate(det_coords):
                dist = np.hypot(pred[0] - det[0], pred[1] - det[1])
                cost_matrix[i, j] = dist

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    results = []
    assigned_det = set()
    assigned_ids = set()

    for r, c in zip(row_ind, col_ind):
        dist = cost_matrix[r, c]
        if dist <= max_distance:
            lid = team_ids[r]
            coord = det_coords[c]
            crop = detected_players[c][1]

            old_pos = prev_positions[lid]
            if old_pos is not None:
                new_vel = (coord[0] - old_pos[0], coord[1] - old_pos[1])
                velocities[lid] = (0.7 * velocities[lid][0] + 0.3 * new_vel[0],
                                   0.7 * velocities[lid][1] + 0.3 * new_vel[1])

            prev_positions[lid] = coord
            assigned_det.add(c)
            assigned_ids.add(lid)
            results.append((lid, coord, crop))

    return results

raw_frame_count = 0
step_index = 0

print(f"'{video_path}' 영상 분석 시뮬레이션을 시작합니다...")

while True:
    success, frame = cap.read()
    if not success:
        break

    if raw_frame_count % skip_interval == 0:
        # 실제 프레임 위치에서 시각을 구한다.
        #   예전에는 step_index * 0.1 로 적었는데, skip_interval 이 int(fps*0.1)
        #   이라 fps 가 30·50·60 이 아니면 실제 간격이 0.1초가 아니다.
        #   (25fps -> 0.08초, 29.97fps -> 0.067초) 그런데도 0.1초라고 적으므로
        #   시각이 최대 50% 틀리고, 이 시각으로 속도를 구하는 뒷단(압박 기여도,
        #   이상치 속도 판정)이 통째로 그만큼 어긋난다.
        timestamp_sec = round(raw_frame_count / fps, 3)
        ball_info, player_list = separate_ball_and_players(frame)

        minimap = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)
        minimap[:] = (34, 139, 34)
        cv2.rectangle(minimap, (40, 40), (MAP_W - 40, MAP_H - 40), (255, 255, 255), 3)
        cv2.line(minimap, (40, CY), (MAP_W - 40, CY), (255, 255, 255), 3)
        cv2.circle(minimap, (CX, CY), R_PX, (255, 255, 255), 3)
        cv2.circle(minimap, (CX, CY), 5, (255, 255, 255), -1)

        current_frame_data = {id_name: None for id_name in ALL_15_IDS}

        if ball_info is not None:
            bx, by, br = ball_info
            norm_bx, norm_by = bx / vw, by / vh
            current_frame_data["BALL"] = (round((norm_bx - 0.5) * 100, 2), round((0.5 - norm_by) * 100, 2))

            bmx, bmy = int(norm_bx * MAP_W), int(norm_by * MAP_H)
            if 0 <= bmx < MAP_W and 0 <= bmy < MAP_H:
                cv2.circle(minimap, (bmx, bmy), 8, (0, 255, 255), -1)
                cv2.circle(minimap, (bmx, bmy), 8, (0, 0, 0), 2)
            cv2.circle(frame, (bx, by), br + 3, (0, 255, 255), 2)

        red_players, blue_players = [], []
        for cx, cy, r, crop in player_list:
            dot_color, team_type = classify_team_color(crop)
            if team_type == "Red": red_players.append(((cx, cy), crop))
            else: blue_players.append(((cx, cy), crop))

        matched_red = match_team_robust(red_players, FIXED_RED_IDS, max_distance=60.0)
        matched_blue = match_team_robust(blue_players, FIXED_BLUE_IDS, max_distance=60.0)

        for label, (cx, cy), crop in matched_red + matched_blue:
            dot_color = (0, 0, 220) if label.startswith("A") else (220, 50, 50)
            norm_x, norm_y = cx / vw, cy / vh
            current_frame_data[label] = (round((norm_x - 0.5) * 100, 2), round((0.5 - norm_y) * 100, 2))

            mx, my = int(norm_x * MAP_W), int(norm_y * MAP_H)
            if 0 <= mx < MAP_W and 0 <= my < MAP_H:
                cv2.circle(minimap, (mx, my), 12, dot_color, -1)
                cv2.circle(minimap, (mx, my), 12, (255, 255, 255), 2)
                cv2.putText(minimap, label, (mx - 10, my + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.circle(frame, (cx, cy), 15, dot_color, 2)

        rows_to_append = []
        for id_name in ALL_15_IDS:
            team_str = 'Ball' if id_name == 'BALL' else ('Red' if id_name.startswith('A') else 'Blue')
            coords = current_frame_data[id_name]
            x_val = coords[0] if coords else None
            y_val = coords[1] if coords else None
            rows_to_append.append([timestamp_sec, id_name, team_str, x_val, y_val])

        with open(csv_filename, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerows(rows_to_append)

        cv2.imshow("Tactical Analysis", frame)
        cv2.imshow("2D Tactical Minimap", minimap)

        if cv2.waitKey(1) & 0xFF == ord('q'): break
        step_index += 1

    raw_frame_count += 1

cap.release()
cv2.destroyAllWindows()

raw_df = pd.read_csv(csv_filename)

if not raw_df.empty:
    # 짧은 끊김만 메운다.
    #   limit_direction='both' 는 구간 길이를 안 보고 전부 메우고, 앞뒤 끝의
    #   빈 칸까지 첫값·끝값으로 채운다. 그러면 한 번도 안 잡힌 구간 30초가
    #   직선으로 만들어지고, 그 만들어낸 좌표가 진짜와 구별되지 않은 채
    #   기여도 계산에 들어간다. 검출이 안 된 구간은 비워 두는 편이 낫다 —
    #   뒷단은 빈 칸을 '모름'으로 처리하지만, 채워진 거짓값은 믿어 버린다.
    MAX_FILL_STEPS = 3          # 0.1초 간격 기준 0.3초까지만
    n_before = raw_df[['X_Coord', 'Y_Coord']].isna().sum().sum()
    for c in ('X_Coord', 'Y_Coord'):
        raw_df[c] = raw_df.groupby('ID')[c].transform(
            lambda g: g.interpolate(method='linear', limit=MAX_FILL_STEPS,
                                    limit_area='inside'))
    n_after = raw_df[['X_Coord', 'Y_Coord']].isna().sum().sum()
    print(f"- 짧은 끊김 보간: {n_before - n_after}칸 채움 / {n_after}칸은 비워 둠 "
          f"(최대 {MAX_FILL_STEPS}칸 = {MAX_FILL_STEPS * 0.1:.1f}초)")

    raw_df['X_Coord'] = raw_df['X_Coord'].round(2)
    raw_df['Y_Coord'] = raw_df['Y_Coord'].round(2)

    raw_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')

    total_rows = len(raw_df)
    print("\n[보정 완료]")
    print(f"- 저장 완료: 총 {total_rows}행")
    print(f"- 15의 배수 검증: {'정상 완료 (TRUE)' if total_rows % 15 == 0 else '오류 (FALSE)'}")
