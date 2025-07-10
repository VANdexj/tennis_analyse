import cv2
import pandas as pd
from copy import deepcopy
from utils import (
    measure_distance,
    draw_player_stats,
    convert_pixel_distance_to_meters,
    save_video,
    get_center_of_bbox
)
import constants
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt


def update_average_speeds(stats):
    for key in stats.keys():
        if key.endswith('_number_of_shots'):
            pid = key.split('_')[1]
            # 平均击球速度
            shots = stats[f'player_{pid}_number_of_shots']
            total_shot_speed = stats[f'player_{pid}_total_shot_speed']
            stats[f'player_{pid}_average_shot_speed'] = total_shot_speed / shots if shots > 0 else 0

        if key.endswith('_total_player_speed'):
            pid = key.split('_')[1]
            total_player_speed = stats[f'player_{pid}_total_player_speed'] # 平均跑动速度
            # 这里假设你想统计总跑动速度和次数类似击球次数，否则需要自己定义累计次数
            # 如果没有累积跑动速度的次数，可以用frame数代替，或者另设字段
            # 这里简化用当前帧数作为跑动速度累积的除数
            stats[f'player_{pid}_average_player_speed'] = total_player_speed / stats['frame_num'] if stats['frame_num'] > 0 else 0
    return stats

def main():
    input_video_path = "input_videos/1.mp4"
    output_video_path = "output_videos/1_output.mp4"

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    player_tracker = PlayerTracker(model_path='yolov8n.bin')
    ball_tracker = BallTracker(model_path='models/yolo5_last.pt')
    court_line_detector = CourtLineDetector("models/keypoints_model.pth")

    all_player_detections = {}
    all_ball_detections = {}
    player_stats_data = [{
        'frame_num': 0,
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,
        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,
    }]

    court_keypoints = None
    mini_court = None
    chosen_players = []

    player_mini_court_detections = {}
    ball_mini_court_detections = {}

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        player_dets = player_tracker.detect_frame(frame)
        ball_dets = ball_tracker.detect_frame(frame)

        all_player_detections[frame_id] = player_dets
        all_ball_detections[frame_id] = ball_dets

        if frame_id == 0:
            court_keypoints = court_line_detector.predict(frame)
            chosen_players = player_tracker.choose_players(court_keypoints, player_dets)
            # 用真实 ID 初始化 stats
            initial_stats = {
                'frame_num': 0,
            }
            for pid in chosen_players:
                initial_stats[f'player_{pid}_number_of_shots'] = 0
                initial_stats[f'player_{pid}_total_shot_speed'] = 0
                initial_stats[f'player_{pid}_last_shot_speed'] = 0
                initial_stats[f'player_{pid}_total_player_speed'] = 0
                initial_stats[f'player_{pid}_last_player_speed'] = 0
                initial_stats[f'player_{pid}_average_shot_speed'] = 0
                initial_stats[f'player_{pid}_average_player_speed'] = 0
            player_stats_data = [initial_stats]
            player_dets = {k: v for k, v in player_dets.items() if k in chosen_players}
            mini_court = MiniCourt(frame)
        else:
            player_dets = {k: v for k, v in player_dets.items() if k in chosen_players}

        player_positions_mini = {}
        for pid, bbox in player_dets.items():
            player_positions_mini[pid] = mini_court.convert_bbox_to_mini_court_coords(bbox, court_keypoints)  # TODO: 替换为真实 mini court 坐标

        ball_pos = ball_dets.get(1, None)
        if ball_pos:
            ball_center = get_center_of_bbox(ball_pos)
            ball_pos_mini = mini_court.convert_bbox_to_mini_court_coords(
        [ball_center[0], ball_center[1], ball_center[0], ball_center[1]],
        court_keypoints
    )
            ball_positions_mini = {1: ball_pos_mini}  # TODO: 替换为真实 mini court 坐标
            
        else:
            ball_positions_mini = {1: (0, 0)}

        player_mini_court_detections[frame_id] = player_positions_mini
        ball_mini_court_detections[frame_id] = ball_positions_mini

        # 补全：球速与跑动速度逻辑（仅在有前一帧时执行）
        if frame_id > 0:
            prev_frame_id = frame_id - 1
            ball_prev = ball_mini_court_detections[prev_frame_id][1]
            ball_curr = ball_mini_court_detections[frame_id][1]
            player_prev = player_mini_court_detections[prev_frame_id]
            player_curr = player_mini_court_detections[frame_id]

            ball_distance_pixels = measure_distance(ball_prev, ball_curr)
            ball_distance_meters = convert_pixel_distance_to_meters(
                ball_distance_pixels,
                constants.DOUBLE_LINE_WIDTH,
                mini_court.get_width_of_mini_court()
            )
            ball_speed_kmph = ball_distance_meters * fps * 3.6

            player_distances = {}
            for pid in chosen_players:
                if pid in player_prev and pid in player_curr:
                    player_distance_pixels = measure_distance(player_prev[pid], player_curr[pid])
                    player_distance_meters = convert_pixel_distance_to_meters(
                        player_distance_pixels,
                        constants.DOUBLE_LINE_WIDTH,
                        mini_court.get_width_of_mini_court()
                    )
                    player_distances[pid] = player_distance_meters * fps * 3.6
                else:
                    # 如果缺少检测位置，速度设为0或跳过
                    player_distances[pid] = 0

            # 判断谁击球（离球更近的）
            shooter_id = min(player_curr.keys(), key=lambda pid: measure_distance(player_curr[pid], ball_curr))
            opponent_id = [pid for pid in chosen_players if pid != shooter_id][0]

            current_stats = deepcopy(player_stats_data[-1])
            current_stats['frame_num'] = frame_id
            current_stats[f'player_{shooter_id}_number_of_shots'] += 1
            current_stats[f'player_{shooter_id}_total_shot_speed'] += ball_speed_kmph
            current_stats[f'player_{shooter_id}_last_shot_speed'] = ball_speed_kmph
            current_stats[f'player_{opponent_id}_total_player_speed'] += player_distances[opponent_id]
            current_stats[f'player_{opponent_id}_last_player_speed'] = player_distances[opponent_id]
            # 平均速度计算
            current_stats = update_average_speeds(current_stats)
            
            player_stats_data.append(current_stats)
        else:
            current_stats = deepcopy(player_stats_data[-1])
            current_stats = update_average_speeds(current_stats)
            player_stats_data.append(current_stats)

        # 可视化
        frame = player_tracker.draw_bboxes([frame], [player_dets])[0]
        frame = ball_tracker.draw_bboxes([frame], [ball_dets])[0]
        frame = court_line_detector.draw_keypoints_on_video([frame], court_keypoints)[0]
        frame = mini_court.draw_mini_court([frame])[0]
        frame = mini_court.draw_points_on_mini_court([frame], {0: player_positions_mini})[0]
        frame = mini_court.draw_points_on_mini_court([frame], {0: ball_positions_mini}, color=(0, 255, 255))[0]
        frame = draw_player_stats([frame], pd.DataFrame([player_stats_data[-1]]),chosen_players)[0]

        cv2.putText(frame, f"Frame: {frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()

if __name__ == "__main__":
    main()
