#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import pandas as pd
from copy import deepcopy

# ==== 导入你已有的工具和模块 ====
from utils import (
    measure_distance,
    draw_player_stats,
    convert_pixel_distance_to_meters,
    get_center_of_bbox
)
import constants
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt

class TennisAnalyzer:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback, queue_size=1, buff_size=2**24)
        self.image_pub = rospy.Publisher("/tennis_analyse/image_processed", Image, queue_size=1)

        self.player_tracker = PlayerTracker(model_path='yolov8n.bin')
        self.ball_tracker = BallTracker(model_path='models/yolo5_last.pt')
        self.court_line_detector = CourtLineDetector("models/keypoints_model.pth")

        self.player_stats_data = []
        self.chosen_players = []
        self.court_keypoints = None
        self.mini_court = None

        self.all_player_detections = {}
        self.all_ball_detections = {}
        self.player_mini_court_detections = {}
        self.ball_mini_court_detections = {}

        self.frame_id = 0

    def update_average_speeds(self, stats):
        for key in stats.keys():
            if key.endswith('_number_of_shots'):
                pid = key.split('_')[1]
                shots = stats[f'player_{pid}_number_of_shots']
                total_shot_speed = stats[f'player_{pid}_total_shot_speed']
                stats[f'player_{pid}_average_shot_speed'] = total_shot_speed / shots if shots > 0 else 0

            if key.endswith('_total_player_speed'):
                pid = key.split('_')[1]
                total_player_speed = stats[f'player_{pid}_total_player_speed']
                stats[f'player_{pid}_average_player_speed'] = total_player_speed / stats['frame_num'] if stats['frame_num'] > 0 else 0
        return stats

    def callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            fps = 30  # 假设帧率，或读取摄像头属性
            frame_id = self.frame_id
            self.frame_id += 1

            player_dets = self.player_tracker.detect_frame(frame)
            ball_dets = self.ball_tracker.detect_frame(frame)

            self.all_player_detections[frame_id] = player_dets
            self.all_ball_detections[frame_id] = ball_dets

            if frame_id == 0:
                self.court_keypoints = self.court_line_detector.predict(frame)
                self.chosen_players = self.player_tracker.choose_players(self.court_keypoints, player_dets)
                initial_stats = { 'frame_num': 0 }
                for pid in self.chosen_players:
                    for key in ['number_of_shots', 'total_shot_speed', 'last_shot_speed', 'total_player_speed', 'last_player_speed', 'average_shot_speed', 'average_player_speed']:
                        initial_stats[f'player_{pid}_{key}'] = 0
                self.player_stats_data = [initial_stats]
                player_dets = {k: v for k, v in player_dets.items() if k in self.chosen_players}
                self.mini_court = MiniCourt(frame)
            else:
                player_dets = {k: v for k, v in player_dets.items() if k in self.chosen_players}

            player_positions_mini = {pid: self.mini_court.convert_bbox_to_mini_court_coords(bbox, self.court_keypoints) for pid, bbox in player_dets.items()}

            ball_pos = ball_dets.get(1, None)
            if ball_pos:
                ball_center = get_center_of_bbox(ball_pos)
                ball_pos_mini = self.mini_court.convert_bbox_to_mini_court_coords([ball_center[0], ball_center[1], ball_center[0], ball_center[1]], self.court_keypoints)
                ball_positions_mini = {1: ball_pos_mini}
            else:
                ball_positions_mini = {1: (0, 0)}

            self.player_mini_court_detections[frame_id] = player_positions_mini
            self.ball_mini_court_detections[frame_id] = ball_positions_mini

            if frame_id > 0:
                prev = frame_id - 1
                ball_speed = convert_pixel_distance_to_meters(measure_distance(self.ball_mini_court_detections[prev][1], ball_positions_mini[1]), constants.DOUBLE_LINE_WIDTH, self.mini_court.get_width_of_mini_court()) * fps * 3.6
                player_speeds = {pid: convert_pixel_distance_to_meters(measure_distance(self.player_mini_court_detections[prev].get(pid, (0, 0)), player_positions_mini.get(pid, (0, 0))), constants.DOUBLE_LINE_WIDTH, self.mini_court.get_width_of_mini_court()) * fps * 3.6 for pid in self.chosen_players}
                shooter = min(player_positions_mini.keys(), key=lambda pid: measure_distance(player_positions_mini[pid], ball_positions_mini[1]))
                opponent = [pid for pid in self.chosen_players if pid != shooter][0]
                current_stats = deepcopy(self.player_stats_data[-1])
                current_stats['frame_num'] = frame_id
                current_stats[f'player_{shooter}_number_of_shots'] += 1
                current_stats[f'player_{shooter}_total_shot_speed'] += ball_speed
                current_stats[f'player_{shooter}_last_shot_speed'] = ball_speed
                current_stats[f'player_{opponent}_total_player_speed'] += player_speeds[opponent]
                current_stats[f'player_{opponent}_last_player_speed'] = player_speeds[opponent]
                current_stats = self.update_average_speeds(current_stats)
                self.player_stats_data.append(current_stats)
            else:
                self.player_stats_data.append(self.update_average_speeds(deepcopy(self.player_stats_data[-1])))

            # 可视化
            frame = self.player_tracker.draw_bboxes([frame], [player_dets])[0]
            frame = self.ball_tracker.draw_bboxes([frame], [ball_dets])[0]
            frame = self.court_line_detector.draw_keypoints_on_video([frame], self.court_keypoints)[0]
            frame = self.mini_court.draw_mini_court([frame])[0]
            frame = self.mini_court.draw_points_on_mini_court([frame], {0: player_positions_mini})[0]
            frame = self.mini_court.draw_points_on_mini_court([frame], {0: ball_positions_mini}, color=(0, 255, 255))[0]
            frame = draw_player_stats([frame], pd.DataFrame([self.player_stats_data[-1]]), self.chosen_players)[0]

            self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

        except Exception as e:
            rospy.logerr(f"处理帧失败: {e}")

if __name__ == '__main__':
    rospy.init_node('tennis_analyse_node', anonymous=True)
    analyzer = TennisAnalyzer()
    rospy.loginfo("网球分析节点已启动")
    rospy.spin()
