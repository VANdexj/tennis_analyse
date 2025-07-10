import cv2
import pickle
import numpy as np
from hobot_dnn import pyeasy_dnn as dnn
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path):
        # 加载模型并打印输入输出信息
        try:
            self.model = dnn.load(model_path)[0]
            print("模型输入信息:")
            for i, in_tensor in enumerate(self.model.inputs):
                print(f"输入[{i}]: 名称={in_tensor.name}, 形状={in_tensor.properties.shape}")
            print("模型输出信息:")
            for i, out_tensor in enumerate(self.model.outputs):
                print(f"输出[{i}]: 名称={out_tensor.name}, 形状={out_tensor.properties.shape}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise
        
        # 保存模型输入尺寸和反量化系数
        self.input_tensor = self.model.inputs[0]
        self.input_h, self.input_w = self.input_tensor.properties.shape[2:4]
        
        # 存储输出张量的反量化系数
        self.output_scales = [out_tensor.properties.scale_data for out_tensor in self.model.outputs]
        
        # 预计算锚点和DFL权重（与示例程序类似）
        self._prepare_anchors_and_dfl()
    
    def _prepare_anchors_and_dfl(self):
        """预计算锚点和DFL权重，与示例程序保持一致"""
        # DFL求期望的系数
        self.weights_static = np.array([i for i in range(16)]).astype(np.float32)[np.newaxis, np.newaxis, :]
        
        # 生成不同尺度的锚点
        self.s_anchor = np.stack([
            np.tile(np.linspace(0.5, 79.5, 80), reps=80),
            np.repeat(np.arange(0.5, 80.5, 1), 80)
        ], axis=0).transpose(1,0)
        
        self.m_anchor = np.stack([
            np.tile(np.linspace(0.5, 39.5, 40), reps=40),
            np.repeat(np.arange(0.5, 40.5, 1), 40)
        ], axis=0).transpose(1,0)
        
        self.l_anchor = np.stack([
            np.tile(np.linspace(0.5, 19.5, 20), reps=20),
            np.repeat(np.arange(0.5, 20.5, 1), 20)
        ], axis=0).transpose(1,0)
    
    def detect_frame(self, frame):
        """检测单帧中的球员"""
        # 保存原始图像尺寸，用于后处理时还原边界框
        orig_h, orig_w = frame.shape[:2]
        
        # 图像预处理：调整大小并转换为NV12格式
        input_tensor = self._preprocess(frame)
        
        # 模型推理
        outputs = self.model.forward(input_tensor)
        output_array = [out.buffer for out in outputs]
        
        # 后处理：解析模型输出，提取球员边界框
        return self._postprocess(output_array, orig_w, orig_h)
    
    def _preprocess(self, img):
        """图像预处理：调整大小并转换为NV12格式"""
        # 调整图像大小
        resized_img = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_NEAREST)
        
        # 转换为NV12格式（与示例程序一致）
        height, width = resized_img.shape[0], resized_img.shape[1]
        area = height * width
        
        # BGR转YUV420P
        yuv420p = cv2.cvtColor(resized_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        
        # 分离Y和UV分量并重组为NV12
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        
        return nv12

    def _postprocess(self, outputs, orig_w, orig_h):
        input_h = self.input_h
        input_w = self.input_w

        decoded_detections = {}  # 使用字典存储检测结果，键为临时ID，值为边界框信息
        next_id = 0  # 临时ID生成器

        for i in range(3):  # 对三个尺度
            # 提取bbox回归输出
            bbox_feat = outputs[i][0]  # shape: (H, W, 64)
            cls_feat = outputs[i + 3][0]  # shape: (H, W, 80)

            H, W, _ = bbox_feat.shape
            stride = input_w // W  # 例如 640/80=8, 640/40=16

            # 网格坐标
            grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            grid_x = grid_x.reshape(-1)
            grid_y = grid_y.reshape(-1)

            # reshape到一维
            bbox_feat = bbox_feat.reshape(-1, 64)  # (H*W, 64)
            cls_feat = cls_feat.reshape(-1, 80)    # (H*W, 80)

            # 解码 bbox：取前4个通道
            raw_xywh = bbox_feat[:, :4]  # 形状 (N, 4)

            # 以下是假设性的解码逻辑，根据你模型实际可能要调整
            cx = (self._sigmoid(raw_xywh[:, 0]) + grid_x) * stride
            cy = (self._sigmoid(raw_xywh[:, 1]) + grid_y) * stride
            w = np.exp(raw_xywh[:, 2]) * stride
            h = np.exp(raw_xywh[:, 3]) * stride

            # 处理指数溢出问题
            w = np.clip(w, 0, orig_w)  # 限制宽度不超过原图宽度
            h = np.clip(h, 0, orig_h)  # 限制高度不超过原图高度

            # 左上右下
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            bboxes = np.stack([x1, y1, x2, y2], axis=1)

            # 置信度和类别
            scores = self._sigmoid(cls_feat)  # softmax 也可
            class_ids = np.argmax(scores, axis=1)
            confs = scores[np.arange(len(scores)), class_ids]

            # 按置信度筛选
            mask = confs > 0.3
            bboxes = bboxes[mask]
            class_ids = class_ids[mask]
            confs = confs[mask]

            # 缩放 bbox 到原图尺寸
            bboxes = self._scale_coords(bboxes, input_shape=(input_h, input_w), original_shape=(orig_h, orig_w))

            # 转换为整数并添加到字典
            for box, score, cls in zip(bboxes, confs, class_ids):
                x1, y1, x2, y2 = map(int, box)
                decoded_detections[next_id] = [x1, y1, x2, y2]
                next_id += 1

        return decoded_detections  # 返回字典而不是列表

    def _scale_coords(self, coords, input_shape, original_shape):
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError(f"Unexpected coords shape: {coords.shape}")

        gain_w = original_shape[1] / input_shape[1]
        gain_h = original_shape[0] / input_shape[0]

        coords[:, 0] *= gain_w  # x1
        coords[:, 2] *= gain_w  # x2
        coords[:, 1] *= gain_h  # y1
        coords[:, 3] *= gain_h  # y2

        return coords

    
    def _softmax(self, x, axis=None):
        """自定义softmax函数，避免依赖外部库"""
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)
    
    # 以下方法保持不变，与原代码一致
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def choose_and_filter_players(self, court_keypoints, player_detections):
        first_frame_players = player_detections[0]
        chosen = self.choose_players(court_keypoints, first_frame_players)

        filtered = []
        for frame_detections in player_detections:
            filtered.append({pid: bbox for pid, bbox in frame_detections.items() if pid in chosen})
        return filtered

    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)
            min_dist = min(measure_distance(player_center, (court_keypoints[i], court_keypoints[i+1]))
                           for i in range(0, len(court_keypoints), 2))
            distances.append((track_id, min_dist))
        distances.sort(key=lambda x: x[1])
        return [distances[0][0], distances[1][0]]

    def draw_bboxes(self, frames, player_detections):
        output_frames = []
        for frame, detections in zip(frames, player_detections):
            for pid, bbox in detections.items():
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"ID: {pid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            output_frames.append(frame)
        return output_frames

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _multiclass_nms(self, boxes, scores, class_ids, iou_threshold=0.5, score_threshold=0.1):
        """
        对多类别框进行非极大值抑制
        boxes: numpy array (N, 4)
        scores: numpy array (N,)
        class_ids: numpy array (N,)
        返回值是保留下来的索引列表
        """
        keep = []
        unique_classes = np.unique(class_ids)
        
        for cls in unique_classes:
            cls_mask = (class_ids == cls)
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            
            # 过滤掉低分框
            valid_mask = cls_scores > score_threshold
            if not np.any(valid_mask):
                continue
            
            cls_boxes = cls_boxes[valid_mask]
            cls_scores = cls_scores[valid_mask]
            
            # 调用已有的 nms 方法
            keep_cls = self.nms(cls_boxes, cls_scores, iou_threshold)
            
            # 将局部索引转换成全局索引
            global_indices = np.where(cls_mask)[0][valid_mask][keep_cls]
            keep.extend(global_indices.tolist())
        
        return keep


    def nms(self, boxes, scores, iou_thresh):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]
        return keep

    
