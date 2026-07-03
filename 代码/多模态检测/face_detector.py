import os
import cv2
import numpy as np
from rknnlite.api import RKNNLite

from collections import deque

# ==================== 全局配置 ====================
MODEL_ROOT = os.path.join(os.path.dirname(__file__), "../models_file/yolov8_pose")
FACE_MODEL = os.path.join(MODEL_ROOT, "yolov8n-face.rknn")
EMO_MODEL = os.path.join(MODEL_ROOT, "simple_CNN.rknn")

FACE_IMG_SIZE = 640
FACE_CONF_THRESH = 0.55
FACE_IOU_THRESH = 0.5

EMO_IMG_SIZE = (64, 64)

# ==================== 表情识别调参区 ====================
CONFIDENCE_THRESHOLD = 0.45
NEUTRAL_MARGIN = 0.25
ANGRY_MIN_SCORE = 0.45

SMOOTHING_FRAMES = 3
FACE_CROP_MARGIN = 0.25
EMOTION_SKIP_FRAME = 1

DEBUG_EMOTION = True

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

FACE_MATCH_IOU_THRESH = 0.3
FACE_LOST_TIMEOUT = 3
FACE_MAX_TRACK = 5





# ==================== 通用 letterbox ====================
def letterbox_face(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]

    if shape[0] <= 0 or shape[1] <= 0:
        return img, 1.0, (0, 0)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    img = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    return img, r, (dw, dh)


# ==================== 工具函数 ====================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x) + 1e-6)


def xywh2xyxy(x):
    y = np.copy(x)

    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2

    return y


def nms(boxes, scores, iou_threshold):
    if boxes is None or len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
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
        union = areas[i] + areas[order[1:]] - inter

        iou = inter / (union + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def box_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    area1 = max(0, x2_1 - x1_1) * max(0, y2_1 - y1_1)
    area2 = max(0, x2_2 - x1_2) * max(0, y2_2 - y1_2)

    union_area = area1 + area2 - inter_area

    return inter_area / (union_area + 1e-6)


def crop_face_with_margin(frame, box, margin=FACE_CROP_MARGIN):
    """
    扩大裁脸区域。
    Angry 主要依赖眉毛、眼睛、嘴角，如果裁得太紧，很容易被识别成 Neutral。
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    bw = x2 - x1
    bh = y2 - y1

    if bw <= 1 or bh <= 1:
        return None

    size = max(bw, bh)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    size = size * (1.0 + margin)

    nx1 = int(cx - size / 2.0)
    ny1 = int(cy - size / 2.0)
    nx2 = int(cx + size / 2.0)
    ny2 = int(cy + size / 2.0)

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(w, nx2)
    ny2 = min(h, ny2)

    if nx2 <= nx1 or ny2 <= ny1:
        return None

    face_img = frame[ny1:ny2, nx1:nx2]

    if face_img is None or face_img.size == 0:
        return None

    return face_img


# ==================== 人脸后处理 ====================
def postprocess_face(outputs, img_shape, r, dwdh):
    if outputs is None or len(outputs) == 0 or outputs[0] is None:
        return []

    outputs = outputs[0]
    outputs = np.squeeze(outputs)

    if outputs.ndim != 2:
        return []

    # 兼容 YOLOv8 输出:
    # 常见为 (5, 8400)，需要转置为 (8400, 5)
    if outputs.shape[0] <= outputs.shape[1]:
        outputs = outputs.T

    if outputs.shape[1] < 5:
        return []

    boxes = outputs[:, :4]
    scores = outputs[:, 4]

    mask = scores > FACE_CONF_THRESH
    boxes = boxes[mask]
    scores = scores[mask]

    if len(boxes) == 0:
        return []

    boxes = xywh2xyxy(boxes)

    keep = nms(boxes, scores, FACE_IOU_THRESH)

    if len(keep) == 0:
        return []

    boxes = boxes[keep]

    dw, dh = dwdh

    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes /= r

    h, w = img_shape[:2]

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)

    boxes = boxes.astype(int)

    valid_boxes = []

    for box in boxes:
        x1, y1, x2, y2 = box
        if x2 > x1 and y2 > y1:
            valid_boxes.append((int(x1), int(y1), int(x2), int(y2)))

    return valid_boxes


# ==================== 表情预处理 ====================
def preprocess_emotion(face_img):
    """
    重要：
    你的 simple_CNN.rknn 日志显示支持输入 shape 是：

        [1, 64, 1, 64], layout = NHWC

    所以这里必须输出：

        (1, 64, 1, 64)

    不能输出标准的：

        (1, 64, 64, 1)

    否则会报：
        rknn_set_input_shapes error
        Set input shape failed
    """
    if face_img is None or face_img.size == 0:
        return None

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    except Exception:
        pass

    resized = cv2.resize(gray, EMO_IMG_SIZE, interpolation=cv2.INTER_LINEAR)

    normalized = resized.astype(np.float32) / 255.0

    # 先构造标准 NHWC: (1, 64, 64, 1)
    img_input = normalized[np.newaxis, :, :, np.newaxis]

    # 再转成你的 RKNN 支持的 shape: (1, 64, 1, 64)
    img_input = img_input.transpose(0, 2, 3, 1)

    img_input = np.ascontiguousarray(img_input, dtype=np.float32)

    if DEBUG_EMOTION:
        print(f"📐 [表情输入shape] {img_input.shape}")

    return img_input


# ==================== 表情后处理 ====================
def postprocess_emotion(outputs):
    """
    改进版表情后处理：

    1. 如果输出不是概率，自动 softmax。
    2. Angry 达到较低阈值就保留。
    3. Neutral 和 Angry 很接近时，优先修正为 Angry。
    4. 打印每类概率，方便判断模型真实输出。
    """
    neutral_idx = CLASS_NAMES.index("Neutral")
    angry_idx = CLASS_NAMES.index("Angry")

    if outputs is None or len(outputs) == 0 or outputs[0] is None:
        prob = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        prob[neutral_idx] = 1.0
        return neutral_idx, 0.0, "Neutral", prob

    pred = np.asarray(outputs[0]).reshape(-1).astype(np.float32)

    if pred.size < len(CLASS_NAMES):
        prob = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        prob[neutral_idx] = 1.0
        return neutral_idx, 0.0, "Neutral", prob

    pred = pred[:len(CLASS_NAMES)]

    # 判断是否已经是概率；不是就 softmax
    if pred.min() < 0 or pred.max() > 1 or abs(float(pred.sum()) - 1.0) > 0.2:
        prob = softmax(pred)
    else:
        prob = pred.astype(np.float32)

    pred_idx = int(np.argmax(prob))
    pred_score = float(prob[pred_idx])
    pred_label = CLASS_NAMES[pred_idx]

    angry_score = float(prob[angry_idx])
    neutral_score = float(prob[neutral_idx])

    if DEBUG_EMOTION:
        prob_text = " | ".join(
            [f"{CLASS_NAMES[i]}:{float(prob[i]):.3f}" for i in range(len(CLASS_NAMES))]
        )
        print(f"😐 [表情概率] top={pred_label}:{pred_score:.3f} | {prob_text}")

    # 情况1：top1 本来就是 Angry，只要不是特别低，就保留
    if pred_label == "Angry" and pred_score >= ANGRY_MIN_SCORE:
        return angry_idx, pred_score, "Angry", prob

    # 情况3：其它异常表情，只要超过阈值就保留
    if pred_label != "Neutral" and pred_score >= CONFIDENCE_THRESHOLD:
        return pred_idx, pred_score, pred_label, prob

    # 情况4：确实是 Neutral
    return neutral_idx, neutral_score, "Neutral", prob


# ==================== 人脸跟踪 + 表情平滑 ====================
class FaceEmotionTracker:
    def __init__(self, rknn_emo_model):
        self.rknn_emo = rknn_emo_model
        self.tracked_faces = {}
        self.next_id = 1
        self.reset()

    def reset(self):
        self.tracked_faces = {}
        self.next_id = 1
        print("🔄 FaceTracker 已重置：无旧人脸残留")

    def _run_emotion(self, frame, face_box):
        neutral_idx = CLASS_NAMES.index("Neutral")

        face_img = crop_face_with_margin(frame, face_box, margin=FACE_CROP_MARGIN)

        if face_img is None or face_img.size == 0:
            prob = np.zeros(len(CLASS_NAMES), dtype=np.float32)
            prob[neutral_idx] = 1.0
            return neutral_idx, 0.0, "Neutral", prob

        emo_input = preprocess_emotion(face_img)

        if emo_input is None:
            prob = np.zeros(len(CLASS_NAMES), dtype=np.float32)
            prob[neutral_idx] = 1.0
            return neutral_idx, 0.0, "Neutral", prob

        try:
            emo_outputs = self.rknn_emo.inference(
                inputs=[emo_input],
                data_format="nhwc",
                data_type="float32",
            )
        except Exception as e:
            print(f"❌ [表情RKNN推理失败] input_shape={emo_input.shape}, error={e}")

            # 兜底再试一次：有些 RKNN 模型不喜欢显式 data_format/data_type
            try:
                print("🔁 [表情RKNN] 尝试不传 data_format/data_type 再推理一次...")
                emo_outputs = self.rknn_emo.inference(inputs=[emo_input])
            except Exception as e2:
                print(f"❌ [表情RKNN二次推理仍失败] input_shape={emo_input.shape}, error={e2}")
                prob = np.zeros(len(CLASS_NAMES), dtype=np.float32)
                prob[neutral_idx] = 1.0
                return neutral_idx, 0.0, "Neutral", prob

        idx, score, label, prob = postprocess_emotion(emo_outputs)

        return idx, score, label, prob

    def _get_smoothed_emotion(self, prob_queue):
        """
        改进版平滑：
        1. 使用概率平均，比标签投票更稳定。
        2. 如果 Neutral 和 Angry 接近，优先输出 Angry。
        3. 避免 Neutral 长期压制 Angry。
        """
        neutral_idx = CLASS_NAMES.index("Neutral")
        angry_idx = CLASS_NAMES.index("Angry")

        if prob_queue is None or len(prob_queue) == 0:
            return neutral_idx, 0.0, "Neutral"

        avg_prob = np.mean(list(prob_queue), axis=0)

        final_idx = int(np.argmax(avg_prob))
        final_score = float(avg_prob[final_idx])
        final_label = CLASS_NAMES[final_idx]

        angry_score = float(avg_prob[angry_idx])
        neutral_score = float(avg_prob[neutral_idx])

        if DEBUG_EMOTION:
            avg_text = " | ".join(
                [f"{CLASS_NAMES[i]}:{float(avg_prob[i]):.3f}" for i in range(len(CLASS_NAMES))]
            )
            print(f"🧠 [表情平滑] top={final_label}:{final_score:.3f} | {avg_text}")

        # 平滑后 Neutral 第一，但 Angry 接近，优先 Angry
        if final_label == "Neutral":
            if angry_score >= ANGRY_MIN_SCORE and (neutral_score - angry_score) <= NEUTRAL_MARGIN:
                if DEBUG_EMOTION:
                    print(
                        f"⚠️ [表情平滑修正] Neutral 与 Angry 接近，输出 Angry: "
                        f"Neutral={neutral_score:.3f}, Angry={angry_score:.3f}"
                    )
                return angry_idx, angry_score, "Angry"

        # Angry top1，直接保留
        if final_label == "Angry" and final_score >= ANGRY_MIN_SCORE:
            return angry_idx, final_score, "Angry"

        # 其它非 Neutral 表情
        if final_label != "Neutral" and final_score >= CONFIDENCE_THRESHOLD:
            return final_idx, final_score, final_label

        return neutral_idx, neutral_score, "Neutral"

    def update(self, face_boxes, frame, run_emotion=True):
        current_face_ids = set()
        h, w = frame.shape[:2]

        if face_boxes is None:
            face_boxes = []

        # 大脸优先
        face_boxes = sorted(
            face_boxes,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
            reverse=True,
        )

        for face_box in face_boxes:
            x1, y1, x2, y2 = face_box

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))

            if x2 <= x1 or y2 <= y1:
                continue

            face_box = (x1, y1, x2, y2)

            best_match_id = None
            best_iou = 0.0

            for tid, face_info in self.tracked_faces.items():
                iou = box_iou(face_box, face_info["bbox"])

                if iou > best_iou and iou > FACE_MATCH_IOU_THRESH:
                    best_iou = iou
                    best_match_id = tid

            if best_match_id is not None:
                self.tracked_faces[best_match_id]["bbox"] = face_box
                self.tracked_faces[best_match_id]["lost"] = 0

                if run_emotion:
                    idx, score, label, prob = self._run_emotion(frame, face_box)

                    self.tracked_faces[best_match_id]["prob_queue"].append(prob)

                    final_idx, final_score, final_label = self._get_smoothed_emotion(
                        self.tracked_faces[best_match_id]["prob_queue"]
                    )

                    self.tracked_faces[best_match_id]["label"] = final_label
                    self.tracked_faces[best_match_id]["score"] = final_score
                    self.tracked_faces[best_match_id]["raw_label"] = label
                    self.tracked_faces[best_match_id]["raw_score"] = score

                current_face_ids.add(best_match_id)

            else:
                if len(self.tracked_faces) >= FACE_MAX_TRACK:
                    continue

                idx, score, label, prob = self._run_emotion(frame, face_box)

                prob_queue = deque([prob], maxlen=SMOOTHING_FRAMES)

                final_idx, final_score, final_label = self._get_smoothed_emotion(prob_queue)

                self.tracked_faces[self.next_id] = {
                    "bbox": face_box,
                    "prob_queue": prob_queue,
                    "label": final_label,
                    "score": final_score,
                    "raw_label": label,
                    "raw_score": score,
                    "lost": 0,
                }

                current_face_ids.add(self.next_id)
                self.next_id += 1

        # 处理丢失人脸
        to_delete = []

        for tid in list(self.tracked_faces.keys()):
            if tid not in current_face_ids:
                self.tracked_faces[tid]["lost"] += 1

                if self.tracked_faces[tid]["lost"] > FACE_LOST_TIMEOUT:
                    to_delete.append(tid)

        for tid in to_delete:
            del self.tracked_faces[tid]

        return self.tracked_faces


# ==================== 对外接口类 ====================
class FaceDetector:
    def __init__(self):
        print("🔹 加载人脸 + 表情模型...")

        if not os.path.exists(FACE_MODEL):
            raise FileNotFoundError(f"人脸模型不存在: {FACE_MODEL}")

        if not os.path.exists(EMO_MODEL):
            raise FileNotFoundError(f"表情模型不存在: {EMO_MODEL}")

        self.rknn_face = RKNNLite()
        ret = self.rknn_face.load_rknn(FACE_MODEL)

        if ret != 0:
            raise RuntimeError(f"加载人脸模型失败: {FACE_MODEL}, ret={ret}")

        ret = self.rknn_face.init_runtime(core_mask=RKNNLite.NPU_CORE_1)

        if ret != 0:
            raise RuntimeError(f"初始化人脸模型 NPU 失败, ret={ret}")

        self.rknn_emo = RKNNLite()
        ret = self.rknn_emo.load_rknn(EMO_MODEL)

        if ret != 0:
            raise RuntimeError(f"加载表情模型失败: {EMO_MODEL}, ret={ret}")

        ret = self.rknn_emo.init_runtime(core_mask=RKNNLite.NPU_CORE_1)

        if ret != 0:
            raise RuntimeError(f"初始化表情模型 NPU 失败, ret={ret}")

        self.face_tracker = FaceEmotionTracker(self.rknn_emo)
        self.frame_count = 0

        print("✅ 人脸检测器初始化完成")
        print(f"✅ 表情阈值 CONFIDENCE_THRESHOLD={CONFIDENCE_THRESHOLD}")
        print(f"✅ Angry 最小分数 ANGRY_MIN_SCORE={ANGRY_MIN_SCORE}")
        print(f"✅ Neutral 修正间隔 NEUTRAL_MARGIN={NEUTRAL_MARGIN}")
        print(f"✅ 裁脸扩大比例 FACE_CROP_MARGIN={FACE_CROP_MARGIN}")
        print(f"✅ 平滑帧数 SMOOTHING_FRAMES={SMOOTHING_FRAMES}")
        print("✅ 表情输入 shape 已修正为 RKNN 需要的 (1, 64, 1, 64)")

    def reset(self):
        print("🔄 FaceDetector 执行重置")
        self.frame_count = 0

        if self.face_tracker:
            self.face_tracker.reset()

    def detect_frame(self, frame):
        self.frame_count += 1

        run_emotion = self.frame_count % EMOTION_SKIP_FRAME == 0

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_pad_face, r_face, dwdh_face = letterbox_face(
            img_rgb,
            (FACE_IMG_SIZE, FACE_IMG_SIZE),
        )

        input_face = np.expand_dims(img_pad_face, axis=0)
        input_face = np.ascontiguousarray(input_face)

        face_outputs = self.rknn_face.inference(
            inputs=[input_face],
            data_format="nhwc",
        )

        face_boxes = postprocess_face(
            face_outputs,
            frame.shape,
            r_face,
            dwdh_face,
        )

        tracked_faces = self.face_tracker.update(
            face_boxes,
            frame,
            run_emotion=run_emotion,
        )

        return frame, tracked_faces

    def release(self):
        try:
            if self.rknn_face:
                self.rknn_face.release()
        except Exception:
            pass

        try:
            if self.rknn_emo:
                self.rknn_emo.release()
        except Exception:
            pass

        print("✅ 人脸模型资源已释放")
