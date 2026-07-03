import cv2
import numpy as np
import os
from rknnlite.api import RKNNLite
from collections import deque
from datetime import datetime
import json

# ==================== 导入配置 ====================
from scene_configs import SceneConfig, DEFAULT_CONFIG

# ==================== 模型路径 ====================
MODEL_ROOT = os.path.join(os.path.dirname(__file__), "../models_file/yolov8_pose")
POSE_MODEL = os.path.join(MODEL_ROOT, "yolov8n-pose.rknn")

# 兼容旧版OpenCV属性名
def get_cv_prop(prop_name):
    if hasattr(cv2, prop_name):
        return getattr(cv2, prop_name)
    old_prop = prop_name.replace("CAP_", "CV_CAP_")
    if hasattr(cv2, old_prop):
        return getattr(cv2, old_prop)
    return None

CAP_PROP_FPS = get_cv_prop("CAP_PROP_FPS")
CAP_PROP_FRAME_WIDTH = get_cv_prop("CAP_PROP_FRAME_WIDTH")
CAP_PROP_FRAME_HEIGHT = get_cv_prop("CAP_PROP_FRAME_HEIGHT")

# ==================== 目标跟踪器（彻底修复残留） ====================
class Tracker:
    def __init__(self, config: SceneConfig = DEFAULT_CONFIG, max_history=60, iou_thresh=0.3, lost_timeout=150):
        self.max_history = max_history
        self.iou_thresh = iou_thresh
        self.config = config
        self.lost_timeout = lost_timeout
        self.reset()

    def reset(self):
        print("\n" + "="*80)
        print("[Tracker Reset] 🔥 开始执行完全重置...")
        
        # 🔥 核心修复：彻底清空所有跟踪状态，无任何残留
        self.tracks = {}                  # 所有跟踪目标
        self.next_id = 1                  # 下一个跟踪ID
        self.abnormal_history = {}        # 异常行为历史
        self.abnormal_ids = set()         # 异常目标ID

        # 🔥 清空所有行为计数器（打架/摔倒/奔跑/追逐/聚集）
        self.swing_counter = {}           # 打架计数器
        self.fall_counter = {}            # 摔倒计数器
        self.fall_hold_counter = {}       # 摔倒持续报警计数器
        self.fall_reset_counter = {}
        self.center_history = {}          # 中心坐标历史
        self.crowd_counter = {}           # 聚集计数器
        self.crowd_groups = []            # 聚集组
        self.run_counter = {}             # 奔跑计数器
        self.chase_counter = {}           # 追逐计数器
        self.chasing_pairs = set()        # 追逐对
        self.speed_smooth = {}            # 奔跑速度平滑队列
        
        print(f"[Tracker Reset] ✅ tracks 已清空: {self.tracks}")
        print(f"[Tracker Reset] ✅ next_id 已重置: {self.next_id}")
        print(f"[Tracker Reset] ✅ abnormal_ids 已清空: {self.abnormal_ids}")
        print(f"[Tracker Reset] ✅ 所有行为计数器已清空")
        print("="*80 + "\n")

    def _iou(self, bbox1, bbox2):
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        xx1 = max(x1_1, x1_2)
        yy1 = max(y1_1, y1_2)
        xx2 = min(x2_1, x2_2)
        yy2 = min(y2_1, y2_2)
        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)
        inter = w * h
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - inter + 1e-6
        return inter / union

    def _center_distance(self, c1, c2):
        return np.hypot(c1[0]-c2[0], c1[1]-c2[1])

    def _get_movement_direction(self, history):
        if len(history) < 2:
            return (0, 0)
        dirs = []
        for i in range(-1, -min(4, len(history)), -1):
            cx, cy, _ = history[i]
            px, py, _ = history[i-1]
            dx = cx - px
            dy = cy - py
            mag = np.hypot(dx, dy)
            if mag > 0:
                dirs.append((dx/mag, dy/mag))
        if len(dirs) == 0:
            return (0, 0)
        return np.mean(dirs, axis=0)

    def _is_follow_relationship(self, dir1, c1, dir2, c2):
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        mag = np.hypot(dx, dy)
        if mag == 0:
            return False
        dx /= mag
        dy /= mag
        dot1 = dx * dir1[0] + dy * dir1[1]
        dot2 = (-dx) * dir2[0] + (-dy) * dir2[1]
        return dot1 > self.config.CHASE_FOLLOW_THRESH and dot2 > self.config.CHASE_FOLLOW_THRESH

    def calc_upper_body_energy(self, track_id):
        if track_id not in self.tracks:
            return 0
        kpts = self.tracks[track_id]["kpts"]
        if not kpts or len(kpts) < 17:
            return 0
        l_shoulder = kpts[5] if kpts[5][2] > 0.5 else None
        r_shoulder = kpts[6] if kpts[6][2] > 0.5 else None
        l_wrist = kpts[9] if kpts[9][2] > 0.5 else None
        r_wrist = kpts[10] if kpts[10][2] > 0.5 else None
        energy = 0
        if l_shoulder and l_wrist:
            dist = np.hypot(l_wrist[0]-l_shoulder[0], l_wrist[1]-l_shoulder[1])
            if dist > self.config.ARM_SWING_THRESH:
                energy += dist
        if r_shoulder and r_wrist:
            dist = np.hypot(r_wrist[0]-r_shoulder[0], r_wrist[1]-r_shoulder[1])
            if dist > self.config.ARM_SWING_THRESH:
                energy += dist
        return energy

    def is_arm_swinging(self, track_id, frame_idx):
        if not self.config.ENABLE_FIGHT_DETECT:
            return False
        if track_id not in self.tracks:
            return False
        track = self.tracks[track_id]
        kpts = track["kpts"]
        if len(kpts) < 17:
            return False
        wrist_left = kpts[9] if kpts[9][2] > 0.5 else None
        wrist_right = kpts[10] if kpts[10][2] > 0.5 else None
        if not wrist_left and not wrist_right:
            return False
        curr_energy = self.calc_upper_body_energy(track_id)
        prev_energy = track.get("prev_energy", curr_energy)
        speed = abs(curr_energy - prev_energy)
        track["prev_energy"] = curr_energy
        if track_id not in self.swing_counter:
            self.swing_counter[track_id] = 0
        if speed > self.config.SWING_SPEED_THRESH and curr_energy > self.config.ARM_SWING_THRESH:
            self.swing_counter[track_id] += 1
            self.swing_counter[track_id] = min(self.swing_counter[track_id], self.config.REQUIRE_SWING_FRAMES + 2)
        else:
            self.swing_counter[track_id] = max(0, self.swing_counter[track_id] - 1)

        print(f"    [打架检测] ID={track_id}, Counter={self.swing_counter[track_id]}/{self.config.REQUIRE_SWING_FRAMES}, Energy={curr_energy:.2f}, Speed={speed:.2f}")
        
        if self.swing_counter[track_id] >= self.config.REQUIRE_SWING_FRAMES:
            self._record_abnormal(track_id, "Fighting", frame_idx, self.swing_counter[track_id])
            self.swing_counter[track_id] = 0 
            return True
        return False

    def _calc_trunk_angle(self, kpts):
        l_shoulder, r_shoulder = kpts[5], kpts[6]
        l_hip, r_hip = kpts[11], kpts[12]
        valid_shoulders = []
        valid_hips = []
        if l_shoulder[2] > 0.5: valid_shoulders.append(l_shoulder)
        if r_shoulder[2] > 0.5: valid_shoulders.append(r_shoulder)
        if l_hip[2] > 0.5: valid_hips.append(l_hip)
        if r_hip[2] > 0.5: valid_hips.append(r_hip)
        if len(valid_shoulders) == 0 or len(valid_hips) == 0:
            return 0
        shoulder_mid_x = np.mean([p[0] for p in valid_shoulders])
        shoulder_mid_y = np.mean([p[1] for p in valid_shoulders])
        hip_mid_x = np.mean([p[0] for p in valid_hips])
        hip_mid_y = np.mean([p[1] for p in valid_hips])
        dx = shoulder_mid_x - hip_mid_x
        dy = shoulder_mid_y - hip_mid_y
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0
        mag = np.hypot(dx, dy)
        dx /= mag
        dy /= mag
        vertical_vec_y = -1
        dot_product = dx * 0 + dy * vertical_vec_y
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        return max(0, min(90, angle_deg))

    def is_fallen(self, track_id, frame_idx):
        if not self.config.ENABLE_FALL_DETECT:
            return False
        if track_id not in self.tracks:
            return False

        if track_id not in self.fall_counter:
            self.fall_counter[track_id] = 0
        if track_id not in self.fall_hold_counter:
            self.fall_hold_counter[track_id] = 0

        track = self.tracks[track_id]
        kpts = track["kpts"]
        bbox = track["bbox"]

        if len(kpts) < 17 or len(bbox) < 4:
            return False

        if self.fall_hold_counter[track_id] > 0:
            self.fall_hold_counter[track_id] -= 1
            self._record_abnormal(track_id, "Fall Detected", frame_idx, self.fall_hold_counter[track_id])
            return True

        trunk_angle = self._calc_trunk_angle(kpts)
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        aspect_ratio = bbox_w / (bbox_h + 1e-6)

        is_angle_ok = trunk_angle > self.config.FALL_ANGLE_THRESH
        is_aspect_ok = aspect_ratio > self.config.FALL_ASPECT_RATIO_THRESH
        is_fall_shape = is_angle_ok or is_aspect_ok

        if is_fall_shape:
            self.fall_counter[track_id] += 1
        else:
            self.fall_counter[track_id] = 0
        
        print(f"    [摔倒检测] ID={track_id}, Counter={self.fall_counter[track_id]}/{self.config.FALL_TRIGGER_FRAMES}, Angle={trunk_angle:.1f}°, Ratio={aspect_ratio:.2f}")

        if self.fall_counter[track_id] >= self.config.FALL_TRIGGER_FRAMES:
            self.fall_hold_counter[track_id] = self.config.FALL_HOLD_FRAMES
            self.fall_counter[track_id] = 0
            self._record_abnormal(track_id, "Fall Detected", frame_idx, self.config.FALL_HOLD_FRAMES)
            return True

        return False

    def detect_crowd(self, frame_idx):
        if not self.config.ENABLE_CROWD_DETECT:
            self.crowd_groups = []
            return False
        self.crowd_groups = []
        if len(self.tracks) < self.config.CROWD_MIN_PERSONS:
            return False
        track_ids = list(self.tracks.keys())
        centers = [self.tracks[tid]["center"] for tid in track_ids]
        visited = [False] * len(track_ids)
        for i in range(len(track_ids)):
            if visited[i]:
                continue
            group = [track_ids[i]]
            queue = [i]
            visited[i] = True
            while queue:
                idx = queue.pop(0)
                for j in range(len(track_ids)):
                    if not visited[j] and self._center_distance(centers[idx], centers[j]) < self.config.CROWD_DIST_THRESH:
                        visited[j] = True
                        group.append(track_ids[j])
                        queue.append(j)
            if len(group) >= self.config.CROWD_MIN_PERSONS:
                xs = [self.tracks[tid]["bbox"][0] for tid in group] + [self.tracks[tid]["bbox"][2] for tid in group]
                ys = [self.tracks[tid]["bbox"][1] for tid in group] + [self.tracks[tid]["bbox"][3] for tid in group]
                cx = int(np.mean([self.tracks[tid]["center"][0] for tid in group]))
                cy = int(np.mean([self.tracks[tid]["center"][1] for tid in group]))
                xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
                group_w = xmax - xmin
                group_h = ymax - ymin
                aspect_ratio = group_w / (group_h + 1e-6)
                head_angles = []
                for tid in group:
                    if "head_angle" in self.tracks[tid]:
                        head_angles.append(self.tracks[tid]["head_angle"])
                angle_std = np.std(head_angles) if len(head_angles) > 1 else 0
                has_swing = any(self.swing_counter.get(tid, 0) > 0 for tid in group)
                is_queue = False
                if aspect_ratio > 2.5 and angle_std < 25:
                    is_queue = True
                if aspect_ratio > 2.0 and not has_swing:
                    is_queue = True
                if is_queue:
                    continue
                self.crowd_groups.append((len(group), (cx, cy), (xmin, ymin, xmax, ymax)))
                for tid in group:
                    if tid not in self.crowd_counter:
                        self.crowd_counter[tid] = 0
                    self.crowd_counter[tid] += 1
                    if self.crowd_counter[tid] >= self.config.CROWD_CONSEC_FRAMES:
                        self._record_abnormal(tid, f"Crowd Gathering({len(group)} persons)", frame_idx, len(group))
        return len(self.crowd_groups) > 0

    def is_running(self, track_id, frame_idx):
        if not self.config.ENABLE_RUN_CHASE_DETECT:
            return False
        if track_id not in self.tracks:
            return False
        history = self.tracks[track_id]["history"]
        if len(history) < 2:
            return False

        cx, cy, _ = history[-1]
        px, py, _ = history[-2]
        curr_dist = np.hypot(cx-px, cy-py)

        if track_id not in self.speed_smooth:
            self.speed_smooth[track_id] = deque(maxlen=self.config.RUN_SMOOTH_WINDOW)
        self.speed_smooth[track_id].append(curr_dist)
        smooth_speed = np.mean(self.speed_smooth[track_id])

        if track_id not in self.run_counter:
            self.run_counter[track_id] = 0
        if smooth_speed > self.config.RUN_SPEED_THRESH:
            self.run_counter[track_id] += 1
        else:
            self.run_counter[track_id] = max(0, self.run_counter[track_id]-1)
        
        print(f"    [奔跑检测] ID={track_id}, Counter={self.run_counter[track_id]}/{self.config.CONSECUTIVE_FRAMES}, Speed={smooth_speed:.2f}")

        if self.run_counter[track_id] >= self.config.CONSECUTIVE_FRAMES:
            self._record_abnormal(track_id, "Running", frame_idx, self.run_counter[track_id])
            return True
        return False

    def detect_chasing(self, frame_idx):
        if not self.config.ENABLE_RUN_CHASE_DETECT:
            self.chasing_pairs = set()
            self.chase_counter = {}
            return
        if len(self.tracks) < 2:
            self.chasing_pairs = set()
            self.chase_counter = {}
            return

        track_ids = list(self.tracks.keys())
        new_chasing_pairs = set()
        temp_chase_counter = {}
        running_status = {tid: self.is_running(tid, frame_idx) for tid in track_ids}

        for i in range(len(track_ids)):
            t1 = track_ids[i]
            if not running_status[t1]: continue
            if t1 not in self.tracks: continue
            c1 = self.tracks[t1]["center"]
            h1 = self.tracks[t1]["history"]
            if len(h1) < 2: continue
            dir1 = self._get_movement_direction(h1)

            for j in range(i+1, len(track_ids)):
                t2 = track_ids[j]
                if not running_status[t2]: continue
                if t2 not in self.tracks: continue
                c2 = self.tracks[t2]["center"]
                h2 = self.tracks[t2]["history"]
                if len(h2) < 2: continue
                dir2 = self._get_movement_direction(h2)

                dist_curr = self._center_distance(c1, c2)
                dist_prev = self._center_distance(h1[-2][:2], h2[-2][:2])
                approach_speed = dist_prev - dist_curr

                distance_ok = self.config.CHASE_MIN_DISTANCE < dist_curr < self.config.CHASE_MAX_DISTANCE
                direction_ok = True
                if self.config.CHASE_REQUIRE_SAME_DIRECTION:
                    dot_product = dir1[0]*dir2[0] + dir1[1]*dir2[1]
                    direction_ok = dot_product > 0.707
                approach_ok = approach_speed > self.config.CHASE_APPROACH_SPEED_THRESH
                follow_ok = True
                if self.config.CHASE_REQUIRE_FOLLOW:
                    follow_ok = self._is_follow_relationship(dir1, c1, dir2, c2) or self._is_follow_relationship(dir2, c2, dir1, c1)

                key = tuple(sorted((t1, t2)))
                if distance_ok and direction_ok and approach_ok and follow_ok:
                    temp_chase_counter[key] = self.chase_counter.get(key, 0) + 1
                    if temp_chase_counter[key] >= self.config.CHASE_CONSEC_FRAMES:
                        new_chasing_pairs.add(key)
                        self._record_abnormal(t1, f"Chasing(ID{t2})", frame_idx, temp_chase_counter[key])
                        self._record_abnormal(t2, "Chased", frame_idx, temp_chase_counter[key])

        self.chase_counter = temp_chase_counter
        self.chasing_pairs = new_chasing_pairs

    def update(self, boxes, all_kpts, frame_idx):
        updated_tracks = {}
        used = set()
        track_list = list(self.tracks.items())
        track_list.sort(key=lambda x: len(x[1]["history"]), reverse=True)

        for tid, track in track_list:
            best_score = 0
            best_idx = -1
            for i, (box, kpt) in enumerate(zip(boxes, all_kpts)):
                if i in used:
                    continue
                iou = self._iou(track["bbox"], box)
                if iou > best_score and iou > self.iou_thresh:
                    best_score = iou
                    best_idx = i

            if best_idx != -1:
                box = boxes[best_idx]
                kpt = all_kpts[best_idx]
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                nose = kpt[0]
                left_eye = kpt[1]
                right_eye = kpt[2]
                head_angle = 0
                if nose[2] > 0.5 and left_eye[2] > 0.5 and right_eye[2] > 0.5:
                    eye_mid_x = (left_eye[0] + right_eye[0]) / 2
                    eye_mid_y = (left_eye[1] + right_eye[1]) / 2
                    dx = nose[0] - eye_mid_x
                    dy = nose[1] - eye_mid_y
                    head_angle = np.degrees(np.arctan2(dx, dy))
                track["bbox"] = box
                track["kpts"] = kpt
                track["center"] = (cx, cy)
                track["head_angle"] = head_angle
                track["history"].append((cx, cy, head_angle))
                if tid not in self.center_history:
                    self.center_history[tid] = deque(maxlen=10)
                self.center_history[tid].append((cx, cy))
                track["lost"] = 0
                updated_tracks[tid] = track
                used.add(best_idx)

        for i, (box, kpt) in enumerate(zip(boxes, all_kpts)):
            if i in used:
                continue
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            nose = kpt[0]
            left_eye = kpt[1]
            right_eye = kpt[2]
            head_angle = 0
            if nose[2] > 0.5 and left_eye[2] > 0.5 and right_eye[2] > 0.5:
                eye_mid_x = (left_eye[0] + right_eye[0]) / 2
                eye_mid_y = (left_eye[1] + right_eye[1]) / 2
                dx = nose[0] - eye_mid_x
                dy = nose[1] - eye_mid_y
                head_angle = np.degrees(np.arctan2(dx, dy))
            updated_tracks[self.next_id] = {
                "bbox": box, "kpts": kpt, "center": (cx, cy),
                "head_angle": head_angle, "history": deque(maxlen=self.max_history),
                "lost": 0, "prev_energy": 0
            }
            updated_tracks[self.next_id]["history"].append((cx, cy, head_angle))
            self.center_history[self.next_id] = deque(maxlen=10)
            self.center_history[self.next_id].append((cx, cy))
            self.next_id += 1

        self.tracks = updated_tracks
        print(f"[Tracker Update] 跟踪更新完成，当前活动目标数: {len(self.tracks)}")
        return self.tracks

    def is_fast_moving(self, track_id, frame_idx):
        if not self.config.ENABLE_FAST_MOVE_DETECT:
            return False
        if track_id not in self.tracks:
            return False
        history = self.tracks[track_id]["history"]
        if len(history) < self.config.CONSECUTIVE_FRAMES + 1:
            return False
        fast_count = 0
        for i in range(1, len(history)):
            px, py, _ = history[i-1]
            cx, cy, _ = history[i]
            dist = np.sqrt((cx-px)**2 + (cy-py)**2)
            if dist > self.config.SPEED_THRESH:
                fast_count += 1
                if fast_count >= self.config.CONSECUTIVE_FRAMES:
                    self._record_abnormal(track_id, "Fast Moving", frame_idx, fast_count)
                    return True
            else:
                fast_count = max(0, fast_count-1)
        return False

    def is_looking_around(self, track_id, frame_idx):
        if not self.config.ENABLE_LOOK_AROUND_DETECT:
            return False
        if track_id not in self.tracks:
            return False
        history = self.tracks[track_id]["history"]
        if len(history) < self.config.CONSECUTIVE_FRAMES + 1:
            return False
        look_count = 0
        for i in range(1, len(history)):
            _, _, a1 = history[i-1]
            _, _, a2 = history[i]
            diff = abs(a2 - a1)
            diff = min(diff, 360 - diff)
            if diff > self.config.ANGLE_THRESH:
                look_count += 1
                if look_count >= self.config.CONSECUTIVE_FRAMES:
                    self._record_abnormal(track_id, "Looking Around", frame_idx, look_count)
                    return True
            else:
                look_count = max(0, look_count-1)
        return False

    def _record_abnormal(self, track_id, behavior_type, frame_idx, duration):
        print(f"\n" + "*"*60)
        print(f"[🔥 异常触发] ID={track_id}, 行为={behavior_type}, 帧号={frame_idx}, 持续={duration}")
        print("*"*60 + "\n")
        
        if track_id not in self.abnormal_history:
            self.abnormal_history[track_id] = []
        if not self.abnormal_history[track_id] or self.abnormal_history[track_id][-1]["behavior"] != behavior_type:
            self.abnormal_history[track_id].append({
                "frame_idx": frame_idx,
                "behavior": behavior_type,
                "duration": duration,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        self.abnormal_ids.add(track_id)

    def export_abnormal_log(self, save_path):
        clean_history = {}
        for tid, records in self.abnormal_history.items():
            if len(records) > 0:
                clean_history[str(tid)] = records
        log_data = {
            "process_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_abnormal_ids": len(clean_history),
            "abnormal_ids": list(clean_history.keys()),
            "abnormal_history": clean_history
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Abnormal behavior log saved to: {save_path}")

# ==================== 姿态模型辅助函数 ====================
def letterbox(img, new_shape=(640,640), color=(114,114,114)):
    shape = img.shape[:2]
    r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
    new_unpad = (int(round(shape[1]*r)), int(round(shape[0]*r)))
    dw, dh = new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
    dw_left = dw // 2
    dw_right = dw - dw_left
    dh_top = dh // 2
    dh_bottom = dh - dh_top
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, dh_top, dh_bottom, dw_left, dw_right, cv2.BORDER_CONSTANT, value=color)
    assert img.shape[0] == new_shape[0] and img.shape[1] == new_shape[1]
    return img, (dw_left, dh_top), r

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def non_max_suppression_pose(boxes, scores, iou_thresh=0.3):
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    indices = np.argsort(scores)[::-1]
    keep = []
    while len(indices) > 0:
        curr = indices[0]
        keep.append(curr)
        if len(indices) == 1:
            break
        x1, y1, x2, y2 = boxes[curr]
        rest_boxes = boxes[indices[1:]]
        xx1 = np.maximum(x1, rest_boxes[:, 0])
        yy1 = np.maximum(y1, rest_boxes[:, 1])
        xx2 = np.minimum(x2, rest_boxes[:, 2])
        yy2 = np.minimum(y2, rest_boxes[:, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_curr = (x2 - x1) * (y2 - y1)
        area_rest = (rest_boxes[:, 2] - rest_boxes[:, 0]) * (rest_boxes[:, 3] - rest_boxes[:, 1])
        union = area_curr + area_rest - inter
        iou = inter / (union + 1e-6)
        indices = indices[1:][iou < iou_thresh]
    return np.array(keep, dtype=np.int32)

def postprocess_pose(output, img_shape, pad, scale):
    if output is None:
        return [], [], []
    output = np.squeeze(output)
    if output.shape[0] == 56 and output.shape[1] == 8400:
        pred = output.T
    elif output.shape[0] == 8400 and output.shape[1] == 56:
        pred = output
    else:
        print(f"警告：模型输出维度异常 {output.shape}，跳过该帧")
        return [], [], []
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    obj_conf = sigmoid(pred[:, 4])
    kpt_data = pred[:, 5:]
    mask = obj_conf > 0.55
    if not np.any(mask):
        return [], [], []
    cx, cy, w, h, obj_conf, kpt_data = cx[mask], cy[mask], w[mask], h[mask], obj_conf[mask], kpt_data[mask]
    size_mask = (w > 12) & (h > 12)
    if not np.any(size_mask):
        return [], [], []
    cx, cy, w, h, obj_conf, kpt_data = cx[size_mask], cy[size_mask], w[size_mask], h[size_mask], obj_conf[size_mask], kpt_data[size_mask]
    left, top = pad
    x1 = (cx - w/2 - left) / scale
    y1 = (cy - h/2 - top) / scale
    x2 = (cx + w/2 - left) / scale
    y2 = (cy + h/2 - top) / scale
    x1 = np.clip(x1, 0, img_shape[1] - 1)
    y1 = np.clip(y1, 0, img_shape[0] - 1)
    x2 = np.clip(x2, 0, img_shape[1] - 1)
    y2 = np.clip(y2, 0, img_shape[0] - 1)
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.int32)
    keep = non_max_suppression_pose(boxes, obj_conf, 0.3)
    if len(keep) == 0:
        return [], [], []
    boxes = boxes[keep]
    obj_conf = obj_conf[keep]
    kpt_data = kpt_data[keep]
    all_kpts = []
    for d in kpt_data:
        kpts = []
        for i in range(17):
            kx = (d[i*3] - left) / scale
            ky = (d[i*3+1] - top) / scale
            kc = sigmoid(d[i*3+2])
            kpts.append((kx, ky, kc))
        all_kpts.append(kpts)
    return boxes, all_kpts, obj_conf

class PoseDetector:
    def __init__(self, config: SceneConfig = DEFAULT_CONFIG):
        self.pose_model_path = POSE_MODEL
        self.rknn_pose = None
        self.tracker = None
        self.frame_count = 0
        self.config = config

        print("🔹 Loading Pose RKNN model...")
        self.rknn_pose = RKNNLite()
        self.rknn_pose.load_rknn(self.pose_model_path)
        # NPU 3核全开
        self.rknn_pose.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        self.tracker = Tracker(config=self.config)
        print("✅ Pose detector initialized (NPU 3核)")

    def reset(self):
        print("\n" + "#"*80)
        print("[PoseDetector Reset] 🔥 检测到视频切换，执行完全重置...")
        # 🔥 核心修复：强制清空所有状态，帧计数归零
        self.frame_count = 0
        print(f"[PoseDetector Reset] ✅ frame_count 已重置为: {self.frame_count}")
        if self.tracker:
            self.tracker.reset()
        print("#"*80 + "\n")

    def detect_frame(self, frame):
        self.frame_count += 1
        print(f"\n[PoseDetector] 🔵 处理帧号: {self.frame_count}")
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pre_pose, pad_pose, scale_pose = letterbox(img_rgb)
        input_pose = np.expand_dims(img_pre_pose, axis=0).astype(np.float32)
        outputs_pose = self.rknn_pose.inference(inputs=[input_pose])
        pose_boxes, pose_kpts, pose_scores = postprocess_pose(outputs_pose[0], frame.shape[:2], pad_pose, scale_pose)
        
        abnormal_events = []
        if len(pose_boxes) > 0:
            print(f"[PoseDetector] 检测到 {len(pose_boxes)} 个人体框，开始跟踪更新...")
            tracks = self.tracker.update(pose_boxes, pose_kpts, self.frame_count)
            self.tracker.detect_crowd(self.frame_count)
            self.tracker.detect_chasing(self.frame_count)
            print(f"\n[DEBUG Pose] 帧号: {self.frame_count}, 跟踪到人数: {len(tracks)}")
            
            for tid, t in tracks.items():
                print(f"  -> 跟踪ID: {tid}, BBox: {t['bbox']}")
                if self.tracker.is_arm_swinging(tid, self.frame_count):
                    evt = {"type": "Fighting", "target_id": tid}
                    abnormal_events.append(evt)
                    print(f"[🔥 异常] 追加事件: {evt}")
                    
                if self.tracker.is_fallen(tid, self.frame_count):
                    evt = {"type": "Fall Detected", "target_id": tid}
                    abnormal_events.append(evt)
                    print(f"    [🔥 异常] 追加事件: {evt}")
                if self.tracker.is_running(tid, self.frame_count):
                    evt = {"type": "Running", "target_id": tid}
                    abnormal_events.append(evt)
                    print(f"    [🔥 异常] 追加事件: {evt}")
                if any(p[0]==tid for p in self.tracker.chasing_pairs):
                    evt = {"type": "Chasing", "target_id": tid}
                    abnormal_events.append(evt)
                    print(f"    [🔥 异常] 追加事件: {evt}")
                if any(p[1]==tid for p in self.tracker.chasing_pairs):
                    evt = {"type": "Chased", "target_id": tid}
                    abnormal_events.append(evt)
                    print(f"    [🔥 异常] 追加事件: {evt}")
        else:
            print(f"[PoseDetector] 本帧未检测到人体框")
                    
        print(f"\n[DEBUG Pose] 帧号: {self.frame_count}, 本帧生成的 abnormal_events: {abnormal_events}")            
        return frame, pose_boxes, pose_kpts, pose_scores, abnormal_events

    def release(self):
        if self.rknn_pose:
            self.rknn_pose.release()
        print("✅ Pose detector resources released.")