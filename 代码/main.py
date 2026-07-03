import os
import sys
import logging
from datetime import datetime
from calculate_alert import positive_window_manager
# 核心：强制OpenCV使用修复后的FFmpeg硬件解码
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_VIDEOIO_PRIORITY_BACKEND"] = "FFMPEG"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "flags;low_delay|discard-corrupt;1|rtsp_transport;tcp"

import threading
import time
import socket
import cv2
import numpy as np
from flask import Flask, Response
from queue import Queue, Empty
from scene_configs import VIDEO_TO_SCENE, DEFAULT_CONFIG
import tempfile
import json
import subprocess
import warnings
from llm_vision_analyzer import analyze_alarm_image

from models.pose import PoseDetector
from models.face_detector import FaceDetector
from models.audio_detector import AudioDetector

warnings.filterwarnings("ignore", category=RuntimeWarning)

from calculate_alert import calculate_final_severity, get_alert_level, window_manager
import re

# ====================== ✅ 按区域分文件+人物ID日志系统 ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "."))
LOG_DIR = os.path.join(PROJECT_ROOT, "event_logs")
PERF_LOG_FILE = os.path.join(LOG_DIR, "performance_metrics.jsonl")
ALARM_FRAME_DIR = os.path.join(PROJECT_ROOT, "alarm_frames")


os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ALARM_FRAME_DIR, exist_ok=True)




switch_perf_versions = set()
switch_perf_lock = threading.Lock()

logger_cache = {}
logger_lock = threading.Lock()
perf_lock = threading.Lock()

def save_alarm_frame(frame, area="unknown", alert_level="unknown"):
    try:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{area}_{alert_level}_{time_str}.jpg"
        img_path = os.path.join(ALARM_FRAME_DIR, filename)

        cv2.imwrite(img_path, frame)
        print(f"📸 [大模型Demo] 告警帧已保存: {img_path}")
        return img_path

    except Exception as e:
        print(f"⚠️ [大模型Demo] 保存告警帧失败: {e}")
        return None

def run_llm_analysis_async(
    image_path,
    action_text,
    face_text,
    audio_text,
    transcript,
    abnormal_sounds,
    positive_sounds,
    alert_level,
    final_severity,
    version,
):
    try:
        audio_desc = []

        if abnormal_sounds:
            audio_desc.append("检测到异常声音：" + "、".join(abnormal_sounds))

        if positive_sounds:
            audio_desc.append("检测到积极声音：" + "、".join(positive_sounds))

        if transcript:
            audio_desc.append("语音内容：" + transcript)

        if len(audio_desc) == 0:
            audio_prompt = audio_text or "无明显音频异常"
        else:
            audio_prompt = "；".join(audio_desc)

        llm_result = analyze_alarm_image(
            image_path,
            action_text=action_text,
            face_text=face_text,
            audio_text=audio_prompt,
            transcript=transcript,
            alert_level=alert_level,
            final_severity=final_severity,
        )

        print("🤖 [大模型Demo] 异步分析结果：")
        print(llm_result)

        send_tcp_message({
            "type": "llm_analysis",
            "llm_analysis": llm_result,
            "alarm_img_path": image_path,
            "version": version,
        })

    except Exception as e:
        print(f"⚠️ [大模型Demo] 异步分析失败: {e}")


def write_perf_metric(metric_type, **kwargs):
    """写入性能指标日志，每行一个 JSON，方便后续单独统计"""
    try:
        data = {
            "type": metric_type,
            "time": time.time(),
            "time_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            **kwargs,
        }

        with perf_lock:
            with open(PERF_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"⚠️ [性能日志] 写入失败: {e}")
        
        
def get_area_logger(area):
    """获取指定区域的日志器，自动创建对应的日志文件"""
    with logger_lock:
        if area in logger_cache:
            return logger_cache[area]

        logger = logging.getLogger(f"event_logger_{area}")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        log_file = os.path.join(LOG_DIR, f"{area}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "[%(asctime)s] 区域:%(area)s | ID:%(person_ids)s | 事件:%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger_cache[area] = logger
        return logger


def log_event(area, message, person_ids):
    """记录事件日志，自动按区域分文件，ID为检测到的人物ID组"""
    try:
        logger = get_area_logger(area)
        logger.info(message, extra={"area": area, "person_ids": person_ids})
    except Exception as e:
        print(f"⚠️ [日志系统] 记录失败: {e}")


# ====================== ✅ 全局常量统一定义 ======================
TARGET_FPS = 25
AUDIO_INTERVAL = 3
ALERT_COOLDOWN = 3
FRAME_QUEUE_MAXSIZE = 2
AUDIO_QUEUE_MAXSIZE = 3
MODAL_SEND_INTERVAL = 1.5

# ====================== ✅ RTSP 底部裁剪控制 ======================
# 只对 rtsp/rtmp/http/https 网络视频流生效；本地 mp4 不裁剪
RTSP_BOTTOM_CROP_ENABLED = True

# 从画面底部往上偏移多少像素开始裁剪
# 0 = 贴着最底部裁剪；数值越大，裁剪区域越往上移
RTSP_BOTTOM_CROP_Y_OFFSET_FROM_BOTTOM = 0

# 底部裁剪高度，单位像素
RTSP_BOTTOM_CROP_HEIGHT = 480

# 横向裁剪起点，单位像素
RTSP_BOTTOM_CROP_X = 0

# 横向裁剪宽度，单位像素；0 表示使用原始完整宽度
RTSP_BOTTOM_CROP_WIDTH = 0

ACTION_LATCH_SECONDS = 3.0
ABNORMAL_MODAL_SEND_INTERVAL = 0.5
action_latch_until = 0.0
action_latch_text = "正常"

audio_play_process = None
audio_play_lock = threading.Lock()

ALARM_SOUND_DIR = os.path.join(PROJECT_ROOT, "alarm_sounds")

ALARM_SOUND_MAP = {
    "打架": "打架.wav",
    "打闹": "打闹.wav",
    "摔倒": "奔跑摔倒.wav",
    "奔跑": "奔跑摔倒.wav",
    "攀爬": "攀爬.wav",
    "聚集": "聚集.wav",
    "人群聚集": "聚集.wav",
    "异常声音": "声音事件.wav",
}

ALARM_SOUND_COOLDOWN = 3.0
last_alarm_sound_time = 0.0
alarm_sound_lock = threading.Lock()

# ====================== ✅ 全局检测开关（随场景动态更新） ======================
ENABLE_AUDIO_EVENT_DETECT = True
ENABLE_AUDIO_SEMANTIC_DETECT = True
ENABLE_FACE_EMOTION_DETECT = True

audio_config_lock = threading.Lock()

# ====================== ✅ 模态状态缓存 ======================
last_modal_send_time = 0
last_action_abnormal = False
last_action_text = "正常"
last_face_abnormal = False
last_face_text = "正常"
last_face_positive = False
last_audio_abnormal = False
last_audio_text = "正常"
last_sound_positive = False
llm_face_label = "Neutral"

# ====================== ✅ 全局版本号控制 ======================
current_version = 0
version_lock = threading.Lock()

# ====================== ✅ 全局当前区域变量 ======================
current_area = "禁止聚集区"
area_lock = threading.Lock()


def get_current_version():
    with version_lock:
        return current_version


def empty_audio_status():
    return {
        "is_alert": False,
        "alert_msg": "无异常",
        "risk_level": 0,
        "risk_text": "正常",
        "transcript": "",
        "abnormal_sounds": [],
        "top5_sounds": [],
        "semantic_score": 0.5,
        "text_abnormal": False,
        "last_update": time.time(),
        "sound_abnormal": False,
        "positive_sounds": [],
        "sound_positive": False,
    }


def reset_modal_cache():
    """切换视频时重置后端模态缓存，避免状态不变导致不发送正常状态。"""
    global last_modal_send_time
    global last_action_abnormal, last_action_text
    global last_face_abnormal, last_face_text, last_face_positive
    global last_audio_abnormal, last_audio_text, last_sound_positive
    global action_latch_until, action_latch_text

    last_modal_send_time = 0
    last_action_abnormal = False
    last_action_text = "正常"
    last_face_abnormal = False
    last_face_text = "正常"
    last_face_positive = False
    last_audio_abnormal = False
    last_audio_text = "正常"
    last_sound_positive = False
    action_latch_until = 0.0
    action_latch_text = "正常"


def clear_queue(q, delete_files=False):
    while not q.empty():
        try:
            item = q.get_nowait()
            if delete_files and isinstance(item, tuple) and len(item) >= 2:
                seg_path = item[0]
                if isinstance(seg_path, str) and seg_path.endswith(".wav") and os.path.exists(seg_path):
                    try:
                        os.unlink(seg_path)
                    except Exception:
                        pass
        except Exception:
            pass


if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

VIDEO_MAP = {
    "走廊1": os.path.join(PROJECT_ROOT, "datasets/rel_fight_small.mp4"),
    "走廊2": os.path.join(PROJECT_ROOT, "datasets/rel_fight1.mp4"),
    "慢行区": os.path.join(PROJECT_ROOT, "datasets/rel_falldown.mp4"),
    "禁止聚集区": os.path.join(PROJECT_ROOT, "datasets/rel_gether.mp4"),
    "东南门": os.path.join(PROJECT_ROOT, "datasets/rel_climb.mp4"),
    #"东南门": "rtsp://admin:bhjdipc2019@10.9.1.112:554/Streaming/Channels/101",
    "走廊3": os.path.join(PROJECT_ROOT, "datasets/rel_break_slap.mp4"),
}


def is_stream_source(path):
    return isinstance(path, str) and path.lower().startswith(
        ("rtsp://", "rtmp://", "http://", "https://")
    )


def source_available(path):
    return is_stream_source(path) or os.path.exists(path)


def source_name(path):
    if is_stream_source(path):
        return "RTSP视频流"
    return os.path.basename(path)


def crop_rtsp_bottom_frame(frame, source_path):
    """
    只对 RTSP/网络视频流裁剪掉底部区域。
    本地 mp4 不裁剪。

    注意：
    这里不是“保留底部区域”，而是“删除底部区域，保留上方画面”。

    裁剪逻辑：
    y2 = 原图高度 - RTSP_BOTTOM_CROP_Y_OFFSET_FROM_BOTTOM
    y1 = y2 - RTSP_BOTTOM_CROP_HEIGHT
    y1:y2 是要裁掉的底部区域，最终返回 frame[:y1, :]

    可调变量：
    - RTSP_BOTTOM_CROP_ENABLED：是否启用裁剪
    - RTSP_BOTTOM_CROP_Y_OFFSET_FROM_BOTTOM：从底部往上偏移多少像素
    - RTSP_BOTTOM_CROP_HEIGHT：裁剪高度
    - RTSP_BOTTOM_CROP_X / RTSP_BOTTOM_CROP_WIDTH：保留兼容旧配置，但删除底部时不再按横向局部裁剪
    """
    if frame is None:
        return frame

    if not is_stream_source(source_path):
        return frame

    if not RTSP_BOTTOM_CROP_ENABLED:
        return frame

    h, w = frame.shape[:2]

    crop_h = int(RTSP_BOTTOM_CROP_HEIGHT)
    y_offset = int(RTSP_BOTTOM_CROP_Y_OFFSET_FROM_BOTTOM)

    crop_h = max(1, min(crop_h, h))
    y_offset = max(0, min(y_offset, h - 1))

    # 要裁掉的底部区域范围：[y1, y2)
    y2 = h - y_offset
    y1 = max(0, y2 - crop_h)

    # 如果计算异常，或者裁剪高度覆盖整张图，为了避免输出空画面，直接返回原图
    if y1 <= 0 or y2 <= y1:
        return frame

    # 返回底部裁剪区域以上的画面：也就是裁掉底部，保留上方
    return frame[:y1, :].copy()


BOARD_IP = "0.0.0.0"
STREAM_PORT = 5000
VM_IP = "127.0.0.1"
VM_PORT = 9999
CONTROL_PORT = 10000

INITIAL_VIDEO_PATH = VIDEO_MAP["禁止聚集区"]

if not source_available(INITIAL_VIDEO_PATH):
    print(f"❌ 初始视频不存在: {INITIAL_VIDEO_PATH}")
    sys.exit(1)

raw_frame_queue = Queue(maxsize=FRAME_QUEUE_MAXSIZE)
pose_result_queue = Queue(maxsize=FRAME_QUEUE_MAXSIZE)
face_result_queue = Queue(maxsize=FRAME_QUEUE_MAXSIZE)
frame_queue = Queue(maxsize=FRAME_QUEUE_MAXSIZE)
tracker_queue = Queue(maxsize=1)
audio_task_queue = Queue(maxsize=AUDIO_QUEUE_MAXSIZE)

current_video_path = INITIAL_VIDEO_PATH
shutdown_event = threading.Event()

switch_state = 0
switch_lock = threading.Lock()
target_video_path = None

audio_alert_status = empty_audio_status()

tcp_clients = []
tcp_clients_lock = threading.Lock()

app = Flask(__name__)


def generate_frames():
    while not shutdown_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
            ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        except Empty:
            continue


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    return "<h1>系统运行中</h1><p>请访问 <a href='/video_feed'>/video_feed</a> 查看视频流</p>"


def start_stream_server():
    print("✅ Flask 推流线程启动")
    app.run(host=BOARD_IP, port=STREAM_PORT, threaded=True, debug=False, use_reloader=False)


def clear_tcp_send_buffer():
    global tcp_clients

    with tcp_clients_lock:
        for client in tcp_clients:
            try:
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                client.sendall(b"")
            except Exception:
                pass

    time.sleep(0.05)


def send_tcp_message(data):
    global tcp_clients, current_version

    data = dict(data)

    if "version" not in data:
        with version_lock:
            data["version"] = current_version

    message = json.dumps(data, ensure_ascii=False).encode("utf-8") + b"\n"

    with tcp_clients_lock:
        clients = tcp_clients.copy()

    disconnected = []

    for client in clients:
        try:
            client.sendall(message)
        except Exception:
            disconnected.append(client)

    if disconnected:
        with tcp_clients_lock:
            for client in disconnected:
                if client in tcp_clients:
                    tcp_clients.remove(client)
                try:
                    client.close()
                except Exception:
                    pass


def send_alert(text):
    try:
        send_tcp_message(json.loads(text))
    except Exception as e:
        print(f"⚠️ [后端] 发送报警失败: {e}")



def get_alarm_sound_key(action_text, sound_abnormal=False):
    if sound_abnormal:
        return "异常声音"

    if not action_text or action_text == "正常":
        return None

    if "打架" in action_text:
        return "打架"
    if "打闹" in action_text:
        return "打闹"
    if "摔倒" in action_text:
        return "摔倒"
    if "奔跑" in action_text:
        return "奔跑"
    if "攀爬" in action_text:
        return "攀爬"
    if "聚集" in action_text:
        return "聚集"

    return None


def play_alarm_sound(sound_key):
    global last_alarm_sound_time

    if not sound_key:
        return

    wav_name = ALARM_SOUND_MAP.get(sound_key)
    if not wav_name:
        return

    wav_path = os.path.join(ALARM_SOUND_DIR, wav_name)

    if not os.path.exists(wav_path):
        print(f"⚠️ [报警音频] 文件不存在: {wav_path}")
        return

    now = time.time()

    with alarm_sound_lock:
        if now - last_alarm_sound_time < ALARM_SOUND_COOLDOWN:
            return

        last_alarm_sound_time = now

        try:
            subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"🔊 [报警音频] 播放: {wav_name}")

        except Exception as e:
            print(f"⚠️ [报警音频] 播放失败: {e}")

def send_risk_status(final_severity, alert_level, version=None):
    try:
        data = {
            "type": "risk_status",
            "final_severity": round(final_severity, 2),
            "alert_level": alert_level,
        }

        if version is not None:
            data["version"] = version

        send_tcp_message(data)

    except Exception as e:
        print(f"⚠️ [后端] 发送风险状态失败: {e}")


def send_audio_status_to_qt(audio_status, version=None):
    try:
        data = {
            "type": "audio_status",
            "is_alert": audio_status["is_alert"],
            "alert_msg": audio_status["alert_msg"],
            "transcript": audio_status["transcript"],
            "top5_sounds": audio_status["top5_sounds"],
            "abnormal_sounds": audio_status["abnormal_sounds"],
            "sound_abnormal": audio_status["sound_abnormal"],
            "positive_sounds": audio_status["positive_sounds"],
            "sound_positive": audio_status["sound_positive"],
        }

        if version is not None:
            data["version"] = version

        send_tcp_message(data)

        print(
            f"📤 [后端] 发送音频状态: version={data.get('version')}, "
            f"sound_abnormal={data['sound_abnormal']}, "
            f"abnormal_sounds={data['abnormal_sounds']}, "
            f"sound_positive={data['sound_positive']}"
        )

    except Exception as e:
        print(f"⚠️ [后端] 发送音频状态失败: {e}")


def send_modal_status_to_qt(
    action_abnormal,
    action_text,
    face_abnormal,
    face_text,
    audio_abnormal,
    audio_text,
    face_positive=False,
    sound_positive=False,
    version=None,
):
    try:
        data = {
            "type": "modal_status",
            "action_abnormal": action_abnormal,
            "action_text": action_text,
            "face_abnormal": face_abnormal,
            "face_text": face_text,
            "audio_abnormal": audio_abnormal,
            "audio_text": audio_text,
            "face_positive": face_positive,
            "sound_positive": sound_positive,
        }

        if version is not None:
            data["version"] = version

        send_tcp_message(data)

    except Exception as e:
        print(f"⚠️ [后端] 发送模态状态失败: {e}")


def send_video_reset_to_qt(version=None):
    try:
        data = {
            "type": "video_loop_reset",
            "clear_alert": True,
        }

        if version is not None:
            data["version"] = version

        send_tcp_message(data)

    except Exception as e:
        print(f"⚠️ [后端] 发送重置消息失败: {e}")


def audio_worker(audio_detector):
    global audio_alert_status, current_version, current_area
    global ENABLE_AUDIO_EVENT_DETECT, ENABLE_AUDIO_SEMANTIC_DETECT

    print("✅ 音频检测线程启动")

    last_reset_sent_version = -1

    while not shutdown_event.is_set():
        seg_path = None

        try:
            with switch_lock:
                local_switch_state = switch_state

            if local_switch_state != 0:
                current_ver = get_current_version()
                clear_queue(audio_task_queue, delete_files=True)
                audio_alert_status = empty_audio_status()

                if last_reset_sent_version != current_ver:
                    print("🔄 [音频线程] 检测到切换/非运行状态，清空音频任务队列")
                    send_audio_status_to_qt(audio_alert_status, version=current_ver)
                    last_reset_sent_version = current_ver

                time.sleep(0.2)
                continue

            last_reset_sent_version = -1

            seg_path, task_version = audio_task_queue.get(timeout=0.5)

            print(f"\n🔍 [音频线程] 收到音频任务: 文件={os.path.basename(seg_path)}, 任务版本={task_version}")

            with switch_lock, version_lock:
                if switch_state != 0 or task_version != current_version:
                    print(
                        f"❌ [音频线程] 检测前丢弃旧音频任务: "
                        f"任务版本={task_version}, 当前版本={current_version}, state={switch_state}"
                    )
                    if os.path.exists(seg_path):
                        os.unlink(seg_path)
                    continue

            if not os.path.exists(seg_path):
                print(f"⚠️ [音频线程] 文件不存在，跳过: {os.path.basename(seg_path)}")
                continue

            if os.path.getsize(seg_path) < 100:
                print(f"⚠️ [音频线程] 文件为空，跳过: {os.path.basename(seg_path)}")
                if os.path.exists(seg_path):
                    os.unlink(seg_path)
                continue
            
            
            audio_total_start = time.perf_counter()
            
            res = audio_detector.detect(seg_path)

            audio_total_end = time.perf_counter()

            write_perf_metric(
                "audio_branch_inference",
                 version=task_version,
                 file=os.path.basename(seg_path),
                 total_cost_ms=round((audio_total_end - audio_total_start) * 1000, 2),
                 yamnet_cost_ms=res.get("yamnet_cost_ms"),
                 whisper_cost_ms=res.get("whisper_cost_ms"),
                 bert_cost_ms=res.get("bert_cost_ms"),
                 audio_model_cost_ms=res.get("audio_total_cost_ms"),
                 sound_abnormal=res.get("sound_abnormal", False),
                 text_abnormal=res.get("text_abnormal", False),
                 transcript_len=len(res.get("transcript", "")),)


            print(f"🔍 [音频线程] 检测完成: 文件={os.path.basename(seg_path)}")
            print(f"   code={res.get('code', -1)}, sound_abnormal={res.get('sound_abnormal', False)}")
            print(f"   abnormal_sounds={res.get('abnormal_sounds', [])}, top5={res.get('top5_sounds', [])}")
            print(f"   sound_positive={res.get('sound_positive', False)}, positive_sounds={res.get('positive_sounds', [])}")

            with switch_lock, version_lock:
                if switch_state != 0 or task_version != current_version:
                    print(
                        f"❌ [音频线程] 检测后丢弃旧音频结果: "
                        f"任务版本={task_version}, 当前版本={current_version}, state={switch_state}"
                    )
                    if os.path.exists(seg_path):
                        os.unlink(seg_path)
                    continue

            with audio_config_lock:
                event_enabled = ENABLE_AUDIO_EVENT_DETECT
                semantic_enabled = ENABLE_AUDIO_SEMANTIC_DETECT

            if not event_enabled:
                res["sound_abnormal"] = False
                res["abnormal_sounds"] = []
                res["positive_sounds"] = []
                res["sound_positive"] = False
                res["is_alert"] = res.get("text_abnormal", False)

            if not semantic_enabled:
                res["text_abnormal"] = False
                res["transcript"] = ""
                res["semantic_score"] = 0.5
                res["is_alert"] = res.get("sound_abnormal", False)

            if not event_enabled and not semantic_enabled:
                res["is_alert"] = False
                res["alert_msg"] = "无异常"
                res["risk_level"] = 0

            if res.get("code") == 200:
                audio_alert_status = {
                    "is_alert": res.get("is_alert", False),
                    "alert_msg": res.get("alert_msg", "无异常"),
                    "risk_level": res.get("risk_level", 0),
                    "risk_text": res.get("risk_level_text", "正常"),
                    "transcript": res.get("transcript", ""),
                    "abnormal_sounds": res.get("abnormal_sounds", []),
                    "top5_sounds": res.get("top5_sounds", []),
                    "semantic_score": res.get("semantic_score", 0.5),
                    "text_abnormal": res.get("text_abnormal", False),
                    "last_update": time.time(),
                    "sound_abnormal": res.get("sound_abnormal", False),
                    "positive_sounds": res.get("positive_sounds", []),
                    "sound_positive": res.get("sound_positive", False),
                }

                print(
                    f"✅ [音频线程] 结果有效: version={task_version}, "
                    f"sound_abnormal={audio_alert_status['sound_abnormal']}, "
                    f"abnormal_sounds={audio_alert_status['abnormal_sounds']}, "
                    f"sound_positive={audio_alert_status['sound_positive']}"
                )

                send_audio_status_to_qt(audio_alert_status, version=task_version)

                if audio_alert_status["sound_abnormal"] and len(audio_alert_status["abnormal_sounds"]) > 0:
                    with area_lock:
                        current_area_val = current_area

                    sounds_str = "、".join(audio_alert_status["abnormal_sounds"])
                    log_message = f"检测到危险声音: {sounds_str}"
                    log_event(current_area_val, log_message, "全局")
                    print(f"📝 [日志] 已记录音频事件: {log_message}")

            if os.path.exists(seg_path):
                os.unlink(seg_path)
                print(f"🗑️ [音频线程] 已删除临时文件: {os.path.basename(seg_path)}")

        except Empty:
            continue

        except Exception as e:
            print(f"⚠️ [音频线程] 异常: {e}")
            try:
                if seg_path and os.path.exists(seg_path):
                    os.unlink(seg_path)
            except Exception:
                pass
            time.sleep(0.1)


def audio_play_thread():
    global audio_play_process

    print("✅ 音频播放线程启动")

    while not shutdown_event.is_set():
        try:
            with switch_lock:
                play_video_path = current_video_path
                system_state = switch_state

            if system_state != 0:
                with audio_play_lock:
                    if audio_play_process is not None:
                        try:
                            audio_play_process.terminate()
                            audio_play_process.wait(timeout=0.5)
                            if audio_play_process.poll() is None:
                                audio_play_process.kill()
                        except Exception:
                            pass
                        audio_play_process = None

                time.sleep(0.1)
                continue

            if is_stream_source(play_video_path):
                time.sleep(0.1)
                continue

            if audio_play_process is None and os.path.exists(play_video_path):
                with audio_play_lock:
                    cmd = [
                        "ffplay",
                        "-vn",
                        "-nodisp",
                        "-autoexit",
                        "-sync",
                        "audio",
                        "-i",
                        play_video_path,
                        "-loop",
                        "0",
                    ]

                    audio_play_process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

                    print(f"🔊 音频已切换：{os.path.basename(play_video_path)}")

            time.sleep(0.1)

        except Exception as e:
            print(f"⚠️ [音频播放] 异常：{e}")
            time.sleep(0.1)

    with audio_play_lock:
        if audio_play_process is not None:
            try:
                audio_play_process.terminate()
                audio_play_process.wait(timeout=0.5)
                if audio_play_process.poll() is None:
                    audio_play_process.kill()
            except Exception:
                pass


def control_listen():
    print("✅ TCP控制监听线程启动")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", CONTROL_PORT))
    server_socket.listen(5)
    server_socket.settimeout(1.0)

    while not shutdown_event.is_set():
        try:
            client_socket, addr = server_socket.accept()

            print(f"✅ [TCP] 新控制连接来自: {addr}")

            client_socket.settimeout(1.0)

            threading.Thread(
                target=handle_control_client,
                args=(client_socket, addr),
                daemon=True,
            ).start()

        except socket.timeout:
            continue

        except Exception as e:
            print(f"⚠️ [控制线程] 接受连接异常: {e}")
            continue

    server_socket.close()


def handle_control_client(client_socket, addr):
    global switch_state, target_video_path, current_video_path, current_version
    global current_area, audio_alert_status, audio_play_process

    buffer = b""

    while not shutdown_event.is_set():
        try:
            data = client_socket.recv(1024)

            if not data:
                break

            buffer += data

            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                area = line.decode().strip()
                ts = time.strftime("%H:%M:%S")

                print(f"\n🎮 [{ts}] 收到控制指令: {area}")
               

                
                if area not in VIDEO_MAP:
                    print(f"❌ 无效指令：{area}")
                    continue

                target_path = VIDEO_MAP[area]

                if not source_available(target_path):
                    print("❌ 切换失败：视频源不存在")
                    continue

                with switch_lock:
                    if switch_state != 0:
                        print(f"⚠️ [{ts}] [CTL] 系统正忙 (state={switch_state})，忽略此次指令")
                        continue

                    with version_lock:
                        old_version = current_version
                        current_version += 1
                        new_version = current_version
                    with switch_perf_lock:
                        switch_perf_versions.add(new_version)

                    write_perf_metric(
                         "video_switch_start",
                          area=area,
                          version=new_version,
                          source=source_name(target_path),)

                    print(f"🔄 [{ts}] [CTL] 版本号更新: {old_version} -> {new_version}")

                    target_video_path = target_path
                    current_video_path = target_path
                    switch_state = 1

                    with area_lock:
                        current_area = area

                    print(f"   ✅ [CTL] 当前区域已更新为: {area}")

                    with audio_play_lock:
                        if audio_play_process is not None:
                            try:
                                audio_play_process.terminate()
                                audio_play_process.wait(timeout=0.5)
                                if audio_play_process.poll() is None:
                                    audio_play_process.kill()
                            except Exception:
                                pass
                            audio_play_process = None

                    print("   ✅ [CTL] 旧音频播放进程已强制终止")

                    clear_queue(raw_frame_queue)
                    clear_queue(pose_result_queue)
                    clear_queue(face_result_queue)
                    clear_queue(frame_queue)
                    clear_queue(audio_task_queue, delete_files=True)

                    audio_alert_status = empty_audio_status()
                    window_manager.reset()
                    reset_modal_cache()

                    print("   ✅ [CTL] 队列、风险窗口、后端缓存、音频状态已重置")

                clear_tcp_send_buffer()

                for _ in range(5):
                    send_video_reset_to_qt(new_version)
                    time.sleep(0.02)

                send_modal_status_to_qt(
                    False,
                    "正常",
                    False,
                    "正常",
                    False,
                    "正常",
                    False,
                    False,
                    version=new_version,
                )

                for _ in range(3):
                    send_audio_status_to_qt(audio_alert_status, version=new_version)
                    time.sleep(0.01)

                send_risk_status(0.0, "正常", version=new_version)

                print("   ✅ [CTL] 已发送重置状态给Qt")

        except socket.timeout:
            continue

        except Exception as e:
            print(f"⚠️ [控制客户端] 异常: {e}")
            break

    print(f"❌ [TCP] 控制连接断开: {addr}")
    client_socket.close()


def tcp_data_server():
    global tcp_clients

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", VM_PORT))
    server_socket.listen(5)
    server_socket.settimeout(1.0)

    print(f"✅ [TCP] 数据服务器启动，端口{VM_PORT}")

    while not shutdown_event.is_set():
        try:
            client_socket, addr = server_socket.accept()
            client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            print(f"✅ [TCP] 新数据客户端连接: {addr}")

            with tcp_clients_lock:
                tcp_clients.append(client_socket)

        except socket.timeout:
            continue

        except Exception as e:
            print(f"⚠️ [TCP数据服务器] 异常: {e}")
            continue

    with tcp_clients_lock:
        for client in tcp_clients:
            try:
                client.close()
            except Exception:
                pass

        tcp_clients.clear()

    server_socket.close()


def pose_thread():
    global switch_state, target_video_path, current_video_path
    global ENABLE_AUDIO_EVENT_DETECT, ENABLE_AUDIO_SEMANTIC_DETECT
    global ENABLE_FACE_EMOTION_DETECT

    print("✅ 姿态检测线程启动")
    pose_detector = None

    print("🔄 [姿态线程] 正在预加载初始模型...")

    config = VIDEO_TO_SCENE.get(os.path.basename(INITIAL_VIDEO_PATH), DEFAULT_CONFIG)

    with audio_config_lock:
        ENABLE_AUDIO_EVENT_DETECT = config.ENABLE_AUDIO_EVENT_DETECT
        ENABLE_AUDIO_SEMANTIC_DETECT = config.ENABLE_AUDIO_SEMANTIC_DETECT
        ENABLE_FACE_EMOTION_DETECT = config.ENABLE_FACE_EMOTION_DETECT

    print(
        f"✅ [姿态线程] 初始配置: "
        f"音频事件={ENABLE_AUDIO_EVENT_DETECT}, "
        f"语义检测={ENABLE_AUDIO_SEMANTIC_DETECT}, "
        f"表情识别={ENABLE_FACE_EMOTION_DETECT}"
    )

    pose_detector = PoseDetector(config=config)
    tracker_queue.put(pose_detector.tracker)

    print("✅ [姿态线程] 初始模型加载完成！")

    while not shutdown_event.is_set():
        try:
            do_switch = False
            local_path = None

            with switch_lock:
                if switch_state == 1:
                    do_switch = True
                    local_path = target_video_path

            if do_switch:
                ts = time.strftime("%H:%M:%S")
                print(f"🔄 [{ts}] [姿态] 检测到切换信号，开始执行...")

                print("   ⏳ [姿态] 正在释放旧模型...")

                if pose_detector is not None:
                    pose_detector.reset()
                    pose_detector.release()
                    pose_detector = None

                print("   ✅ [姿态] 旧模型释放完毕")

                print("   ⏳ [姿态] 正在清空队列...")

                clear_queue(raw_frame_queue)
                clear_queue(pose_result_queue)
                clear_queue(face_result_queue)
                clear_queue(tracker_queue)

                print("   ✅ [姿态] 队列清空完毕")

                print(f"   ⏳ [姿态] 正在加载新模型: {source_name(local_path)}...")

                config = VIDEO_TO_SCENE.get(os.path.basename(local_path), DEFAULT_CONFIG)

                with audio_config_lock:
                    ENABLE_AUDIO_EVENT_DETECT = config.ENABLE_AUDIO_EVENT_DETECT
                    ENABLE_AUDIO_SEMANTIC_DETECT = config.ENABLE_AUDIO_SEMANTIC_DETECT
                    ENABLE_FACE_EMOTION_DETECT = config.ENABLE_FACE_EMOTION_DETECT

                print(
                    f"   ✅ [姿态] 配置更新: "
                    f"音频事件={ENABLE_AUDIO_EVENT_DETECT}, "
                    f"语义检测={ENABLE_AUDIO_SEMANTIC_DETECT}, "
                    f"表情识别={ENABLE_FACE_EMOTION_DETECT}"
                )

                pose_detector = PoseDetector(config=config)
                tracker_queue.put(pose_detector.tracker)

                print("   ✅ [姿态] 新模型加载完成！")

                with switch_lock:
                    switch_state = 2
                    print(f"🔄 [{ts}] [姿态] 处理完毕，状态置为 2 (通知主线程重启视频)")

                continue

            try:
                frame = raw_frame_queue.get(timeout=0.01)

                if pose_detector is None:
                    continue

                frame_version = frame.get("version", -1)

                with switch_lock, version_lock:
                    if switch_state != 0 or frame_version != current_version:
                        continue

                if frame["count"] % 2 == 0:
                    infer_start = time.time()
                    
                    # ✅ detect_frame 当前返回 6 个值：img, boxes, kpts, scores, events, tracks_snapshot
                    # tracks_snapshot 是 Tracker 当前活跃轨迹快照，用于后续日志 ID 兜底，避免从 boxes 下标猜 ID。
                    img, boxes, kpts, scores, events, tracks_snapshot = pose_detector.detect_frame(frame["img"])
                    
                    infer_end = time.time()
                    
                    write_perf_metric(
                       "vision_inference",
                        version=frame_version,
                        frame_count=frame["count"],
                        cost_ms=round((infer_end - infer_start) * 1000, 2),
                        has_event=len(events) > 0,
                          )

                    with switch_lock, version_lock:
                        if switch_state != 0 or frame_version != current_version:
                            continue

                    pose_result_queue.put(
                        {
                            "img": img,
                            "boxes": boxes,
                            "kpts": kpts,
                            "scores": scores,
                            "events": events,
                            "tracks_snapshot": tracks_snapshot,
                            "count": frame["count"],
                            "version": frame_version,
                        }
                    )

                    if tracker_queue.empty():
                        tracker_queue.put(pose_detector.tracker)

            except Empty:
                continue

        except Exception as e:
            print(f"⚠️ [姿态线程] 异常: {e}")
            time.sleep(0.1)

    if pose_detector:
        pose_detector.release()


def face_thread():
    global ENABLE_FACE_EMOTION_DETECT

    print("✅ 人脸检测线程启动")

    face_detector = None

    while not shutdown_event.is_set():
        try:
            with switch_lock:
                local_switch_state = switch_state

            if local_switch_state != 0:
                if face_detector:
                    face_detector.reset()
                    face_detector.release()
                    face_detector = None

                clear_queue(pose_result_queue)
                clear_queue(face_result_queue)

                time.sleep(0.1)
                continue

            try:
                data = pose_result_queue.get(timeout=0.01)
                data_version = data.get("version", -1)

                with switch_lock, version_lock:
                    if switch_state != 0 or data_version != current_version:
                        continue

                with audio_config_lock:
                    face_enabled = ENABLE_FACE_EMOTION_DETECT

                if not face_enabled:
                    if face_detector:
                        face_detector.reset()
                        face_detector.release()
                        face_detector = None
                        print("🔒 [人脸线程] 表情识别关闭，已释放人脸模型")

                    data["faces"] = {}

                    with switch_lock, version_lock:
                        if switch_state != 0 or data_version != current_version:
                            continue

                    face_result_queue.put(data)
                    continue

                if not face_detector:
                    face_detector = FaceDetector()

                if data["count"] % 6 == 0:
                    img, faces = face_detector.detect_frame(data["img"])

                    with switch_lock, version_lock:
                        if switch_state != 0 or data_version != current_version:
                            continue

                    data["img"] = img
                    data["faces"] = faces

                with switch_lock, version_lock:
                    if switch_state != 0 or data_version != current_version:
                        continue

                face_result_queue.put(data)

            except Empty:
                continue

        except Exception as e:
            print(f"⚠️ [人脸线程] 异常: {e}")
            time.sleep(0.1)

    if face_detector:
        face_detector.release()


def main_thread():
    global switch_state, target_video_path, current_video_path, current_version
    global last_modal_send_time, last_action_abnormal, last_action_text
    global last_face_abnormal, last_face_text, last_face_positive
    global last_audio_abnormal, last_audio_text, last_sound_positive
    global action_latch_until, action_latch_text
    global current_area
    global ENABLE_FACE_EMOTION_DETECT

    print("✅ 主线程（视频读取）启动")

    cap = None
    cap_version = -1
    cap_path = None
    frame_interval = 1.0 / TARGET_FPS

    while not shutdown_event.is_set():
        try:
            if cap is None or not cap.isOpened():
                cap_path = current_video_path

                with version_lock:
                    cap_version = current_version

                print(f"   ⏳ [主线程] 正在打开视频: {source_name(cap_path)}...")

                cap = cv2.VideoCapture(cap_path, cv2.CAP_FFMPEG)

                if is_stream_source(cap_path):
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if not cap.isOpened():
                    print("   ❌ [主线程] 无法打开视频，5秒后重试...")
                    time.sleep(5)
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS)

                if fps <= 0:
                    fps = TARGET_FPS

                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                print(
                    f"🎥 视频信息: version={cap_version}, "
                    f"实际帧率={fps:.2f}fps, 分辨率={w}x{h}"
                )

                frame_interval = 1.0 / fps
                frame_count = 0
                last_audio_time = 0
                last_render = None
                last_alert = 0

            while not shutdown_event.is_set():
                start_time = time.time()

                with switch_lock:
                    local_state = switch_state

                if local_state == 2:
                    ts = time.strftime("%H:%M:%S")
                    print(f"\n🔄 [{ts}] [主线程] 检测到状态 2，正在重启视频源...")

                    if cap is not None and cap.isOpened():
                        cap.release()

                    print("   ✅ [主线程] 旧 VideoCapture 释放完毕")
                    print("   ⏳ [主线程] 正在清空所有队列...")

                    clear_queue(raw_frame_queue)
                    clear_queue(pose_result_queue)
                    clear_queue(face_result_queue)
                    clear_queue(frame_queue)
                    clear_queue(audio_task_queue, delete_files=True)

                    window_manager.reset()
                    reset_modal_cache()

                    print("   ✅ [主线程] 所有队列、报警窗口、缓存已清空")

                    frame_count = 0
                    last_audio_time = 0
                    last_render = None
                    last_alert = 0

                    cap_path = current_video_path

                    with version_lock:
                        cap_version = current_version
                        print(f"   ✅ [主线程] 当前有效版本号: {cap_version}")

                    cap = cv2.VideoCapture(cap_path, cv2.CAP_FFMPEG)

                    if is_stream_source(cap_path):
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS)

                        if fps <= 0:
                            fps = TARGET_FPS

                        frame_interval = 1.0 / fps

                        print(f"   ✅ [主线程] 新视频打开成功: {source_name(cap_path)}")
                    else:
                        print("   ❌ [主线程] 新视频打开失败！")
                        break

                    with switch_lock:
                        switch_state = 0

                    send_modal_status_to_qt(
                        False,
                        "正常",
                        False,
                        "正常",
                        False,
                        "正常",
                        False,
                        False,
                        version=cap_version,
                    )

                    send_audio_status_to_qt(empty_audio_status(), version=cap_version)
                    send_risk_status(0.0, "正常", version=cap_version)

                    print(f"🔄 [{ts}] [主线程] 视频重启完成，状态置为 0 (恢复正常流)")
                    
                    with switch_perf_lock:
                        should_log_switch_done = cap_version in switch_perf_versions
                        if should_log_switch_done:
                            switch_perf_versions.remove(cap_version)     
                    
                    if should_log_switch_done:
                        write_perf_metric(
                          "video_switch_done",
                           version=cap_version,
                           source=source_name(cap_path),)                    

                    time.sleep(0.3)
                    continue

                if local_state != 0:
                    time.sleep(0.1)
                    continue

                with version_lock:
                    version_mismatch = cap_version != current_version
                    current_ver_snapshot = current_version

                if version_mismatch:
                    print(
                        f"❌ [主线程] cap版本过期，等待切换流程: "
                        f"cap_version={cap_version}, current_version={current_ver_snapshot}"
                    )
                    time.sleep(0.05)
                    continue

                ret, frame = cap.read()

                if ret:
                    frame = crop_rtsp_bottom_frame(frame, cap_path)

                if not ret:
                    if is_stream_source(cap_path):
                        print("⚠️ [主线程] RTSP读取失败或断流，准备重连...")

                        if cap is not None and cap.isOpened():
                            cap.release()

                        cap = None

                        clear_queue(raw_frame_queue)
                        clear_queue(pose_result_queue)
                        clear_queue(face_result_queue)
                        clear_queue(frame_queue)
                        clear_queue(audio_task_queue, delete_files=True)

                        time.sleep(1)
                        break

                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_count = 0
                        last_render = None

                        with switch_lock:
                            if switch_state == 0:
                                target_video_path = current_video_path
                                switch_state = 1
                                window_manager.reset()

                        time.sleep(0.1)
                        continue

                frame_count += 1
                frame_version = cap_version

                with switch_lock, version_lock:
                    if switch_state != 0 or frame_version != current_version:
                        continue

                if last_render is None:
                    last_render = frame.copy()

                    if frame_queue.empty():
                        frame_queue.put(last_render)

                if raw_frame_queue.full():
                    clear_queue(raw_frame_queue)

                raw_frame_queue.put(
                    {
                        "img": frame.copy(),
                        "count": frame_count,
                        "version": frame_version,
                    }
                )

                now = time.time()

                with audio_config_lock:
                    audio_detect_enabled = (
                        ENABLE_AUDIO_EVENT_DETECT or ENABLE_AUDIO_SEMANTIC_DETECT
                    ) and (not is_stream_source(cap_path))

                if audio_detect_enabled and now - last_audio_time >= AUDIO_INTERVAL and audio_task_queue.qsize() < 3:
                    last_audio_time = now
                    task_version = cap_version
                    audio_video_path = cap_path

                    with switch_lock, version_lock:
                        if switch_state != 0 or task_version != current_version:
                            continue

                    current_pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                    start_sec = max(0.0, current_pos_ms / 1000.0 - 0.5)

                    cmd = [
                        "ffmpeg",
                        "-ss",
                        f"{start_sec:.2f}",
                        "-i",
                        audio_video_path,
                        "-t",
                        "3.0",
                        "-y",
                        "-v",
                        "quiet",
                        "-vn",
                        "-acodec",
                        "pcm_s16le",
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        tmp,
                    ]

                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    with switch_lock, version_lock:
                        if switch_state != 0 or task_version != current_version:
                            if os.path.exists(tmp):
                                os.unlink(tmp)
                            continue

                    audio_task_queue.put((tmp, task_version))

                    print(
                        f"🔊 [主线程] 音频任务入队: 文件={os.path.basename(tmp)}, "
                        f"版本号={task_version}, 队列大小={audio_task_queue.qsize()}"
                    )

                try:
                    res = face_result_queue.get(timeout=0.01)
                    res_version = res.get("version", -1)

                    with switch_lock, version_lock:
                        if switch_state != 0 or res_version != current_version:
                            print(
                                f"❌ [主线程] 丢弃旧视觉结果: "
                                f"res_version={res_version}, current_version={current_version}, state={switch_state}"
                            )
                            continue

                    from utils.visualizer import draw_pose, draw_face_emotion

                    tracker = None

                    if not tracker_queue.empty():
                        tracker = tracker_queue.get()
                        tracker_queue.put(tracker)

                    img = draw_pose(
                        res["img"],
                        res.get("boxes", []),
                        res.get("kpts", []),
                        res.get("scores", []),
                        tracker,
                        False,
                        (0, 0, 0, 0),
                        res.get("events", []),
                    )

                    if "faces" in res:
                        img = draw_face_emotion(img, res["faces"])

                    last_render = img.copy()

                    if not frame_queue.full():
                        frame_queue.put(img)

                    # ====================== ✅ 动作状态 ======================
                    events = res.get("events", [])
                    raw_action_abnormal = len(events) > 0
                    current_time_for_action = time.time()
                    event_detect_time = None

                    if raw_action_abnormal:
                        event_detect_time = time.time()
                        event_name_map = {
                            "Fighting": "打架",
                            "Fall Detected": "摔倒",
                            "Running": "奔跑",
                            "Climbing Detected": "攀爬",
                            "Crowd Gathering": "人群聚集",
                            "Chasing": "追逐",
                        }

                        unique_event_types = set()
                        unique_person_ids = set()

                        for event in events:
                            try:
                                event_type = "未知异常"
                                event_ids = []

                                if isinstance(event, dict):
                                    event_type = str(event.get("type", "未知异常"))

                                    track_id = event.get("track_id")
                                    track_ids = event.get("track_ids", [])
                                    target_id = event.get("target_id")

                                    # ✅ 单人事件：Fighting/Fall/Running/Climbing
                                    if track_id is not None:
                                        event_ids.append(track_id)

                                    # ✅ 多人事件：Crowd Gathering/Chasing
                                    if isinstance(track_ids, (list, tuple, set)):
                                        event_ids.extend(track_ids)

                                    # ✅ 兼容旧字段
                                    if target_id is not None:
                                        event_ids.append(target_id)

                                elif isinstance(event, (list, tuple, np.ndarray)) and len(event) >= 2:
                                    event_type = str(event[0])
                                    event_ids.append(event[1])

                                else:
                                    event_type = str(event)

                                # ✅ 事件类型中文化
                                if event_type.startswith("Crowd Gathering"):
                                    match = re.search(r"\((\d+) persons\)", event_type)
                                    if match:
                                        count = match.group(1)
                                        unique_event_types.add(f"聚集({count}人)")
                                    else:
                                        unique_event_types.add("聚集")
                                else:
                                    unique_event_types.add(event_name_map.get(event_type, event_type))

                                # ✅ 人物 ID 只从 track_id / track_ids / target_id 取
                                # ❌ 不再从 boxes、box[4]、检测框下标猜 ID，避免人物 ID 混乱。
                                for tid in event_ids:
                                    try:
                                        tid = int(tid)
                                        if tid > 0:
                                            unique_person_ids.add(tid)
                                    except Exception:
                                        continue

                            except Exception as e:
                                print(f"⚠️ [主线程] 解析事件异常: {e}, 事件内容: {event}")
                                continue

                        # ✅ 如果事件本身没有带 ID，就用 Tracker 活跃轨迹快照兜底
                        # 这里仍然只用 Tracker 的 ID，不使用检测框下标。
                        if not unique_person_ids:
                            tracks_snapshot = res.get("tracks_snapshot", {})

                            if isinstance(tracks_snapshot, dict):
                                for tid, track_info in tracks_snapshot.items():
                                    try:
                                        tid = int(tid)
                                        if tid <= 0:
                                            continue

                                        if isinstance(track_info, dict):
                                            lost = int(track_info.get("lost", 0))
                                            if lost == 0:
                                                unique_person_ids.add(tid)
                                        else:
                                            unique_person_ids.add(tid)
                                    except Exception:
                                        continue

                        detected_action_text = "、".join(sorted(unique_event_types)) if unique_event_types else "异常行为"

                        if unique_person_ids:
                            person_ids_str = "、".join([f"ID{tid}" for tid in sorted(unique_person_ids)])
                        else:
                            person_ids_str = "未知"

                        action_latch_until = current_time_for_action + ACTION_LATCH_SECONDS
                        action_latch_text = detected_action_text

                        with area_lock:
                            current_area_val = current_area

                        log_message = f"检测到异常行为: {detected_action_text}"
                        log_event(current_area_val, log_message, person_ids_str)

                        print(f"📝 [日志] 已记录动作事件: {log_message} | 涉及人物: {person_ids_str}")

                    if current_time_for_action < action_latch_until:
                        action_abnormal = True
                        action_text = action_latch_text
                    else:
                        action_abnormal = False
                        action_text = "正常"

                    # ====================== ✅ 人脸表情 ======================
                    face_abnormal = False
                    face_text = "正常"
                    face_positive = False
                    face_confidence_val = 0.2
                    face_data = res.get("faces", {})

                    face_name_map = {
                        "Angry": "愤怒",
                        "Happy": "开心",
                        "Neutral": "正常",
                        "正常": "正常",
                    }

                    normal_face_labels = {"Neutral", "正常", "neutral", "None", "", None}
                    positive_face_labels = {"Happy", "开心", "happy"}
                    abnormal_face_labels = {"Angry", "愤怒"}

                    try:
                        detected_faces = []

                        if isinstance(face_data, dict) and len(face_data) > 0:
                            for face_id, face_info in face_data.items():
                                if not isinstance(face_info, dict):
                                    continue

                                label = face_info.get("label", "Neutral")

                                if label is None:
                                    label = "Neutral"

                                label = str(label).strip()

                                try:
                                    score = float(face_info.get("score", face_info.get("raw_score", 0.0)))
                                except Exception:
                                    score = 0.0

                                score = max(0.0, min(score, 1.0))

                                detected_faces.append(
                                    {
                                        "label": label,
                                        "score": score,
                                    }
                                )

                        abnormal_labels = []
                        abnormal_scores = []
                        has_positive_face = False

                        for item in detected_faces:
                            label = item.get("label", "Neutral")
                            score = item.get("score", 0.0)

                            if label in positive_face_labels:
                                has_positive_face = True
                                continue

                            if label in normal_face_labels:
                                continue

                            if label in abnormal_face_labels:
                                abnormal_labels.append(label)
                                abnormal_scores.append(score)
                                continue

                            print(f"⚠️ [主线程] 未知表情标签，默认不报警: {label}")

                        if len(abnormal_labels) > 0:
                            face_abnormal = True
                            llm_face_label = "Angry"

                            abnormal_texts = []

                            for label in abnormal_labels:
                                abnormal_texts.append(face_name_map.get(label, label))

                            face_text = "、".join(sorted(set(abnormal_texts)))
                            face_positive = False

                            if len(abnormal_scores) > 0:
                                face_confidence_val = max(abnormal_scores)
                            else:
                                face_confidence_val = 0.65

                        else:
                            face_abnormal = False
                            face_text = "正常"
                            face_positive = has_positive_face
                            face_confidence_val = 0.2
                            
                            if has_positive_face:
                                llm_face_label = "Happy"
                            else:
                                llm_face_label = "Neutral"

                    except Exception as e:
                        print(f"⚠️ [主线程] 解析人脸表情异常: {e}")
                        face_abnormal = False
                        face_text = "正常"
                        face_positive = False
                        face_confidence_val = 0.2
                        llm_face_label = "Neutral"

                    with audio_config_lock:
                        face_detect_enabled = ENABLE_FACE_EMOTION_DETECT

                    if not face_detect_enabled:
                        face_abnormal = False
                        face_text = "正常"
                        face_positive = False
                        face_confidence_val = 0.2
                        llm_face_label = "Neutral"

                    # ====================== ✅ 音频状态 ======================
                    with switch_lock:
                        if switch_state != 0:
                            text_abnormal = False
                            sound_abnormal = False
                            audio_abnormal = False
                            sound_positive = False
                        else:
                            text_abnormal = audio_alert_status.get("text_abnormal", False)
                            sound_abnormal = audio_alert_status.get("sound_abnormal", False)
                            audio_abnormal = text_abnormal or sound_abnormal
                            sound_positive = audio_alert_status.get("sound_positive", False)

                    audio_text = "报警中" if audio_abnormal else "正常"

                    with audio_config_lock:
                        audio_detect_enabled = ENABLE_AUDIO_EVENT_DETECT or ENABLE_AUDIO_SEMANTIC_DETECT

                    if not audio_detect_enabled:
                        audio_abnormal = False
                        audio_text = "正常"
                        sound_positive = False

                    current_time = time.time()

                    state_changed = (
                        action_abnormal != last_action_abnormal
                        or action_text != last_action_text
                        or face_abnormal != last_face_abnormal
                        or face_text != last_face_text
                        or face_positive != last_face_positive
                        or audio_abnormal != last_audio_abnormal
                        or audio_text != last_audio_text
                        or sound_positive != last_sound_positive
                    )

                    has_abnormal_now = action_abnormal or face_abnormal or audio_abnormal

                    if (
                        has_abnormal_now
                        and current_time - last_modal_send_time >= ABNORMAL_MODAL_SEND_INTERVAL
                    ) or (
                        state_changed
                        and current_time - last_modal_send_time >= MODAL_SEND_INTERVAL
                    ) or (
                        current_time - last_modal_send_time >= MODAL_SEND_INTERVAL * 2
                    ):
                        send_modal_status_to_qt(
                            action_abnormal,
                            action_text,
                            face_abnormal,
                            face_text,
                            audio_abnormal,
                            audio_text,
                            face_positive,
                            sound_positive,
                            version=res_version,
                        )

                        last_modal_send_time = current_time
                        last_action_abnormal = action_abnormal
                        last_action_text = action_text
                        last_face_abnormal = face_abnormal
                        last_face_text = face_text
                        last_face_positive = face_positive
                        last_audio_abnormal = audio_abnormal
                        last_audio_text = audio_text
                        last_sound_positive = sound_positive

                    with switch_lock, version_lock:
                        if switch_state != 0 or res_version != current_version:
                            continue

                    # ====================== ✅ 风险计算 ======================
                    action_abnormal_val = action_abnormal
                    face_abnormal_val = face_abnormal

                    with audio_config_lock:
                        event_enabled = ENABLE_AUDIO_EVENT_DETECT
                        semantic_enabled = ENABLE_AUDIO_SEMANTIC_DETECT
                        face_detect_enabled = ENABLE_FACE_EMOTION_DETECT

                    text_abnormal_val = (
                        audio_alert_status.get("text_abnormal", False)
                        if semantic_enabled
                        else False
                    )

                    semantic_score_val = (
                        audio_alert_status.get("semantic_score", 0.5)
                        if semantic_enabled
                        else 0.5
                    )

                    sound_abnormal_val = (
                        audio_alert_status.get("sound_abnormal", False)
                        if event_enabled
                        else False
                    )

                    sound_positive_val = (
                        audio_alert_status.get("sound_positive", False)
                        if event_enabled
                        else False
                    )

                    action_score = 0.8 if action_abnormal_val else 0.2

                    if face_abnormal_val:
                        face_score = max(0.65, min(face_confidence_val, 0.8))
                    else:
                        face_score = 0.2

                    final_severity, window_mode_count, (action_win, face_win, txt_win) = calculate_final_severity(
                        action_score,
                        face_score,
                        semantic_score_val,
                        action_abnormal_val,
                        face_abnormal_val,
                        text_abnormal_val,
                        sound_abnormal_val,
                        face_positive,
                        sound_positive_val,
                        face_detect_enabled,
                        semantic_enabled,
                        event_enabled,
                    )

                    alert_level = get_alert_level(
                        final_severity,
                        window_mode_count,
                        face_positive,
                        sound_positive_val,
                        sound_abnormal_val,
                        event_enabled,
                    )

                    send_risk_status(final_severity, alert_level, version=res_version)

                    trigger_list = []

                    if action_win:
                        trigger_list.append("动作")

                    if face_win:
                        trigger_list.append("表情")

                    if txt_win:
                        trigger_list.append("语义")

                    if sound_abnormal_val:
                        trigger_list.append("异常声音")

                    trigger_modal = "、".join(trigger_list) if trigger_list else "无"

                    alert_parts = []

                    if action_abnormal_val:
                        alert_parts.append(action_text)
                    elif action_win:
                        alert_parts.append("动作异常")

                    if face_abnormal_val:
                        alert_parts.append(f"表情异常:{face_text}")
                    elif face_win:
                        alert_parts.append("表情异常")

                    if text_abnormal_val:
                        alert_parts.append(audio_alert_status.get("alert_msg", "语义异常"))
                    elif txt_win:
                        alert_parts.append("语义异常")

                    if sound_abnormal_val:
                        abnormal_sounds = audio_alert_status.get("abnormal_sounds", [])

                        if abnormal_sounds:
                            alert_parts.append("异常声音:" + "、".join(abnormal_sounds))
                        else:
                            alert_parts.append(audio_alert_status.get("alert_msg", "异常声音"))

                    has_risk_trigger = (
                        action_abnormal_val
                        or face_abnormal_val
                        or text_abnormal_val
                        or sound_abnormal_val
                        or action_win
                        or face_win
                        or txt_win
                    )

                    if time.time() - last_alert > ALERT_COOLDOWN and final_severity >= 0.2:
                        if not has_risk_trigger:
                            continue

                        with switch_lock, version_lock:
                            if switch_state != 0 or res_version != current_version:
                                continue

                        alert_text = "、".join(alert_parts) if alert_parts else "风险异常"
                        ts = time.strftime("%Y-%m-%d %H:%M:%S")

                        alert_json = {
                            "type": "behavior_alert",
                            "alert_text": alert_text,
                            "final_severity": round(final_severity, 2),
                            "trigger_modal": trigger_modal,
                            "alert_level": alert_level,
                            "timestamp": ts,
                            "version": res_version,
                        }

                        alert_send_time = time.time()

                        if event_detect_time is not None:
                            alert_delay_ms = round((alert_send_time - event_detect_time) * 1000, 2)
                        else:
                            alert_delay_ms = None

                        write_perf_metric(
                            "alert_latency",
                             version=res_version,
                             alert_text=alert_text,
                             final_severity=round(final_severity, 2),
                             alert_level=alert_level,
                             delay_ms=alert_delay_ms,)

                        
                        with area_lock:
                            current_area_val = current_area
                        alarm_img_path = save_alarm_frame(
                                   img,
                                    area=current_area_val,
                                alert_level=alert_level,
                                )
                        if alarm_img_path:
                           alert_json["alarm_img_path"] = alarm_img_path 
                           
                           if alert_level in ["低风险", "中风险", "高风险"]:
                               alert_json["llm_analysis"] = "AI辅助研判生成中..."
                               threading.Thread(
                                   target=run_llm_analysis_async,
                                   args=(
                                      alarm_img_path,
                                        action_text,
                                         llm_face_label,
                                        audio_text,
                                        audio_alert_status.get("transcript", ""),
                                         audio_alert_status.get("abnormal_sounds", []),
                                          audio_alert_status.get("positive_sounds", []),
                                       alert_level,
                                       final_severity,
                                        res_version,
                                   ),
                                  daemon=True,).start()
                        else:
                            alert_json["llm_analysis"] = "低风险事件，未触发AI辅助研判"    
                        send_alert(json.dumps(alert_json, ensure_ascii=False))
                        
                        sound_key = get_alarm_sound_key(
                             action_text,
                             sound_abnormal=sound_abnormal_val,)
                        if (
                              "打架" in action_text
                              and positive_window_manager.is_active()):
                             sound_key = "打闹"

                        play_alarm_sound(sound_key)     


                        last_alert = time.time()

                except Empty:
                    if last_render is not None and not frame_queue.full():
                        frame_queue.put(last_render)

                elapsed = time.time() - start_time

                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

        except Exception as e:
            print(f"❌ [主线程] 致命异常: {e}")
            import traceback

            traceback.print_exc()
            print("   ⏳ 5秒后自动重启主线程...")

            if cap is not None and cap.isOpened():
                cap.release()

            cap = None

            clear_queue(raw_frame_queue)
            clear_queue(pose_result_queue)
            clear_queue(face_result_queue)
            clear_queue(frame_queue)
            clear_queue(audio_task_queue, delete_files=True)

            time.sleep(5)


if __name__ == "__main__":
    print("=" * 50)
    print("🚀 系统启动中...")
    print("=" * 50)

    audio_detector = AudioDetector(PROJECT_ROOT)

    print("✅ 音频模型加载完成")

    threading.Thread(target=start_stream_server, daemon=True).start()
    threading.Thread(target=tcp_data_server, daemon=True).start()
    threading.Thread(target=control_listen, daemon=True).start()
    threading.Thread(target=pose_thread, daemon=True).start()
    threading.Thread(target=face_thread, daemon=True).start()
    threading.Thread(target=main_thread, daemon=True).start()
    threading.Thread(target=audio_worker, args=(audio_detector,), daemon=True).start()
    threading.Thread(target=audio_play_thread, daemon=True).start()

    print("✅ 系统运行中")
    print(f"✅ 推流地址: http://192.168.137.53:{STREAM_PORT}/video_feed")
    print(f"✅ 日志目录: {LOG_DIR}")
    print(f"✅ 各区域日志文件: {[f'{area}.log' for area in VIDEO_MAP.keys()]}")
    print("✅ 日志格式: [时间] 区域:XXX | ID:ID1、ID2 | 事件:XXX")

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 正在关闭系统...")
        
        shutdown_event.set()
        
        try:
            audio_detector.release()
        except Exception:
            pass

        time.sleep(1)
        os._exit(0)
        audio_detector.release()
        print("✅ 系统已退出")



