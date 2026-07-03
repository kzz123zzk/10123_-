import os
import sys
import time
import socket
import threading
import cv2
import numpy as np
import requests
import io
import json
from PIL import Image
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

os.environ["QT_LOGGING_RULES"] = "*=false"
os.environ["QT_DEBUG_PLUGINS"] = "0"

BOARD_STREAM_URL = "http://192.168.137.53:5000/video_feed"
PROCESSED_VIDEO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "result_final.mp4"
)

BOARD_IP = "127.0.0.1"
DATA_PORT = 9999
CONTROL_PORT = 10000


class VideoStreamThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.running = True
        self.session = requests.Session()
        self.setPriority(QThread.LowestPriority)

    def run(self):
        print(f"   📡 [流线程] 开始连接: {self.url}")
        frame_counter = 0

        while self.running:
            try:
                response = self.session.get(self.url, stream=True, timeout=5)

                if response.status_code != 200:
                    self.msleep(1000)
                    continue

                buffer = b""

                for chunk in response.iter_content(chunk_size=8192):
                    if not self.running:
                        break

                    buffer += chunk
                    a = buffer.find(b"\xff\xd8")
                    b = buffer.find(b"\xff\xd9")

                    if a != -1 and b != -1 and b > a:
                        jpg_data = buffer[a: b + 2]
                        buffer = buffer[b + 2:]

                        frame_counter += 1

                        if frame_counter % 3 != 0:
                            continue

                        img = Image.open(io.BytesIO(jpg_data))
                        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        self.frame_signal.emit(frame)

            except Exception:
                self.msleep(1000)

        print("   🛑 [流线程] 已停止")

    def stop(self):
        self.running = False

        try:
            self.session.close()
        except Exception:
            pass

        self.wait()


def send_switch_cmd(area):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect((BOARD_IP, CONTROL_PORT))
        sock.sendall((area + "\n").encode("utf-8"))
        sock.close()

    except Exception as e:
        print(f"⚠️ 发送控制指令失败: {e}")


class GradientProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(0, 100)
        self.setValue(0)
        self.setTextVisible(False)

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.fillRect(self.rect(), QColor("#222222"))

        value = self.value()

        if value > 0:
            w = self.width()
            h = self.height()
            fill_width = int(w * value / 100)

            gradient = QLinearGradient(0, 0, fill_width, 0)
            gradient.setColorAt(0.0, QColor(0, 204, 0))
            gradient.setColorAt(0.3, QColor(153, 204, 0))
            gradient.setColorAt(0.5, QColor(255, 204, 0))
            gradient.setColorAt(0.7, QColor(255, 153, 0))
            gradient.setColorAt(1.0, QColor(255, 0, 0))

            painter.fillRect(
                2,
                2,
                max(0, fill_width - 4),
                h - 4,
                gradient,
            )

        pen = QPen(QColor("#444444"), 1)
        painter.setPen(pen)
        painter.drawRoundedRect(
            self.rect().adjusted(1, 1, -1, -1),
            4,
            4,
        )


class AlertUI(QMainWindow):
    msg_signal = pyqtSignal(dict)
    audio_signal = pyqtSignal(dict)
    modal_signal = pyqtSignal(dict)
    reset_signal = pyqtSignal(int)
    risk_signal = pyqtSignal(dict)
    llm_signal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()

        self.video_area = None
        self.alert_status_label = None
        self.alert_time_label = None
        self.risk_level_label = None
        self.risk_progress_bar = None

        self.video_label = None
        self.switch_overlay = None
        self.switch_overlay_label = None

        self.action_label = None
        self.face_label = None
        self.audio_label = None
        self.alert_list_widget = None
        self.audio_sound_list = None
        self.abnormal_sound_list = None
        self.audio_event_label = None
        self.audio_transcript_label = None
        self.setWindowTitle("校园安全行为异常检测系统")
        QTimer.singleShot(
            100,
            lambda: self.setWindowTitle("校园安全行为异常检测系统"))
        self.setGeometry(100, 100, 1400, 750)

        self.event_mapping = {
            "Fighting": "打架斗殴",
            "Fall Detected": "人员摔倒",
            "Running": "快速奔跑",
            "Chasing": "追逐打闹",
            "Crowd Gathering": "人群聚集",
        }

        self.alert_history = []

        self.is_resetting = False
        self.switch_lock = False
        self.ignore_updates = False
        self.switching_in_progress = False

        self.switch_timestamp = 0.0

        self.current_action_abnormal = False
        self.current_face_abnormal = False
        self.current_audio_abnormal = False

        self.alarm_locked_until = 0
        self.alarm_lock_seconds = 3

        self.all_normal_start_time = 0
        self.need_wait_normal_seconds = 3

        self.ignore_audio_until = 0

        self.th = None
        self.pending_timer = None

        self.tcp_socket = None
        self.current_valid_version = 0

        # ====================== 风险等级显示平滑参数 ======================
        self.display_severity = 0.0
        self.last_risk_level = "正常"
        self.last_level_change_time = 0.0
        self.level_hold_seconds = 2.0

        self.cached_audio_status = {
            "sound_abnormal": False,
            "abnormal_sounds": [],
            "sound_positive": False,
            "positive_sounds": [],
        }

        self.initUI()

        self.msg_signal.connect(self.update_alert_info)
        self.audio_signal.connect(self.update_audio_panel)
        self.modal_signal.connect(self.update_modal_status)
        self.reset_signal.connect(self.reset_all_status)
        self.risk_signal.connect(self.update_risk_status)
        self.llm_signal.connect(self.update_llm_analysis)

        QTimer.singleShot(500, lambda: self.switch_video("走廊1"))

    def initUI(self):
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        self.main_layout = QVBoxLayout(self.main_widget)

        btn_layout = QHBoxLayout()

        area_list = [
            "走廊1",
            "走廊2",
            "慢行区",
            "禁止聚集区",
            "东南门",
            "走廊3",
        ]

        for name in area_list:
            btn = QPushButton(name)
            btn.setStyleSheet(
                "background-color:#1E88E5;"
                "color:white;"
                "padding:10px 20px;"
                "font-size:14px;"
            )
            btn.clicked.connect(lambda checked, n=name: self.switch_video(n))
            btn_layout.addWidget(btn)

        self.main_layout.addLayout(btn_layout)

        self.video_container_layout = QHBoxLayout()
        self.main_layout.addLayout(self.video_container_layout)

    def check_and_update_version(self, msg_version):
        if msg_version > self.current_valid_version:
            print("\n⚠️ [Qt] 检测到新版本消息，自动触发状态重置")
            print(
                f"   旧版本: {self.current_valid_version}, "
                f"新版本: {msg_version}"
            )
            self.reset_all_status(msg_version)
            return True

        return False

    def reset_all_status(self, new_version=0):
        self.is_resetting = True
        self.alert_history.clear()

        print(f"\n🔄 [Qt] 开始重置所有状态，新版本号: {new_version}")

        if new_version > 0:
            self.current_valid_version = new_version
            print(
                f"   ✅ [Qt] 当前有效版本号已更新为: "
                f"{self.current_valid_version}"
            )

        self.switch_timestamp = time.time()
        print(f"   ✅ [Qt] 切换时间戳: {time.ctime(self.switch_timestamp)}")

        self.cached_audio_status = {
            "sound_abnormal": False,
            "abnormal_sounds": [],
            "sound_positive": False,
            "positive_sounds": [],
        }
        print("   ✅ [Qt] 音频缓存状态已重置")

        if self.alert_list_widget is not None:
            self.alert_list_widget.clear()
        if hasattr(self, "llm_analysis_text") and self.llm_analysis_text is not None:
            self.llm_analysis_text.setPlainText("暂无AI研判结果")
        if self.alert_status_label is not None:
            self.alert_status_label.setText("✅ 正常")
            self.alert_status_label.setStyleSheet(
                "color:white;"
                "font-size:16px;"
            )

        if self.alert_time_label is not None:
            self.alert_time_label.setText("")

        if self.risk_level_label is not None:
            self.risk_level_label.setText("正常（风险值:0.00）")
            self.risk_level_label.setStyleSheet(
                "color:white;"
                "font-size:15px;"
            )

        if self.risk_progress_bar is not None:
            self.risk_progress_bar.setValue(0)

        self.current_action_abnormal = False
        self.current_face_abnormal = False
        self.current_audio_abnormal = False

        self.alarm_locked_until = 0
        self.all_normal_start_time = 0

        # ====================== 重置风险平滑缓存 ======================
        self.display_severity = 0.0
        self.last_risk_level = "正常"
        self.last_level_change_time = time.time()

        if self.action_label is not None:
            self.action_label.setText("✅ 正常")
            self.action_label.setStyleSheet(
                "color:#0c0;"
                "font-size:15px;"
            )

        if self.face_label is not None:
            self.face_label.setText("✅ 正常")
            self.face_label.setStyleSheet(
                "color:#0c0;"
                "font-size:15px;"
            )

        if self.audio_label is not None:
            self.audio_label.setText("✅ 正常")
            self.audio_label.setStyleSheet(
                "color:#0c0;"
                "font-size:15px;"
            )

        if self.audio_sound_list is not None:
            self.audio_sound_list.clear()

        if self.audio_event_label is not None:
            self.audio_event_label.setText("无")
            self.audio_event_label.setStyleSheet(
                "color:#0c0;"
                "font-size:15px;"
            )

        self.ignore_audio_until = time.time() + 1.5
        print(
            f"   ✅ [Qt] 设置音频忽略截止时间: "
            f"{time.ctime(self.ignore_audio_until)}"
        )

        QTimer.singleShot(1500, self.unlock_status)

        print("✅ [Qt] 所有状态重置完成")

    def unlock_status(self):
        self.is_resetting = False

    def switch_video(self, area):
        if self.switch_lock:
            return

        self.switch_timestamp = time.time()
        print(f"\n🔄 [Qt] 切换时间戳提前设置: {time.ctime(self.switch_timestamp)}")

        self.switching_in_progress = True
        print("🔒 [Qt] 开始切换区域，全局屏蔽TCP消息")

        self.switch_lock = True
        self.ignore_updates = True

        QTimer.singleShot(700, self.unlock_ignore_updates)
        QTimer.singleShot(700, self.unlock_switch)

        if self.pending_timer is not None:
            self.pending_timer.stop()
            self.pending_timer.deleteLater()
            self.pending_timer = None

        self.stop_all_threads()
        send_switch_cmd(area)

        while self.video_container_layout.count():
            item = self.video_container_layout.takeAt(0)
            widget = item.widget()

            if widget:
                widget.setParent(None)

        video_container = QWidget()
        video_container.setStyleSheet("background:#222;")

        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(0)

        self.video_label = QLabel()
        self.video_label.setStyleSheet("background:#222;")
        video_layout.addWidget(self.video_label)

        self.switch_overlay = QWidget(video_container)
        self.switch_overlay.setStyleSheet(
            "background-color: rgba(0, 0, 0, 0.9);"
        )

        overlay_layout = QVBoxLayout(self.switch_overlay)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        overlay_layout.setSpacing(0)

        self.switch_overlay_label = QLabel("监控区域切换中......")
        self.switch_overlay_label.setStyleSheet(
            "color: white;"
            "font-size: 24px;"
            "font-weight: bold;"
        )
        self.switch_overlay_label.setAlignment(Qt.AlignCenter)
        overlay_layout.addWidget(self.switch_overlay_label)

        self.switch_overlay.show()
        self.switch_overlay.raise_()

        QTimer.singleShot(
            10,
            lambda: self.switch_overlay.setGeometry(video_container.rect()),
        )

        video_container.resizeEvent = (
            lambda event: self.switch_overlay.setGeometry(video_container.rect())
        )

        self.video_container_layout.addWidget(video_container, stretch=1)

        right_panel = self.create_right_panel()
        self.video_container_layout.addWidget(right_panel)

        self.pending_timer = QTimer()
        self.pending_timer.setSingleShot(True)
        self.pending_timer.timeout.connect(self.start_stream)
        self.pending_timer.start(300)

        QTimer.singleShot(1800, self.unlock_switching_progress)

    def unlock_switch(self):
        self.switch_lock = False

    def unlock_switching_progress(self):
        self.switching_in_progress = False
        print("✅ [Qt] 区域切换完成，恢复TCP消息处理")

    def unlock_ignore_updates(self):
        self.ignore_updates = False
        print("✅ [Qt] 全局消息屏蔽已解除")

    def create_right_panel(self):
        w = QWidget()
        w.setFixedWidth(480)
        w.setStyleSheet("background:#181818;")

        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("🚨 行为异常报警")
        title.setStyleSheet(
            "font-size:20px;"
            "color:white;"
            "font-weight:bold;"
            "padding:10px 0;"
            "border-bottom:1px solid #444;"
        )
        layout.addWidget(title)

        def kv(label):
            h = QHBoxLayout()

            l = QLabel(label)
            l.setStyleSheet(
                "color:#bbb;"
                "font-size:15px;"
                "min-width:100px;"
            )

            r = QLabel("")
            r.setStyleSheet(
                "color:white;"
                "font-size:15px;"
            )
            r.setWordWrap(True)

            h.addWidget(l)
            h.addWidget(r, 1)

            layout.addLayout(h)

            return r

        self.alert_status_label = kv("报警状态：")
        self.alert_time_label = kv("报警时间：")
        self.risk_level_label = kv("风险等级：")

        self.risk_progress_bar = GradientProgressBar()
        self.risk_progress_bar.setFixedHeight(16)

        layout.addWidget(self.risk_progress_bar)

        status_title = QLabel("📡 状态异常监测")
        status_title.setStyleSheet(
            "font-size:17px;"
            "color:white;"
            "font-weight:bold;"
            "padding:5px 0;"
            "border-top:1px solid #444;"
        )
        layout.addWidget(status_title)

        status_main_layout = QHBoxLayout()
        status_main_layout.setSpacing(8)

        left_layout = QVBoxLayout()
        left_layout.setSpacing(6)

        a1 = QHBoxLayout()
        l1 = QLabel("动作：")
        l1.setStyleSheet(
            "color:#bbb;"
            "font-size:15px;"
            "min-width:80px;"
        )
        self.action_label = QLabel("✅ 正常")
        self.action_label.setStyleSheet(
            "color:#0c0;"
            "font-size:15px;"
        )
        a1.addWidget(l1)
        a1.addWidget(self.action_label, 1)
        left_layout.addLayout(a1)

        a2 = QHBoxLayout()
        l2 = QLabel("表情：")
        l2.setStyleSheet(
            "color:#bbb;"
            "font-size:15px;"
            "min-width:80px;"
        )
        self.face_label = QLabel("✅ 正常")
        self.face_label.setStyleSheet(
            "color:#0c0;"
            "font-size:15px;"
        )
        a2.addWidget(l2)
        a2.addWidget(self.face_label, 1)
        left_layout.addLayout(a2)

        a3 = QHBoxLayout()
        l3 = QLabel("音频：")
        l3.setStyleSheet(
            "color:#bbb;"
            "font-size:15px;"
            "min-width:80px;"
        )
        self.audio_label = QLabel("✅ 正常")
        self.audio_label.setStyleSheet(
            "color:#0c0;"
            "font-size:15px;"
        )
        a3.addWidget(l3)
        a3.addWidget(self.audio_label, 1)
        left_layout.addLayout(a3)

        a4 = QHBoxLayout()
        l4 = QLabel("音频事件：")
        l4.setStyleSheet(
            "color:#bbb;"
            "font-size:15px;"
            "min-width:80px;"
        )
        self.audio_event_label = QLabel("无")
        self.audio_event_label.setStyleSheet(
            "color:#0c0;"
            "font-size:15px;"
        )
        self.audio_event_label.setWordWrap(True)

        a4.addWidget(l4)
        a4.addWidget(self.audio_event_label, 1)
        left_layout.addLayout(a4)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(3)
        right_layout.setContentsMargins(0, 0, 0, 0)

        sound_title = QLabel("🎵 识别声音")
        sound_title.setStyleSheet(
            "color:#bbb;"
            "font-size:14px;"
            "padding:0;"
        )
        right_layout.addWidget(sound_title)

        self.audio_sound_list = QListWidget()
        self.audio_sound_list.setStyleSheet(
            """
            QListWidget {
                background-color: #222;
                color: white;
                font-size: 12px;
                border: none;
                border-radius:4px;
            }

            QListWidget::item {
                padding:3px;
            }
            """
        )
        self.audio_sound_list.setFixedHeight(80)
        right_layout.addWidget(self.audio_sound_list)

        status_main_layout.addLayout(left_layout, stretch=1)
        status_main_layout.addLayout(right_layout, stretch=1)

        layout.addLayout(status_main_layout)

        alt = QLabel("📋 报警历史")
        alt.setStyleSheet(
            "font-size:17px;"
            "color:white;"
            "font-weight:bold;"
            "padding:10px 0;"
            "border-top:1px solid #444;"
        )
        layout.addWidget(alt)

        self.alert_list_widget = QListWidget()
        self.alert_list_widget.setWordWrap(True)
        self.alert_list_widget.setTextElideMode(Qt.ElideNone)
        self.alert_list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.alert_list_widget.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.alert_list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.alert_list_widget.setUniformItemSizes(False)
        self.alert_list_widget.setStyleSheet(
            """
            QListWidget {
                background:#222;
                color:white;
                font-size:12px;
                border:none;
            }

            QListWidget::item {
                padding:6px;
                border-bottom:1px solid #333;
            }

            QListWidget::item:selected {
                background:#333;
            }
            """
        )
        layout.addWidget(self.alert_list_widget, 1)

        llm_title = QLabel("🤖 AI辅助研判")
        llm_title.setStyleSheet(
            "font-size:16px;"
            "color:white;"
            "font-weight:bold;"
            "padding:5px 0;"
            "border-top:1px solid #444;"
        )
        layout.addWidget(llm_title)

        self.llm_analysis_text = QTextEdit()
        self.llm_analysis_text.setReadOnly(True)
        self.llm_analysis_text.setPlainText("暂无AI研判结果")
        self.llm_analysis_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #222;
                color: #00e5ff;
                font-size:13px;
                border:1px solid #333;
                border-radius:4px;
            }
            """
        )
        self.llm_analysis_text.setFixedHeight(220)
        layout.addWidget(self.llm_analysis_text, 1)

        return w

    def message_allowed(self, data, msg_name):
        msg_version = data.get("version", 0)

        self.check_and_update_version(msg_version)

        if msg_version != self.current_valid_version:
            print(
                f"❌ [Qt] 丢弃版本不匹配的{msg_name}消息: "
                f"消息版本={msg_version}, "
                f"当前版本={self.current_valid_version}"
            )
            return False

        if self.switching_in_progress or self.is_resetting or self.ignore_updates:
            return False

        msg_time = time.time()

        if msg_time < self.switch_timestamp:
            print(
                f"❌ [Qt] 丢弃旧{msg_name}消息: "
                f"消息时间={time.ctime(msg_time)}, "
                f"切换时间={time.ctime(self.switch_timestamp)}"
            )
            return False

        if msg_time < self.switch_timestamp + 1.0:
            print(f"❌ [Qt] 丢弃切换过渡期{msg_name}消息")
            return False

        return True

    def update_risk_status(self, data):
        if not self.message_allowed(data, "风险"):
            return

        try:
            raw_severity = float(data.get("final_severity", 0.0))
        except Exception:
            raw_severity = 0.0

        raw_level = data.get("alert_level", "正常")
        now = time.time()

        # ====================== 1. 风险值平滑 ======================
        # 风险上升快，下降慢，避免进度条频繁跳动
        if raw_severity > self.display_severity:
            alpha = 0.65
        else:
            alpha = 0.12

        self.display_severity = (
            alpha * raw_severity
            + (1 - alpha) * self.display_severity
        )

        # 小风险值直接归零，避免轻微抖动
        if self.display_severity < 0.03:
            self.display_severity = 0.0

        # ====================== 2. 风险等级防抖 ======================
        level_priority = {
            "正常": 0,
            "低风险": 1,
            "中风险": 2,
            "高风险": 3,
        }

        old_priority = level_priority.get(self.last_risk_level, 0)
        new_priority = level_priority.get(raw_level, 0)

        # 风险升级立即显示
        if new_priority > old_priority:
            self.last_risk_level = raw_level
            self.last_level_change_time = now

        # 风险降低延迟显示，避免频繁跳变
        elif new_priority < old_priority:
            if now - self.last_level_change_time >= self.level_hold_seconds:
                self.last_risk_level = raw_level
                self.last_level_change_time = now

        # 等级不变，刷新时间
        else:
            self.last_level_change_time = now

        alert_level = self.last_risk_level

        # ====================== 3. 更新界面 ======================
        self.risk_level_label.setText(
            f"{alert_level}（风险值:{self.display_severity:.2f}）"
        )

        progress_val = min(int(self.display_severity * 100), 100)
        self.risk_progress_bar.setValue(progress_val)

        if alert_level == "高风险":
            self.risk_level_label.setStyleSheet(
                "color:red;"
                "font-size:15px;"
                "font-weight:bold;"
            )

        elif alert_level == "中风险":
            self.risk_level_label.setStyleSheet(
                "color:orange;"
                "font-size:15px;"
                "font-weight:bold;"
            )

        elif alert_level == "低风险":
            self.risk_level_label.setStyleSheet(
                "color:yellow;"
                "font-size:15px;"
            )

        else:
            self.risk_level_label.setStyleSheet(
                "color:white;"
                "font-size:15px;"
            )

    def update_modal_status(self, data):
        if not self.message_allowed(data, "模态"):
            return

        current_time = time.time()

        if current_time < self.ignore_audio_until:
            data["audio_abnormal"] = False
            data["audio_text"] = "正常"
            data["sound_positive"] = False

        self.current_action_abnormal = data.get("action_abnormal", False)

        if self.current_action_abnormal:
            self.action_label.setText(f"⚠️ ：{data.get('action_text', '')}")
            self.action_label.setStyleSheet(
                "color:red;"
                "font-size:15px;"
            )
        else:
            self.action_label.setText("✅ 正常")
            self.action_label.setStyleSheet(
                "color:#0c0;"
                "font-size:15px;"
            )

        face_txt = data.get("face_text", "正常")
        face_positive = data.get("face_positive", False)

        self.current_face_abnormal = (
            face_txt != "正常"
            and not face_positive
        )

        if face_positive:
            self.face_label.setText("😊 开心")
            self.face_label.setStyleSheet(
                "color:#0c0;"
                "font-size:15px;"
            )

        elif self.current_face_abnormal:
            self.face_label.setText(f"⚠️ {face_txt}")
            self.face_label.setStyleSheet(
                "color:orange;"
                "font-size:15px;"
            )

        else:
            self.face_label.setText("✅ 正常")
            self.face_label.setStyleSheet(
                "color:#0c0;"
                "font-size:15px;"
            )

        self.current_audio_abnormal = data.get("audio_abnormal", False)
        sound_positive = data.get("sound_positive", False)

        if self.current_audio_abnormal:
            self.audio_label.setText("⚠️ 异常")
            self.audio_label.setStyleSheet(
                "color:red;"
                "font-size:15px;"
            )

        elif sound_positive:
            self.audio_label.setText("😊 笑声")
            self.audio_label.setStyleSheet(
                "color:#0c0;"
                "font-size:15px;"
            )

        else:
            self.audio_label.setText("✅ 正常")
            self.audio_label.setStyleSheet(
                "color:#0c0;"
                "font-size:15px;"
            )

        self.refresh_alarm_status()

    def refresh_alarm_status(self):
        now = time.time()

        any_abnormal = (
            self.current_action_abnormal
            or self.current_face_abnormal
            or self.current_audio_abnormal
        )

        if any_abnormal:
            self.alarm_locked_until = now + self.alarm_lock_seconds
            self.all_normal_start_time = 0

            self.alert_status_label.setText("⚠️ 报警中")
            self.alert_status_label.setStyleSheet(
                "color:red;"
                "font-size:16px;"
            )
            return

        if now < self.alarm_locked_until:
            self.alert_status_label.setText("⚠️ 报警中")
            self.alert_status_label.setStyleSheet(
                "color:red;"
                "font-size:16px;"
            )
            return

        if self.all_normal_start_time == 0:
            self.all_normal_start_time = now

        if now - self.all_normal_start_time >= self.need_wait_normal_seconds:
            self.alert_status_label.setText("✅ 正常")
            self.alert_status_label.setStyleSheet(
                "color:white;"
                "font-size:16px;"
            )

    def update_audio_panel(self, data):
        if not self.message_allowed(data, "音频"):
            return

        current_time = time.time()

        if current_time < self.ignore_audio_until:
            remaining = self.ignore_audio_until - current_time
            print(f"❌ [Qt] 忽略切换期间的音频结果，剩余: {remaining:.2f}秒")
            return

        sound_abnormal = data.get("sound_abnormal", False)
        abnormal_list = data.get("abnormal_sounds", [])
        sound_positive = data.get("sound_positive", False)
        positive_list = data.get("positive_sounds", [])

        self.cached_audio_status = {
            "sound_abnormal": sound_abnormal,
            "abnormal_sounds": abnormal_list,
            "sound_positive": sound_positive,
            "positive_sounds": positive_list,
        }

        unique_abnormal = list(set(abnormal_list)) if abnormal_list else []
        unique_positive = list(set(positive_list)) if positive_list else []

        if self.audio_sound_list is not None:
            self.audio_sound_list.clear()

        top5 = data.get("top5_sounds", [])

        if not top5:
            item = QListWidgetItem("无声音数据")
            item.setForeground(QColor("#888"))
            self.audio_sound_list.addItem(item)

        else:
            for sound_info in top5:
                if isinstance(sound_info, dict):
                    name = sound_info.get("name", "未知")
                    score = sound_info.get("score", 0.0)

                elif isinstance(sound_info, (list, tuple)) and len(sound_info) >= 2:
                    name, score = sound_info[0], sound_info[1]

                else:
                    name = str(sound_info)
                    score = 0.0

                if score > 0.7:
                    color = QColor(255, 100, 100)

                elif score > 0.4:
                    color = QColor(255, 200, 100)

                else:
                    color = QColor(150, 150, 150)

                item = QListWidgetItem(f"{name}: {score:.2f}")
                item.setForeground(color)
                self.audio_sound_list.addItem(item)

        if self.audio_event_label is not None:
            if sound_abnormal:
                display_text = (
                    "、".join(unique_abnormal)
                    if unique_abnormal
                    else "异常声音触发"
                )
                self.audio_event_label.setText(f"⚠️ {display_text}")
                self.audio_event_label.setStyleSheet(
                    "color:red;"
                    "font-size:15px;"
                )

            elif sound_positive:
                display_text = (
                    "、".join(unique_positive)
                    if unique_positive
                    else "积极音频触发"
                )
                self.audio_event_label.setText(f"😊 {display_text}")
                self.audio_event_label.setStyleSheet(
                    "color:#0c0;"
                    "font-size:15px;"
                )

            else:
                self.audio_event_label.setText("无")
                self.audio_event_label.setStyleSheet(
                    "color:#0c0;"
                    "font-size:15px;"
                )

        transcript = data.get("transcript", "")
        if self.audio_transcript_label is not None:
            self.audio_transcript_label.setPlainText(transcript)

    def rebuild_alert_history_list(self):
        if self.alert_list_widget is None:
            return

        self.alert_list_widget.clear()

        list_width = max(
            260,
            self.alert_list_widget.viewport().width() - 20,
        )

        font_metrics = QFontMetrics(self.alert_list_widget.font())

        for m in reversed(self.alert_history):
            item = QListWidgetItem(m)
            item.setToolTip(m)

            text_rect = font_metrics.boundingRect(
                QRect(0, 0, list_width, 10000),
                Qt.TextWordWrap,
                m,
            )

            item_height = max(34, text_rect.height() + 18)
            item.setSizeHint(QSize(list_width, item_height))

            if (
                "高风险" in m
                or "中风险" in m
                or "异常" in m
                or "报警" in m
            ):
                item.setForeground(QColor(255, 60, 60))

            elif "低风险" in m:
                item.setForeground(QColor(255, 220, 80))

            else:
                item.setForeground(QColor(255, 255, 255))

            self.alert_list_widget.addItem(item)

        self.alert_list_widget.doItemsLayout()
        self.alert_list_widget.viewport().update()

    def update_alert_info(self, data):
        if not self.message_allowed(data, "报警"):
            return

        severity = 0.0
        alert_level = "正常"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        trigger_modal = ""

        if isinstance(data, dict):
            try:
                severity = float(data.get("final_severity", 0.0))
            except Exception:
                severity = 0.0

            alert_level = data.get("alert_level", "正常")
            timestamp = data.get("timestamp", timestamp)
            trigger_modal = data.get("trigger_modal", "")

            if trigger_modal:
                msg = (
                    f"[{timestamp}]【{alert_level}】"
                    f"严重度={severity:.2f} | ({trigger_modal})"
                )
            else:
                msg = (
                    f"[{timestamp}]【{alert_level}】"
                    f"严重度={severity:.2f}"
                )

        else:
            msg = str(data)

        print(f"📋 [Qt] 新增报警历史: {msg}")

        self.alert_history.append(msg)

        if len(self.alert_history) > 20:
            self.alert_history.pop(0)

        self.rebuild_alert_history_list()

        if self.alert_time_label is not None:
            self.alert_time_label.setText(timestamp)

    def update_llm_analysis(self, data):
        if not self.message_allowed(data, "大模型研判"):
            return

        def safe(t):
            return str(t).replace("\n", "<br>")

        result = data.get("llm_analysis", "")

        if not result:
            result = json.dumps({
                "event_desc": "暂无AI研判结果",
                "risk_reason": "",
                "advice": ""
            }, ensure_ascii=False)

        try:
            d = json.loads(result)
        except Exception:
            d = {
                "event_desc": result,
                "risk_reason": "解析失败（非JSON或格式错误）",
                "advice": "建议人工查看现场视频"
            }

        html = f"""
    <div style="font-family:Microsoft YaHei; font-size:13px;">

        <div style="color:#00E5FF; font-size:14px; font-weight:600; margin-bottom:4px;">
            📌 事件描述
        </div>
        <div style="color:#E0E0E0; line-height:1.2; margin-bottom:8px;">
            {safe(d.get("event_desc", ""))}
        </div>

        <div style="height:2px;"></div>

        <div style="color:#FFB300; font-size:14px; font-weight:600;margin-bottom:4px;">
            ⚠️ 风险依据
        </div>
        <div style="color:#E0E0E0; line-height:1.2; margin-bottom:8px;">
            {safe(d.get("risk_reason", ""))}
        </div>

        <div style="height:2px;"></div>

        <div style="color:#69F0AE; font-size:14px; font-weight:600;margin-bottom:4px;">
            🧭 处置建议
        </div>
        <div style="color:#E0E0E0; line-height:1.2;">
            {safe(d.get("advice", ""))}
        </div>

    </div>
    """

        self.llm_analysis_text.setHtml(html)

    def start_stream(self):
        if not hasattr(self, "video_label") or self.video_label is None:
            return

        if self.th is not None and self.th.isRunning():
            self.stop_all_threads()

        self.th = VideoStreamThread(BOARD_STREAM_URL)
        self.th.frame_signal.connect(self.show_frame)
        self.th.start()

    def show_frame(self, frame):
        if self.video_label is None:
            return

        if self.switch_overlay is not None and self.switch_overlay.isVisible():
            self.switch_overlay.hide()
            print("✅ [Qt] 第一帧已显示，隐藏切换覆盖层")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        q_img = QImage(
            rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888,
        )

        self.video_label.setPixmap(
            QPixmap.fromImage(q_img).scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if hasattr(self, "switch_overlay") and self.switch_overlay is not None:
            parent = self.switch_overlay.parentWidget()

            if parent is not None:
                self.switch_overlay.setGeometry(parent.rect())

        if self.alert_list_widget is not None and self.alert_history:
            QTimer.singleShot(100, self.rebuild_alert_history_list)

    def stop_all_threads(self):
        if self.th is not None:
            self.th.stop()
            self.th.wait(1000)
            self.th = None


def tcp_listen(ui):
    print(f"✅ [TCP] 正在连接数据服务器 {BOARD_IP}:{DATA_PORT}")

    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
            sock.settimeout(5.0)
            sock.connect((BOARD_IP, DATA_PORT))
            sock.settimeout(None)

            ui.tcp_socket = sock

            print("✅ [TCP] 数据连接成功，已禁用Nagle算法，缓冲区=64KB")

            buffer = b""

            while True:
                data = sock.recv(16384)

                if not data:
                    break

                buffer += data

                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)

                    try:
                        js = json.loads(line.decode("utf-8"))
                        t = js.get("type", "")

                        if t == "audio_status":
                            ui.audio_signal.emit(js)

                        elif t == "modal_status":
                            ui.modal_signal.emit(js)

                        elif t == "video_loop_reset":
                            version = js.get("version", 0)
                            ui.reset_signal.emit(version)

                        elif t == "risk_status":
                            ui.risk_signal.emit(js)

                        elif t == "llm_analysis":
                            ui.llm_signal.emit(js)

                        else:
                            ui.msg_signal.emit(js)

                    except Exception as e:
                        print(f"⚠️ [TCP] 解析消息失败: {e}")
                        continue

        except Exception as e:
            print(f"⚠️ [TCP] 连接失败或断开: {e}")
            ui.tcp_socket = None
            time.sleep(1.0)
            print("🔄 [TCP] 尝试重新连接...")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    ui = AlertUI()
    ui.show()

    tcp_thread = threading.Thread(
        target=tcp_listen,
        args=(ui,),
        daemon=True,
    )
    tcp_thread.start()

    sys.exit(app.exec_())
