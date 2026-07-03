import math
import time
from collections import deque


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def normalize_score(score: float, epsilon: float = 1e-6) -> float:
    return max(min(score, 1.0), 0.0 + epsilon)


def calc_adaptive_weight(abnormal_flag: bool, base_w: float, enabled: bool = True) -> float:
    if not enabled:
        return 0.0
    return base_w * 1.35 if abnormal_flag else base_w


class TimeWindowManager:
    def __init__(self, window_seconds: float = 6.0):
        self.window_seconds = window_seconds
        self.action_history = deque()
        self.face_history = deque()
        self.text_history = deque()

    def _cleanup(self, history: deque):
        now = time.time()
        while history and now - history[0][0] > self.window_seconds:
            history.popleft()

    def update(self, action_abnormal: bool, face_abnormal: bool, text_abnormal: bool):
        now = time.time()

        self.action_history.append((now, action_abnormal))
        self.face_history.append((now, face_abnormal))
        self.text_history.append((now, text_abnormal))

        self._cleanup(self.action_history)
        self._cleanup(self.face_history)
        self._cleanup(self.text_history)

    def get_window_abnormal(self) -> tuple[bool, bool, bool]:
        self._cleanup(self.action_history)
        self._cleanup(self.face_history)
        self._cleanup(self.text_history)

        return (
            any(abn for _, abn in self.action_history),
            any(abn for _, abn in self.face_history),
            any(abn for _, abn in self.text_history),
        )

    def reset(self):
        self.action_history.clear()
        self.face_history.clear()
        self.text_history.clear()


class PositiveWindowManager:
    def __init__(self, window_seconds: float = 3.0):
        self.window_seconds = window_seconds
        self.last_positive_time = 0.0

    def update(self, face_positive: bool, sound_positive: bool):
        if face_positive or sound_positive:
            self.last_positive_time = time.time()

    def is_active(self) -> bool:
        if self.last_positive_time <= 0:
            return False
        return time.time() - self.last_positive_time <= self.window_seconds

    def reset(self):
        self.last_positive_time = 0.0


window_manager = TimeWindowManager(window_seconds=6)
positive_window_manager = PositiveWindowManager(window_seconds=3)


def get_window_modal_str(action_win: bool, face_win: bool, text_win: bool) -> str:
    modals = []

    if action_win:
        modals.append("动作")

    if face_win:
        modals.append("表情")

    if text_win:
        modals.append("语义")

    return "(" + "、".join(modals) + ")" if modals else "(无)"


def calc_consistency_factor(
    action_abnormal: bool,
    face_abnormal: bool,
    text_abnormal: bool,
    action_abnormal_in_window: bool,
    face_abnormal_in_window: bool,
    text_abnormal_in_window: bool,
    enable_face: bool = True,
    enable_semantic: bool = True,
) -> tuple[float, int]:

    filtered_face_win = face_abnormal_in_window if enable_face else False
    filtered_text_win = text_abnormal_in_window if enable_semantic else False

    window_mode_count = sum([
        action_abnormal_in_window,
        filtered_face_win,
        filtered_text_win,
    ])

    filtered_face_abn = face_abnormal if enable_face else False
    filtered_text_abn = text_abnormal if enable_semantic else False

    current_has_abnormal = any([
        action_abnormal,
        filtered_face_abn,
        filtered_text_abn,
    ])

    if current_has_abnormal:
        if window_mode_count == 3:
            return 2.0, 3

        if window_mode_count == 2:
            return 1.30, 2

        if window_mode_count == 1:
            return 1.10, 1

        return 0.95, 0

    if window_mode_count >= 2:
        return 0.75, 2

    return 0.55, 0


def calculate_final_severity(
    action_score: float,
    face_score: float,
    semantic_score: float,
    action_abnormal: bool,
    face_abnormal: bool,
    text_abnormal: bool,
    sound_abnormal: bool = False,
    face_positive: bool = False,
    sound_positive: bool = False,
    enable_face: bool = True,
    enable_semantic: bool = True,
    enable_audio_event: bool = True,
):

    if not enable_face:
        face_abnormal = False
        face_positive = False
        face_score = 0.2

    if not enable_semantic:
        text_abnormal = False
        semantic_score = 0.5

    if not enable_audio_event:
        sound_abnormal = False
        sound_positive = False

    window_manager.update(action_abnormal, face_abnormal, text_abnormal)
    action_win, face_win, text_win = window_manager.get_window_abnormal()

    positive_window_manager.update(face_positive, sound_positive)
    positive_window_active = positive_window_manager.is_active()

    action_score = normalize_score(action_score)
    face_score = normalize_score(face_score)
    semantic_score = normalize_score(semantic_score)

    if positive_window_active:
        if not action_abnormal:
            action_score = min(action_score, 0.25)

        if not face_abnormal:
            face_score = min(face_score, 0.15)

        if not text_abnormal:
            semantic_score = min(semantic_score, 0.30)

    w_action = calc_adaptive_weight(action_abnormal, 0.40, enabled=True)
    w_face = calc_adaptive_weight(face_abnormal, 0.30, enabled=enable_face)
    w_text = calc_adaptive_weight(text_abnormal, 0.30, enabled=enable_semantic)

    total_w = w_action + w_face + w_text

    if total_w > 0:
        w_action /= total_w
        w_face /= total_w
        w_text /= total_w
    else:
        w_action = 1.0
        w_face = 0.0
        w_text = 0.0

    base_score = (
        action_score * w_action
        + face_score * w_face
        + semantic_score * w_text
    )

    consistency_factor, window_mode_count = calc_consistency_factor(
        action_abnormal,
        face_abnormal,
        text_abnormal,
        action_win,
        face_win,
        text_win,
        enable_face,
        enable_semantic,
    )

    raw_final = base_score * consistency_factor
    final_severity = sigmoid(raw_final * 3.0 - 1.0)

    if enable_audio_event and sound_abnormal:
        final_severity = 1.0
        print("🚨 [风险计算] 检测到异常声音事件，直接触发高风险")
        return round(final_severity, 3), window_mode_count, (action_win, face_win, text_win)

    if positive_window_active:
        has_current_abnormal = action_abnormal or face_abnormal or text_abnormal
        has_window_abnormal = action_win or face_win or text_win

        if has_current_abnormal:
            final_severity = min(final_severity * 0.45, 0.40)
        elif has_window_abnormal:
            final_severity = min(final_severity * 0.30, 0.25)
        else:
            final_severity = min(final_severity * 0.20, 0.18)

        final_severity = max(final_severity, 0.05)

        print(
            f"🔽 [积极窗口] 开心/笑声窗口生效 "
            f"(face_positive={face_positive}, sound_positive={sound_positive})，"
            f"风险限制后: {final_severity:.3f}"
        )

    return round(final_severity, 3), window_mode_count, (action_win, face_win, text_win)


def get_alert_level(
    final_severity: float,
    window_mode_count: int,
    face_positive: bool = False,
    sound_positive: bool = False,
    sound_abnormal: bool = False,
    enable_audio_event: bool = True,
) -> str:

    positive_signal = face_positive or (enable_audio_event and sound_positive)
    positive_window_active = positive_window_manager.is_active() or positive_signal

    if enable_audio_event and sound_abnormal:
        return "高风险"

    if positive_window_active:
        if final_severity >= 0.60:
            return "中风险"

        if final_severity >= 0.10:
            return "低风险"

        return "正常"

    if window_mode_count == 3 and final_severity >= 0.80:
        return "高风险"

    if window_mode_count == 2 and final_severity >= 0.35:
        return "中风险"

    if window_mode_count == 1 and final_severity >= 0.25:
        return "中风险"

    if final_severity >= 0.15:
        return "低风险"

    return "正常"



def should_trigger_llm_analysis(
    alert_level: str,
    final_severity: float,
    sound_abnormal: bool = False,
) -> bool:
    """
    判断是否触发大模型辅助视觉分析
    只在中风险/高风险时触发，避免每帧调用 API
    """
    if sound_abnormal:
        return True

    if alert_level in ["中风险", "高风险"] and final_severity >= 0.35:
        return True

    return False
