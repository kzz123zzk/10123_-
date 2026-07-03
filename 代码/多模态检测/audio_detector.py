from __future__ import division, print_function
import os
import sys
import logging

logging.basicConfig()
logging.addLevelName(logging.WARNING, "WARNING")
logging.addLevelName(logging.INFO, "INFO")
logging.addLevelName(logging.ERROR, "ERROR")
logging.addLevelName(logging.DEBUG, "DEBUG")

try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except ImportError:
    pass

import subprocess
import tempfile
import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf
import importlib.util
import time
import sherpa_onnx

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel(logging.ERROR)


class RKNNSentimentClassifier:
    def __init__(self, rknn_model_path, vocab_path):
        print("🔹 [音频模态] 正在加载RKNN-BERT模型...")
        self.tokenizer = self.SimpleBertTokenizer(vocab_path)
        self.max_seq_len = 128

        from rknnlite.api import RKNNLite

        self.rknn_lite = RKNNLite()

        ret = self.rknn_lite.load_rknn(rknn_model_path)
        if ret != 0:
            raise Exception(f"❌ RKNN模型加载失败！错误码：{ret}")

        ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
        if ret != 0:
            raise Exception(f"❌ RKNN运行时初始化失败！错误码：{ret}")

        print("✅ [音频模态] RKNN-BERT 异常检测模型加载成功\n")

    class SimpleBertTokenizer:
        def __init__(self, vocab_file):
            print("🔹 [音频模态] 加载BERT分词器词汇表...")
            self.vocab = {}

            with open(vocab_file, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    token = line.strip()
                    if token:
                        self.vocab[token] = idx

            self.cls_token_id = self.vocab.get("[CLS]", 101)
            self.sep_token_id = self.vocab.get("[SEP]", 102)
            self.pad_token_id = self.vocab.get("[PAD]", 0)
            self.unk_token_id = self.vocab.get("[UNK]", 100)

            print("✅ [音频模态] 分词器加载完成\n")

        def encode(self, text, max_length):
            chars = list(text)
            tokens = chars[: max_length - 2]
            token_ids = [self.vocab.get(char, self.unk_token_id) for char in tokens]

            input_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]
            pad_len = max_length - len(input_ids)
            input_ids += [self.pad_token_id] * pad_len

            attention_mask = [1] * (len(token_ids) + 2) + [0] * pad_len
            token_type_ids = [0] * max_length

            return (
                np.array([input_ids], dtype=np.int64),
                np.array([attention_mask], dtype=np.int64),
                np.array([token_type_ids], dtype=np.int64),
            )

    def predict(self, text):
        if not text or len(text.strip()) < 2:
            return 1, 0.5

        input_ids, att_mask, token_type_ids = self.tokenizer.encode(
            text,
            self.max_seq_len,
        )

        outputs = self.rknn_lite.inference(
            inputs=[input_ids, att_mask, token_type_ids]
        )

        logits = outputs[0]
        exp_logits = np.exp(logits)
        prob = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        pred_id = np.argmax(logits, axis=1)[0]
        abnormal_prob = prob[0][0]

        return pred_id, abnormal_prob

    def release(self):
        if hasattr(self, "rknn_lite"):
            self.rknn_lite.release()


class AudioDetector:
    def __init__(self, base_dir):
        self.BASE_DIR = base_dir

        self.YAMNET_DIR = os.path.join(self.BASE_DIR, "models_file/yamnet")
        self.YAMNET_H5_PATH = os.path.join(self.YAMNET_DIR, "yamnet.h5")
        self.YAMNET_CLASS_MAP = os.path.join(
            self.YAMNET_DIR,
            "yamnet_class_map.csv",
        )

        self.BERT_MODEL_PATH = os.path.join(
            self.BASE_DIR,
            "models_file/bert/bert_final_no_quant.rknn",
        )
        self.BERT_VOCAB_PATH = os.path.join(
            self.BASE_DIR,
            "models_file/bert/vocab.txt",
        )

        self.SHERPA_MODEL_DIR = os.path.join(
            self.BASE_DIR,
            "models_file/sherpa-onnx-paraformer-zh-small-2024-03-09",
        )
        self.SHERPA_MODEL_PATH = os.path.join(
            self.SHERPA_MODEL_DIR,
            "model.int8.onnx",
        )
        self.SHERPA_TOKENS_PATH = os.path.join(
            self.SHERPA_MODEL_DIR,
            "tokens.txt",
        )

        self.ABNORMAL_SOUND_CLASSES = [
            "Shout", "Bellow", "Yell", "Children shouting", "Screaming",
            "Crying, sobbing", "Baby cry, infant cry", "Whimper", "Wail, moan",
            "Slap, smack", "Whack, thwack", "Smash, crash", "Bang", "Whip",
            "Thump, thud", "Thunk", "Shatter", "Breaking", "Bouncing", "Crushing",
            "Glass", "Crack", "Splinter", "Chink, clink",
            "Gunshot, gunfire", "Explosion", "Fireworks", "Firecracker",
            "Siren", "Police car (siren)", "Ambulance (siren)",
            "Fire engine, fire truck (siren)", "Alarm",
            "Smoke detector, smoke alarm", "Fire alarm", "Buzzer",
            "Hubbub, speech noise, speech babble", "Screech", "Squeal",
            "Rumble", "Clatter",
        ]

        self.SOUND_CN_MAP = {
            "Shout": "大喊",
            "Bellow": "怒吼",
            "Yell": "叫嚷",
            "Children shouting": "儿童大喊",
            "Screaming": "尖叫",
            "Crying, sobbing": "哭喊",
            "Baby cry, infant cry": "婴儿啼哭",
            "Whimper": "呜咽",
            "Wail, moan": "哀嚎",
            "Slap, smack": "拍打/掌击",
            "Whack, thwack": "重击",
            "Smash, crash": "摔砸",
            "Bang": "巨响",
            "Whip": "抽打",
            "Thump, thud": "撞击",
            "Thunk": "闷响",
            "Shatter": "粉碎",
            "Breaking": "破坏",
            "Bouncing": "摔物",
            "Crushing": "碾压",
            "Glass": "玻璃",
            "Crack": "裂痕",
            "Splinter": "碎裂",
            "Chink, clink": "玻璃碰撞",
            "Gunshot, gunfire": "枪击",
            "Explosion": "爆炸",
            "Fireworks": "烟花",
            "Firecracker": "鞭炮",
            "Siren": "警笛",
            "Police car (siren)": "警车警笛",
            "Ambulance (siren)": "救护车警笛",
            "Fire engine, fire truck (siren)": "消防车警笛",
            "Alarm": "警报",
            "Smoke detector, smoke alarm": "烟雾警报",
            "Fire alarm": "火警警报",
            "Buzzer": "蜂鸣警报",
            "Hubbub, speech noise, speech babble": "恶性喧哗",
            "Screech": "尖锐刺耳",
            "Squeal": "刺耳尖叫",
            "Rumble": "剧烈震动",
            "Clatter": "摔砸嘈杂",
            "Laughter": "笑声",
            "Baby laughter": "婴儿笑声",
            "Giggle": "咯咯笑",
            "Chuckle": "轻笑",
            "Snicker": "偷笑",
            "Cheering": "欢呼",
        }

        self.POSITIVE_SOUND_CLASSES = [
            "Laughter",
            "Baby laughter",
            "Giggle",
            "Chuckle",
            "Snicker",
            "Cheering",
        ]

        self.SPEECH_THRESHOLD = 0.3
        self.ABNORMAL_SOUND_THRESHOLD = 0.02
        self.POSITIVE_SOUND_THRESHOLD = 0.02

        print("🔹 [音频模态] 正在加载 sherpa-onnx Paraformer 中文语音识别模型...")

        if not os.path.exists(self.SHERPA_MODEL_PATH):
            raise FileNotFoundError(f"❌ sherpa模型不存在: {self.SHERPA_MODEL_PATH}")

        if not os.path.exists(self.SHERPA_TOKENS_PATH):
            raise FileNotFoundError(f"❌ sherpa tokens不存在: {self.SHERPA_TOKENS_PATH}")

        self.sherpa_recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=self.SHERPA_MODEL_PATH,
            tokens=self.SHERPA_TOKENS_PATH,
            num_threads=2,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
        )

        print("✅ [音频模态] sherpa-onnx Paraformer 中文语音识别模型加载完成\n")

        print("🔹 [音频模态] 正在加载 RKNN-BERT 模型...")
        self.bert_classifier = RKNNSentimentClassifier(
            self.BERT_MODEL_PATH,
            self.BERT_VOCAB_PATH,
        )

        print("🔹 [音频模态] 正在加载 YAMNet 声音分类模型...")

        sys.path.insert(0, self.YAMNET_DIR)

        features_path = os.path.join(self.YAMNET_DIR, "features.py")
        spec = importlib.util.spec_from_file_location("features", features_path)
        features_lib = importlib.util.module_from_spec(spec)
        sys.modules["features"] = features_lib
        spec.loader.exec_module(features_lib)

        params_path = os.path.join(self.YAMNET_DIR, "params.py")
        spec = importlib.util.spec_from_file_location("params", params_path)
        params = importlib.util.module_from_spec(spec)
        sys.modules["params"] = params
        spec.loader.exec_module(params)
        yamnet_params = params

        yamnet_path = os.path.join(self.YAMNET_DIR, "yamnet.py")
        spec = importlib.util.spec_from_file_location("yamnet", yamnet_path)
        yamnet = importlib.util.module_from_spec(spec)
        sys.modules["yamnet"] = yamnet
        spec.loader.exec_module(yamnet)
        yamnet_model = yamnet

        self.params = yamnet_params.Params()
        self.yamnet = yamnet_model.yamnet_frames_model(self.params)
        self.yamnet.load_weights(self.YAMNET_H5_PATH)
        self.yamnet_classes = yamnet_model.class_names(self.YAMNET_CLASS_MAP)

        print("✅ [音频模态] 所有模型加载完成！")

    def _video_to_wav(self, video_path, temp_wav_path):
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            "-v", "quiet",
            temp_wav_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return False

        if not os.path.exists(temp_wav_path) or os.path.getsize(temp_wav_path) < 100:
            return False

        return True

    def _speech_to_text(self, wav_file):
        """使用 sherpa-onnx Paraformer 进行离线中文语音识别"""
        try:
            wav_data, sr = sf.read(wav_file, dtype=np.int16)

            if len(wav_data.shape) > 1:
                wav_data = np.mean(wav_data, axis=1).astype(np.int16)

            if sr != 16000:
                wav_float = wav_data.astype(np.float32) / 32768.0
                wav_float = resampy.resample(wav_float, sr, 16000)
                wav_data = np.clip(
                    wav_float * 32768.0,
                    -32768,
                    32767,
                ).astype(np.int16)
                sr = 16000

            samples = wav_data.astype(np.float32) / 32768.0

            stream = self.sherpa_recognizer.create_stream()
            stream.accept_waveform(sr, samples)
            self.sherpa_recognizer.decode_stream(stream)

            text = stream.result.text.strip()
            return text

        except Exception as e:
            print(f"❌ sherpa-onnx 转录失败: {str(e)}")
            return ""

    def detect(self, file_path):
        temp_wav_path = None

        yamnet_time = 0.0
        asr_time = 0.0
        bert_time = 0.0

        try:
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_wav.close()
            temp_wav_path = temp_wav.name

            is_video = file_path.endswith((".mp4", ".avi", ".mov", ".mkv"))

            if is_video:
                extract_success = self._video_to_wav(file_path, temp_wav_path)
                if not extract_success:
                    return {
                        "code": 500,
                        "msg": "音频提取失败",
                        "is_alert": False,
                        "yamnet_cost_ms": 0.0,
                        "whisper_cost_ms": 0.0,
                        "bert_cost_ms": 0.0,
                        "audio_total_cost_ms": 0.0,
                    }
                wav_file = temp_wav_path
            else:
                wav_file = file_path

            try:
                wav_data, sr = sf.read(wav_file, dtype=np.int16)
            except Exception as e:
                return {
                    "code": 500,
                    "msg": f"音频读取失败: {str(e)}",
                    "is_alert": False,
                    "yamnet_cost_ms": 0.0,
                    "whisper_cost_ms": 0.0,
                    "bert_cost_ms": 0.0,
                    "audio_total_cost_ms": 0.0,
                }

            waveform = wav_data / 32768.0

            if len(waveform.shape) > 1:
                waveform = np.mean(waveform, axis=1)

            if sr != self.params.sample_rate:
                waveform = resampy.resample(
                    waveform,
                    sr,
                    self.params.sample_rate,
                )

            yamnet_start = time.perf_counter()

            try:
                waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
                waveform_tf = tf.squeeze(waveform_tf)
                scores, embeddings, spectrogram = self.yamnet(waveform_tf)
                scores = scores.numpy()
            except Exception:
                waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
                waveform_tf = tf.squeeze(waveform_tf)
                waveform_batch = tf.expand_dims(waveform_tf, axis=0)
                scores, embeddings, spectrogram = self.yamnet.predict_on_batch(
                    waveform_batch
                )

            yamnet_time = (time.perf_counter() - yamnet_start) * 1000

            mean_scores = np.mean(scores, axis=0)

            all_sounds = [
                (self.yamnet_classes[i].strip(), mean_scores[i])
                for i in range(len(mean_scores))
            ]

            top5_idx = np.argsort(mean_scores)[::-1][:5]
            sound_top5 = [
                (self.yamnet_classes[i].strip(), mean_scores[i])
                for i in top5_idx
            ]

            speech_score = next(
                (s for n, s in all_sounds if n == "Speech"),
                0.0,
            )

            text = ""
            bert_label = 1
            semantic_score = 0.5

            if speech_score >= self.SPEECH_THRESHOLD:
                asr_start = time.perf_counter()
                text = self._speech_to_text(wav_file)
                asr_time = (time.perf_counter() - asr_start) * 1000

                bert_start = time.perf_counter()
                if text:
                    bert_label, semantic_score = self.bert_classifier.predict(text)
                bert_time = (time.perf_counter() - bert_start) * 1000
            else:
                text = ""
                bert_label = 1
                semantic_score = 0.5
                asr_time = 0.0
                bert_time = 0.0

            hit_sound = [
                (n, s)
                for n, s in all_sounds
                if n in self.ABNORMAL_SOUND_CLASSES
                and s >= self.ABNORMAL_SOUND_THRESHOLD
            ]

            hit_sound_cn = [
                self.SOUND_CN_MAP.get(n, n)
                for n, s in hit_sound
            ]

            text_abnormal = bert_label == 0 and semantic_score > 0.6

            hit_positive_sound = [
                (n, s)
                for n, s in sound_top5
                if n in self.POSITIVE_SOUND_CLASSES
                and s >= self.POSITIVE_SOUND_THRESHOLD
            ]

            hit_positive_sound_cn = [
                self.SOUND_CN_MAP.get(n, n)
                for n, s in hit_positive_sound
            ]

            is_alert = False
            risk_level = 0
            alert_msg = "无异常"

            if text_abnormal or len(hit_sound) > 0:
                is_alert = True
                risk_level = 1
                alert_parts = []

                if len(hit_sound_cn) > 0:
                    alert_parts.append(
                        f"检测到危险声音: {','.join(hit_sound_cn)}"
                    )

                alert_msg = " | ".join(alert_parts) if alert_parts else ""

            print(
                f"⏱️ 音频模块耗时 | "
                f"YAMNet: {yamnet_time:.2f}ms | "
                f"sherpa-onnx: {asr_time:.2f}ms | "
                f"BERT: {bert_time:.2f}ms | "
                f"speech_score={speech_score:.3f} | "
                f"text={text}"
            )

            return {
                "code": 200,
                "msg": "音频检测完成",
                "is_alert": bool(is_alert),
                "risk_level": int(risk_level),
                "risk_level_text": ["低风险", "中风险", "高风险"][risk_level],
                "alert_msg": alert_msg,
                "transcript": text,
                "speech_score": float(speech_score),
                "abnormal_sounds": hit_sound_cn,
                "text_abnormal": bool(text_abnormal),
                "semantic_score": float(semantic_score),
                "top5_sounds": [
                    {
                        "name": n,
                        "cn_name": self.SOUND_CN_MAP.get(n, n),
                        "score": float(s),
                    }
                    for n, s in sound_top5
                ],
                "sound_abnormal": len(hit_sound) > 0,
                "positive_sounds": hit_positive_sound_cn,
                "sound_positive": len(hit_positive_sound) > 0,

                # 兼容 main.py / performance_monitor.py
                # 字段名仍叫 whisper_cost_ms，但实际是 sherpa-onnx ASR 耗时
                "yamnet_cost_ms": round(yamnet_time, 2),
                "whisper_cost_ms": round(asr_time, 2),
                "bert_cost_ms": round(bert_time, 2),
                "audio_total_cost_ms": round(
                    yamnet_time + asr_time + bert_time,
                    2,
                ),
            }

        finally:
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.unlink(temp_wav_path)
                except Exception:
                    pass

    def release(self):
        if hasattr(self, "bert_classifier"):
            self.bert_classifier.release()

        if hasattr(self, "sherpa_recognizer"):
            del self.sherpa_recognizer

        print("🔹 [音频模态] 资源已释放")
