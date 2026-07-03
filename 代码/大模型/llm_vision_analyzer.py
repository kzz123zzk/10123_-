import os
import json
import base64
from openai import OpenAI


client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_alarm_image(
    image_path,
    action_text="未知",
    face_text="未知",
    audio_text="未知",
    transcript="",
    alert_level="未知",
    final_severity=0.0,
):
    img_base64 = image_to_base64(image_path)

    prompt = f"""
你是校园安全监控辅助研判助手。

系统检测结果：
动作异常：{action_text}
表情状态：{face_text}
音频状态：{audio_text}
语音文本：{transcript}
风险等级：{alert_level}
风险分数：{final_severity}

请结合图片和系统检测结果，严格输出 JSON，不要输出 Markdown，不要输出代码块。

重要规则：
1. 如果检测到异常声音，例如爆炸、枪击、尖叫、哭喊、摔砸、警报、玻璃破碎等，即使画面中动作不明显，也必须视为需要重点关注的风险事件。
2. 画面没有明显暴力行为时，不能直接判定安全，只能说明“画面证据不足”。
3. 如果异常声音与系统风险等级同时出现，应提高警惕，建议人工复核或现场确认。
4. 不要把异常声音简单归为误报，除非画面和音频信息都明显正常。
5. 检测到笑声笑脸 时 打架 会被判定为打闹 我的系统会给予低风险告警

JSON 格式如下：
{{
  "event_desc": "事件描述，说明画面中发生了什么",
  "risk_reason": "风险依据，说明为什么触发风险",
  "advice": "处置建议，说明值班人员应该怎么做"
}}

要求：
1. 内容简洁客观；
2. 不要夸大风险；
3. 如果图片中没有明显异常，要说明可能是系统检测结果与画面存在偏差。
"""

    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen3-VL-8B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            max_tokens=512,
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content

        # 校验一下是不是合法 JSON
        data = json.loads(content)

        event_desc = data.get("event_desc", "")
        risk_reason = data.get("risk_reason", "")
        advice = data.get("advice", "")

        return json.dumps({
             "event_desc": event_desc,
             "risk_reason": risk_reason,
             "advice": advice
           }, ensure_ascii=False)

    except Exception as e:
        print(f"⚠️ [大模型Demo] JSON分析失败: {e}")

        return (
            "AI辅助研判：\n"
            "事件描述：大模型分析失败，已保留告警截图。\n"
            f"风险依据：系统检测结果为动作={action_text}，表情={face_text}，音频={audio_text}，风险等级={alert_level}，风险分数={final_severity}。\n"
            "处置建议：建议值班人员查看现场画面并进行人工确认。"
        )
