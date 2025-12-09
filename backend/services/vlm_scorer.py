"""
VLM (Vision Language Model) Scoring Service
Uses GPT-4V to analyze video frames and provide UPDRS scores
"""

import os
import cv2
import base64
import json
from typing import Dict, Any, Optional, List, Tuple
from openai import OpenAI


class VLMScorer:
    """GPT-4V based video analysis for Parkinson's Disease assessment"""

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found. VLM scoring will not work.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)

    def is_available(self) -> bool:
        """Check if VLM service is available"""
        return self.client is not None

    def _get_prompt(self, task_type: str) -> str:
        """Generate task-specific prompt for GPT-4V"""
        criteria = {
            "finger_tapping": "손가락 태핑의 속도, 진폭, 망설임, 멈춤, 진폭 감소를 평가하세요.",
            "hand_movement": "손 열고 닫기의 속도, 진폭, 망설임, 멈춤, 진폭 감소를 평가하세요.",
            "leg_agility": "발 구르기의 속도, 진폭, 망설임, 멈춤, 진폭 감소를 평가하세요.",
            "gait": "걸음걸이의 보폭, 속도, 발 들어올림 높이, 뒤꿈치 착지, 회전, 팔 흔들림을 평가하세요."
        }

        task_criteria = criteria.get(task_type, "움직임의 질과 파킨슨병 징후를 평가하세요.")
        task_name_kr = {
            "finger_tapping": "손가락 태핑",
            "hand_movement": "손 움직임",
            "leg_agility": "다리 민첩성",
            "gait": "보행"
        }.get(task_type, task_type)

        prompt = f"""당신은 파킨슨병 전문 신경과 의사입니다.
환자가 '{task_name_kr}' 검사를 수행하는 영상 프레임들을 분석하세요.
{task_criteria}

MDS-UPDRS 척도로 운동 장애 심각도를 평가하세요:
0: 정상 (문제 없음)
1: 경미 (약간의 느림/작은 진폭, 감소 없음)
2: 가벼움 (가벼운 느림/진폭, 약간의 감소나 망설임)
3: 중등도 (중등도의 느림/진폭, 빈번한 망설임/멈춤)
4: 심각 (심한 장애, 거의 수행 불가)

반드시 아래 JSON 형식으로만 출력하세요:
{{
  "score": <0-4 정수>,
  "confidence": <0.0-1.0 확신도>,
  "findings": [
    "<관찰된 특징 1>",
    "<관찰된 특징 2>",
    "<관찰된 특징 3>"
  ],
  "reasoning": "<점수 부여 근거 설명>"
}}
"""
        return prompt

    def _extract_frames(self, video_path: str, max_frames: int = 12) -> List[Any]:
        """Extract evenly spaced frames from video"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame indices to sample
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = [int(i * total_frames / max_frames) for i in range(max_frames)]

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize to reduce token usage (512x512)
                frame = cv2.resize(frame, (512, 512))
                frames.append(frame)

        cap.release()
        return frames

    def _encode_frame(self, frame) -> str:
        """Encode OpenCV frame to base64"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    def analyze_video(
        self,
        video_path: str,
        task_type: str,
        max_frames: int = 12
    ) -> Dict[str, Any]:
        """
        Analyze video using GPT-4V and return UPDRS score with reasoning.

        Args:
            video_path: Path to the video file
            task_type: Type of motor task (finger_tapping, gait, etc.)
            max_frames: Maximum number of frames to send to API

        Returns:
            Dict with score, confidence, findings, reasoning
        """
        if not self.client:
            return {
                "success": False,
                "error": "VLM 서비스가 설정되지 않았습니다. (API Key Missing)"
            }

        if not os.path.exists(video_path):
            return {
                "success": False,
                "error": f"영상 파일을 찾을 수 없습니다: {video_path}"
            }

        try:
            # Extract frames
            frames = self._extract_frames(video_path, max_frames)

            if not frames:
                return {
                    "success": False,
                    "error": "영상에서 프레임을 추출할 수 없습니다."
                }

            # Build message content
            prompt = self._get_prompt(task_type)
            content = [{"type": "text", "text": prompt}]

            # Add frames as images
            for frame in frames:
                base64_image = self._encode_frame(frame)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low"  # Use low detail to reduce tokens
                    }
                })

            # Call GPT-4V API
            response = self.client.chat.completions.create(
                model="gpt-4o",  # or "gpt-4-vision-preview"
                messages=[{"role": "user", "content": content}],
                max_tokens=500,
                temperature=0.3
            )

            output_text = response.choices[0].message.content

            # Parse JSON response
            try:
                start = output_text.find('{')
                end = output_text.rfind('}') + 1
                json_str = output_text[start:end]
                data = json.loads(json_str)

                return {
                    "success": True,
                    "score": data.get("score", -1),
                    "confidence": data.get("confidence", 0.0),
                    "findings": data.get("findings", []),
                    "reasoning": data.get("reasoning", ""),
                    "raw_output": output_text,
                    "frames_analyzed": len(frames)
                }
            except json.JSONDecodeError:
                return {
                    "success": True,
                    "score": -1,
                    "confidence": 0.0,
                    "findings": [],
                    "reasoning": output_text,
                    "raw_output": output_text,
                    "frames_analyzed": len(frames),
                    "parse_error": True
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"VLM 분석 중 오류 발생: {str(e)}"
            }

    def format_result_for_chat(self, result: Dict[str, Any], ml_score: Optional[int] = None) -> str:
        """Format VLM result for chat display"""
        if not result.get("success"):
            return f"❌ VLM 분석 실패: {result.get('error', '알 수 없는 오류')}"

        score = result.get("score", -1)
        confidence = result.get("confidence", 0.0)
        findings = result.get("findings", [])
        reasoning = result.get("reasoning", "")

        severity_map = {
            0: "정상 (Normal)",
            1: "경미 (Slight)",
            2: "가벼움 (Mild)",
            3: "중등도 (Moderate)",
            4: "심각 (Severe)"
        }
        severity = severity_map.get(score, "알 수 없음")

        response = f"""🔬 **VLM 정밀 분석 결과** (GPT-4V)

📊 **UPDRS 점수: {score}점** ({severity})
🎯 확신도: {confidence*100:.0f}%

**주요 관찰 소견:**
"""
        for i, finding in enumerate(findings, 1):
            response += f"  {i}. {finding}\n"

        response += f"\n**분석 근거:**\n{reasoning}"

        # Compare with ML score if available
        if ml_score is not None:
            diff = abs(score - ml_score)
            if diff == 0:
                response += f"\n\n✅ ML 모델 결과({ml_score}점)와 **일치**합니다."
            elif diff == 1:
                response += f"\n\n⚠️ ML 모델 결과({ml_score}점)와 **1점 차이**가 있습니다."
            else:
                response += f"\n\n🔴 ML 모델 결과({ml_score}점)와 **{diff}점 차이**가 있습니다. 전문의 확인을 권장합니다."

        return response
