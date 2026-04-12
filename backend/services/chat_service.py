import os
import re
import glob
from openai import OpenAI
from typing import List, Dict, Any, Tuple, Optional
from services.vlm_scorer import VLMScorer


class ChatService:
    # Default upload folder
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
    # VLM trigger keywords
    VLM_KEYWORDS = [
        "vlm", "gpt-4v", "gpt4v", "비전", "vision",
        "정밀 분석", "정밀분석", "다시 분석", "재분석",
        "영상 분석", "영상분석", "이미지 분석", "세컨드 오피니언",
        "더 정밀하게", "자세히 분석", "gpt로 분석"
    ]

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found. Chat service will not work.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)

        self.vlm_scorer = VLMScorer()

    def is_vlm_request(self, message: str) -> bool:
        """Check if the message is requesting VLM analysis"""
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in self.VLM_KEYWORDS)

    def get_response(
        self,
        message: str,
        context: Dict[str, Any] = None
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Get response from ChatGPT based on user message and analysis context.
        Returns tuple of (response_text, vlm_result or None)
        """
        try:
            if not self.client:
                return "죄송합니다. AI 서비스가 설정되지 않았습니다. (API Key Missing)", None

            # Check if this is a VLM request
            if self.is_vlm_request(message) and context:
                return self._handle_vlm_request(message, context)

            # Regular chat response
            return self._get_chat_response(message, context), None

        except Exception as e:
            print(f"Error generating chat response: {e}")
            return f"죄송합니다. 오류가 발생했습니다: {str(e)}", None

    def _handle_vlm_request(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Handle VLM analysis request"""
        # Check if VLM is available
        if not self.vlm_scorer.is_available():
            return "죄송합니다. VLM 서비스가 설정되지 않았습니다. (API Key 확인 필요)", None

        # Get video path and task type from context
        video_path = context.get("video_path")
        task_type = context.get("video_type") or context.get("task_type")
        ml_score = context.get("updrs_score", {}).get("score") if context.get("updrs_score") else None
        video_id = context.get("id") or context.get("video_id")

        # If video_path is not provided, try to find it using video_id
        if not video_path and video_id:
            video_path = self._find_video_path(video_id)

        if not video_path:
            return "죄송합니다. 분석할 영상 정보가 없습니다. 먼저 영상 분석을 진행해주세요.", None

        if not os.path.exists(video_path):
            return f"죄송합니다. 원본 영상 파일을 찾을 수 없습니다: {video_path}", None

        if not task_type:
            return "죄송합니다. 검사 유형을 알 수 없습니다.", None

        # Perform VLM analysis
        result = self.vlm_scorer.analyze_video(video_path, task_type)

        if result.get("success"):
            formatted = self.vlm_scorer.format_result_for_chat(result, ml_score)
            return formatted, result
        else:
            error_msg = result.get("error", "알 수 없는 오류")
            return f"VLM 분석 중 오류가 발생했습니다: {error_msg}", None

    def _find_video_path(self, video_id: str) -> Optional[str]:
        """
        Find video path using video_id.
        Videos are saved with pattern: {video_id}_{original_filename}
        """
        if not video_id:
            return None

        # Check if upload folder exists
        if not os.path.exists(self.UPLOAD_FOLDER):
            print(f"Upload folder not found: {self.UPLOAD_FOLDER}")
            return None

        # Search for video files matching the video_id pattern
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.webm', '*.mkv']

        for ext in video_extensions:
            # Pattern: video_id_*.ext (e.g., 12-105182_r_1765329945_*.mp4)
            pattern = os.path.join(self.UPLOAD_FOLDER, f"{video_id}*{ext[1:]}")
            matches = glob.glob(pattern)

            # Filter out result files, skeleton videos, etc.
            for match in matches:
                basename = os.path.basename(match).lower()
                if not any(x in basename for x in ['_result', '_skeleton', '_heatmap', '_temporal']):
                    print(f"Found video for VLM analysis: {match}")
                    return match

        print(f"No video found for video_id: {video_id}")
        return None

    def _get_chat_response(self, message: str, context: Dict[str, Any] = None) -> str:
        """Get regular chat response"""
        # Construct system prompt with context
        system_prompt = """당신은 파킨슨병 진단 보조 AI 'HawkEye'입니다.
의료진이나 환자가 운동 분석 결과에 대해 물어보면 친절하고 전문적으로 답변해주세요.
단, 당신은 의사가 아니므로 확정적인 진단("파킨슨병입니다")은 피하고,
"데이터상으로는 ~한 특징이 보입니다"와 같이 객관적인 수치를 기반으로 설명하세요.

만약 사용자가 "VLM", "정밀 분석", "GPT-4V", "영상 분석" 등을 요청하면,
"VLM 정밀 분석을 시작합니다..." 라고 답하지 마세요. 시스템이 자동으로 처리합니다.
"""

        if context:
            system_prompt += f"\n\n[현재 분석된 환자 데이터]\n"
            if 'metrics' in context:
                metrics = context['metrics']
                # Format metrics nicely
                if isinstance(metrics, dict):
                    system_prompt += "- 측정 지표:\n"
                    for key, value in list(metrics.items())[:10]:  # Limit to 10 items
                        if isinstance(value, float):
                            system_prompt += f"  • {key}: {value:.3f}\n"
                        else:
                            system_prompt += f"  • {key}: {value}\n"
                else:
                    system_prompt += f"- 측정 지표: {metrics}\n"

            if context.get('updrs_score'):
                score = context['updrs_score']
                system_prompt += f"- UPDRS 점수: {score.get('score')}점 (심각도: {score.get('severity')})\n"

            if 'video_type' in context:
                video_type = context['video_type']
                type_names = {
                    "finger_tapping": "손가락 태핑",
                    "gait": "보행 분석",
                    "hand_movement": "손 움직임",
                    "leg_agility": "다리 민첩성"
                }
                system_prompt += f"- 검사 종류: {type_names.get(video_type, video_type)}\n"

            if context.get('ai_interpretation'):
                system_prompt += f"\n[AI 해석 요약]\n{context['ai_interpretation'][:500]}...\n"

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Upgraded from gpt-3.5-turbo
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0.7,
            max_tokens=800
        )

        return response.choices[0].message.content
