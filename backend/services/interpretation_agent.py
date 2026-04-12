"""
Interpretation Agent (Level 1)
Translates technical UPDRS scores and metrics into patient-friendly explanations
Uses OpenAI GPT for natural language generation
"""

import os
from openai import OpenAI
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class InterpretationResult:
    """Patient-friendly interpretation of analysis results"""
    summary: str  # 한 줄 요약
    explanation: str  # 상세 설명
    recommendations: list[str]  # 권장사항 (3-5개)
    raw_response: str  # LLM 원본 응답


class InterpretationAgent:
    """
    Level 1 Agent: Simple pipeline-based interpretation
    Input: UPDRS score + metrics → Output: Patient-friendly explanation
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI client

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("[WARNING] OpenAI API key not found. Using fallback interpretation.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
            
        self.model = "gpt-4o-mini"  # Cost-effective model for interpretation

    def interpret_finger_tapping(
        self,
        updrs_score: float,
        severity: str,
        metrics: Dict[str, Any],
        details: Dict[str, Any],
        clinical_charts: Optional[str] = None
    ) -> InterpretationResult:
        """
        Interpret finger tapping results for patients

        Args:
            updrs_score: Total UPDRS score (0-4)
            severity: Severity level (Normal, Slight, Mild, Moderate, Severe)
            metrics: Calculated metrics (speed, amplitude, etc.)
            details: UPDRS scoring details

        Returns:
            InterpretationResult with patient-friendly explanation
        """

        # Build context-rich prompt
        prompt = f"""당신은 파킨슨병 환자를 돕는 친절한 의료 AI 어시스턴트입니다.
손가락 태핑 검사 결과를 환자가 이해하기 쉽게 설명해주세요.

## 검사 결과
- UPDRS 점수: {updrs_score} (0-4 척도)
- 중증도: {severity}
- 태핑 속도: {metrics.get('tapping_speed', 0):.2f} Hz (정상: ≥3.0 Hz)
- 진폭: {metrics.get('amplitude_mean', 0):.2f}× (정상: >0.8, 검지손가락 길이 대비)
- 리듬 변동성: {metrics.get('rhythm_variability', 0):.1f}% (정상: <15%)
- 피로율: {metrics.get('fatigue_rate', 0):.1f}% (정상: <20%)
- 주저함: {metrics.get('hesitation_count', 0)}회 (정상: ≤2회)
- 총 탭 수: {metrics.get('total_taps', 0)}
"""
        
        if clinical_charts:
            prompt += f"\n## 상세 차트 데이터\n{clinical_charts}"

        prompt += """

## 요청사항
다음 형식으로 JSON 응답을 생성해주세요:

{{
  "summary": "한 문장으로 요약 (예: 손가락 움직임이 정상 범위입니다)",
  "explanation": "200-300자 정도로 상세 설명. 다음을 포함:
    1. 무엇을 측정했는지
    2. 각 지표가 의미하는 것 (속도, 진폭, 리듬 등)
    3. 정상 범위와 비교했을 때 어떤지
    주의: 전문용어 최소화, 친근한 톤, 구체적 수치 포함",
  "recommendations": [
    "구체적인 조언 1 (예: 매일 5분씩 손가락 운동 연습)",
    "구체적인 조언 2 (예: 일정한 리듬으로 연습하기)",
    "구체적인 조언 3 (예: 진폭을 크게 유지하도록 의식하기)"
  ]
}}

**중요**: 반드시 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful medical AI assistant specializing in Parkinson's disease. Provide clear, empathetic, and accurate interpretations in Korean. Always respond in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=800,
                timeout=15.0,  # 15초 timeout 추가
                response_format={"type": "json_object"}  # Force JSON response
            )

            raw_response = response.choices[0].message.content

            # Parse JSON response
            import json
            parsed = json.loads(raw_response)

            return InterpretationResult(
                summary=parsed.get("summary", "결과 요약을 생성할 수 없습니다."),
                explanation=parsed.get("explanation", "상세 설명을 생성할 수 없습니다."),
                recommendations=parsed.get("recommendations", []),
                raw_response=raw_response
            )

        except Exception as e:
            print(f"[ERROR] Interpretation Agent Error: {str(e)}")
            # Fallback to rule-based interpretation
            return self._fallback_interpretation_finger_tapping(
                updrs_score, severity, metrics
            )

    def interpret_gait(
        self,
        updrs_score: float,
        severity: str,
        metrics: Dict[str, Any],
        details: Dict[str, Any],
        clinical_charts: Optional[str] = None
    ) -> InterpretationResult:
        """
        Interpret gait results for patients

        Args:
            updrs_score: Total UPDRS score (0-4)
            severity: Severity level
            metrics: Calculated gait metrics
            details: UPDRS scoring details

        Returns:
            InterpretationResult with patient-friendly explanation
        """

        prompt = f"""당신은 파킨슨병 환자를 돕는 친절한 의료 AI 어시스턴트입니다.
보행 검사 결과를 환자가 이해하기 쉽게 설명해주세요.

## 검사 결과
- UPDRS 점수: {updrs_score} (0-4 척도)
- 중증도: {severity}
- 보행 속도: {metrics.get('walking_speed', 0):.2f} m/s (정상: ≥0.8 m/s)
- 보행률: {metrics.get('cadence', 0):.0f} steps/min (정상: 100-120)
- 보폭 길이: {metrics.get('stride_length', 0):.2f} m
- 보폭 변동성: {metrics.get('stride_variability', 0):.1f}% (정상: <10%)
- 팔 흔들기 비대칭: {metrics.get('arm_swing_asymmetry', 0):.1f}% (정상: <10%)
- 총 걸음 수: {metrics.get('step_count', 0)}
"""

        if clinical_charts:
            prompt += f"\n## 상세 차트 데이터\n{clinical_charts}"

        prompt += """

## 요청사항
다음 형식으로 JSON 응답을 생성해주세요:

{{
  "summary": "한 문장으로 요약",
  "explanation": "200-300자 정도로 상세 설명. 보행 속도, 보폭, 리듬, 팔 흔들기가 각각 의미하는 것을 포함",
  "recommendations": [
    "구체적인 조언 1 (예: 보폭을 크게 하는 연습)",
    "구체적인 조언 2 (예: 팔을 자연스럽게 흔들며 걷기)",
    "구체적인 조언 3"
  ]
}}

**중요**: 반드시 JSON 형식으로만 응답하세요."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful medical AI assistant specializing in Parkinson's disease. Provide clear, empathetic, and accurate interpretations in Korean. Always respond in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=800,
                timeout=15.0,  # 15초 timeout 추가
                response_format={"type": "json_object"}
            )

            raw_response = response.choices[0].message.content

            import json
            parsed = json.loads(raw_response)

            return InterpretationResult(
                summary=parsed.get("summary", "결과 요약을 생성할 수 없습니다."),
                explanation=parsed.get("explanation", "상세 설명을 생성할 수 없습니다."),
                recommendations=parsed.get("recommendations", []),
                raw_response=raw_response
            )

        except Exception as e:
            print(f"[ERROR] Interpretation Agent Error: {str(e)}")
            return self._fallback_interpretation_gait(updrs_score, severity, metrics)

    def _fallback_interpretation_finger_tapping(
        self,
        updrs_score: float,
        severity: str,
        metrics: Dict[str, Any]
    ) -> InterpretationResult:
        """Fallback rule-based interpretation if API fails"""

        if severity == "Normal":
            summary = "손가락 움직임이 정상 범위입니다."
            explanation = f"태핑 속도 {metrics.get('tapping_speed', 0):.1f}Hz, 진폭 {metrics.get('amplitude_mean', 0):.1f}로 정상 범위에 있습니다. 현재 상태를 잘 유지하고 계십니다."
            recommendations = ["현재 상태 유지", "규칙적인 운동 지속", "정기 검진 권장"]
        elif severity == "Slight":
            summary = "경미한 운동 변화가 관찰됩니다."
            explanation = f"태핑 속도가 {metrics.get('tapping_speed', 0):.1f}Hz로 약간 느려졌습니다. 조기 단계에서 관리하면 호전될 수 있습니다."
            recommendations = ["매일 손가락 운동", "리듬감 있게 연습", "전문의 상담 고려"]
        else:
            summary = f"{severity} 운동 증상이 있습니다."
            explanation = f"UPDRS 점수 {updrs_score}로 운동 기능 저하가 있습니다. 전문의 상담이 필요합니다."
            recommendations = ["즉시 전문의 상담", "약물 치료 검토", "재활 운동 고려"]

        return InterpretationResult(
            summary=summary,
            explanation=explanation,
            recommendations=recommendations,
            raw_response="Fallback interpretation (API unavailable)"
        )

    def _fallback_interpretation_gait(
        self,
        updrs_score: float,
        severity: str,
        metrics: Dict[str, Any]
    ) -> InterpretationResult:
        """Fallback rule-based interpretation for gait"""

        if severity == "Normal":
            summary = "보행 능력이 정상 범위입니다."
            explanation = f"보행 속도 {metrics.get('walking_speed', 0):.2f}m/s로 양호합니다."
            recommendations = ["현재 보행 습관 유지", "규칙적인 걷기 운동", "균형 운동 추가"]
        else:
            summary = f"{severity} 보행 증상이 관찰됩니다."
            explanation = f"보행 속도와 보폭에 변화가 있습니다. UPDRS 점수 {updrs_score}입니다."
            recommendations = ["보행 재활 운동", "전문의 상담", "낙상 예방 조치"]

        return InterpretationResult(
            summary=summary,
            explanation=explanation,
            recommendations=recommendations,
            raw_response="Fallback interpretation (API unavailable)"
        )


# Example usage
if __name__ == "__main__":
    # Test with mock data
    agent = InterpretationAgent()

    # Finger tapping test
    result = agent.interpret_finger_tapping(
        updrs_score=1.2,
        severity="Slight",
        metrics={
            "tapping_speed": 2.5,
            "amplitude_mean": 0.75,
            "rhythm_variability": 18.0,
            "fatigue_rate": 25.0,
            "hesitation_count": 3,
            "total_taps": 25
        },
        details={}
    )

    print("\n=== Finger Tapping Interpretation ===")
    print(f"Summary: {result.summary}")
    print(f"Explanation: {result.explanation}")
    print(f"Recommendations: {result.recommendations}")
