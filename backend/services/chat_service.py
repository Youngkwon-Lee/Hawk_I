import os
from openai import OpenAI
from typing import List, Dict, Any

class ChatService:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found. Chat service will not work.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)

    def get_response(self, message: str, context: Dict[str, Any] = None) -> str:
        """
        Get response from ChatGPT based on user message and analysis context.
        """
        try:
            if not self.client:
                return "죄송합니다. AI 서비스가 설정되지 않았습니다. (API Key Missing)"

            # Construct system prompt with context
            system_prompt = """당신은 파킨슨병 진단 보조 AI 'HawkEye'입니다. 
의료진이나 환자가 운동 분석 결과에 대해 물어보면 친절하고 전문적으로 답변해주세요.
단, 당신은 의사가 아니므로 확정적인 진단("파킨슨병입니다")은 피하고, 
"데이터상으로는 ~한 특징이 보입니다"와 같이 객관적인 수치를 기반으로 설명하세요.
"""

            if context:
                system_prompt += f"\n\n[현재 분석된 환자 데이터]\n"
                if 'metrics' in context:
                    metrics = context['metrics']
                    system_prompt += f"- 측정 지표: {metrics}\n"
                if context.get('updrs_score'):
                    score = context['updrs_score']
                    system_prompt += f"- UPDRS 점수: {score.get('score')}점 (심각도: {score.get('severity')})\n"
                if 'video_type' in context:
                    system_prompt += f"- 검사 종류: {context['video_type']}\n"

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating chat response: {e}")
            return f"죄송합니다. 오류가 발생했습니다: {str(e)}"
