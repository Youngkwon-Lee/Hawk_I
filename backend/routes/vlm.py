"""
VLM Analysis Routes
Provides GPT-4V based video analysis endpoints
"""

from flask import Blueprint, request, jsonify
import os
from services.vlm_scorer import VLMScorer
from domain.context import analysis_results

bp = Blueprint('vlm', __name__, url_prefix='/api/vlm')
vlm_scorer = VLMScorer()


@bp.route('/status', methods=['GET'])
def get_status():
    """Check if VLM service is available"""
    return jsonify({
        "success": True,
        "available": vlm_scorer.is_available(),
        "model": "gpt-4o" if vlm_scorer.is_available() else None
    })


@bp.route('/analyze/<video_id>', methods=['POST'])
def analyze_video(video_id):
    """
    Analyze a previously uploaded video using VLM (GPT-4V)

    Args:
        video_id: ID of the video from previous analysis

    Returns:
        VLM analysis result with UPDRS score and reasoning
    """
    try:
        if not vlm_scorer.is_available():
            return jsonify({
                "success": False,
                "error": "VLM 서비스가 설정되지 않았습니다. OPENAI_API_KEY를 확인하세요."
            }), 503

        # Get previous analysis result to find video path and task type
        if video_id not in analysis_results:
            return jsonify({
                "success": False,
                "error": f"분석 결과를 찾을 수 없습니다: {video_id}"
            }), 404

        previous_result = analysis_results[video_id]
        video_path = previous_result.get("video_path")
        task_type = previous_result.get("video_type") or previous_result.get("task_type")
        ml_score = previous_result.get("updrs_score", {}).get("score")

        if not video_path or not os.path.exists(video_path):
            return jsonify({
                "success": False,
                "error": "원본 영상 파일을 찾을 수 없습니다."
            }), 404

        if not task_type:
            return jsonify({
                "success": False,
                "error": "검사 유형을 알 수 없습니다."
            }), 400

        # Perform VLM analysis
        result = vlm_scorer.analyze_video(video_path, task_type)

        if result.get("success"):
            # Format for chat display
            formatted = vlm_scorer.format_result_for_chat(result, ml_score)

            # Store VLM result in analysis_results
            if "vlm_results" not in analysis_results[video_id]:
                analysis_results[video_id]["vlm_results"] = []
            analysis_results[video_id]["vlm_results"].append(result)

            return jsonify({
                "success": True,
                "data": {
                    "score": result.get("score"),
                    "confidence": result.get("confidence"),
                    "findings": result.get("findings"),
                    "reasoning": result.get("reasoning"),
                    "frames_analyzed": result.get("frames_analyzed"),
                    "ml_score": ml_score,
                    "formatted_response": formatted
                }
            })
        else:
            return jsonify({
                "success": False,
                "error": result.get("error", "VLM 분석 실패")
            }), 500

    except Exception as e:
        print(f"VLM analysis error: {e}")
        return jsonify({
            "success": False,
            "error": f"VLM 분석 중 오류 발생: {str(e)}"
        }), 500


@bp.route('/analyze-direct', methods=['POST'])
def analyze_direct():
    """
    Analyze a video file directly (for testing)
    Expects multipart form with 'video' file and 'task_type' field
    """
    try:
        if not vlm_scorer.is_available():
            return jsonify({
                "success": False,
                "error": "VLM 서비스가 설정되지 않았습니다."
            }), 503

        if 'video' not in request.files:
            return jsonify({
                "success": False,
                "error": "비디오 파일이 필요합니다."
            }), 400

        video = request.files['video']
        task_type = request.form.get('task_type', 'finger_tapping')

        # Save temporarily
        temp_path = f"/tmp/vlm_temp_{video.filename}"
        video.save(temp_path)

        try:
            result = vlm_scorer.analyze_video(temp_path, task_type)
            return jsonify({
                "success": result.get("success", False),
                "data": result
            })
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
