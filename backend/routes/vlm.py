"""
VLM Analysis Routes
Provides GPT-4V based video analysis endpoints
"""

from flask import Blueprint, current_app, request, jsonify
import os
from services.vlm_scorer import VLMScorer
from services.gemini_vlm import GeminiResearchVLM
from services.openai_research_vlm import GPT56TerraResearchVLM
from services.analysis_media import load_analysis_result, resolve_media_path
from domain.context import analysis_results

bp = Blueprint('vlm', __name__, url_prefix='/api/vlm')
vlm_scorer = VLMScorer()
gemini_scorer = GeminiResearchVLM()
gpt56_terra_scorer = GPT56TerraResearchVLM()


def _load_completed_analysis(video_id: str):
    """Resolve persisted media first so gunicorn workers do not lose the result."""
    result = load_analysis_result(current_app.config["UPLOAD_FOLDER"], video_id)
    if result is not None:
        from routes.analyze import _authorize_result_context

        auth_error, status = _authorize_result_context(result)
        if auth_error:
            return None, auth_error, status
        video_path = resolve_media_path(
            current_app.config["UPLOAD_FOLDER"], result, "original_video"
        )
        if video_path is None:
            return None, {"success": False, "error": "원본 영상 파일을 찾을 수 없습니다."}, 404
        return (
            {
                "result": result,
                "video_path": str(video_path),
                "task_type": result.get("video_type") or result.get("task_type"),
                "ml_score": (result.get("updrs_score") or {}).get("total_score"),
            },
            None,
            None,
        )

    # Compatibility fallback for an analysis that is still only in this worker.
    previous_result = analysis_results.get(video_id)
    if previous_result is None:
        return None, {"success": False, "error": f"분석 결과를 찾을 수 없습니다: {video_id}"}, 404
    return (
        {
            "result": previous_result,
            "video_path": previous_result.get("video_path"),
            "task_type": previous_result.get("video_type") or previous_result.get("task_type"),
            "ml_score": (previous_result.get("updrs_score") or {}).get("total_score"),
        },
        None,
        None,
    )


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

        analysis, error, status = _load_completed_analysis(video_id)
        if error:
            return jsonify(error), status
        video_path = analysis["video_path"]
        task_type = analysis["task_type"]
        ml_score = analysis["ml_score"]

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
            if video_id in analysis_results:
                analysis_results[video_id].setdefault("vlm_results", []).append(result)

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


@bp.route('/gemini/status', methods=['GET'])
def get_gemini_status():
    """Expose only safe configuration state; never expose the API key."""
    return jsonify({"success": True, **gemini_scorer.status()})


@bp.route('/gemini/analyze/<video_id>', methods=['POST'])
def analyze_with_gemini(video_id):
    """Run a research-only Gemini observation on an already-authorized video."""
    payload = request.get_json(silent=True) or {}
    if payload.get("research_external_processing_confirmed") is not True:
        return jsonify({
            "success": False,
            "error": "External Gemini processing requires explicit research confirmation."
        }), 400

    analysis, error, status = _load_completed_analysis(video_id)
    if error:
        return jsonify(error), status
    if not analysis["task_type"]:
        return jsonify({"success": False, "error": "검사 유형을 알 수 없습니다."}), 400

    result = gemini_scorer.analyze_video(analysis["video_path"], analysis["task_type"])
    return jsonify(result), 200 if result.get("success") else 502


@bp.route('/gpt56-terra/status', methods=['GET'])
def get_gpt56_terra_status():
    """Expose only safe GPT-5.6 Terra configuration state."""
    return jsonify({"success": True, **gpt56_terra_scorer.status()})


@bp.route('/gpt56-terra/analyze/<video_id>', methods=['POST'])
def analyze_with_gpt56_terra(video_id):
    """Run a frame-sampled research observation with GPT-5.6 Terra."""
    payload = request.get_json(silent=True) or {}
    if payload.get("research_external_processing_confirmed") is not True:
        return jsonify({
            "success": False,
            "error": "External OpenAI processing requires explicit research confirmation."
        }), 400

    analysis, error, status = _load_completed_analysis(video_id)
    if error:
        return jsonify(error), status
    if analysis["task_type"] != "gait":
        return jsonify({
            "success": False,
            "error": "GPT-5.6 Terra comparison is currently limited to gait, matching the C3 experiment scope."
        }), 400
    if not analysis["task_type"]:
        return jsonify({"success": False, "error": "검사 유형을 알 수 없습니다."}), 400

    result = gpt56_terra_scorer.analyze_video(
        analysis["video_path"], analysis["task_type"]
    )
    return jsonify(result), 200 if result.get("success") else 502


@bp.route('/gpt56-terra/evaluate-score/<video_id>', methods=['POST'])
def evaluate_score_with_gpt56_terra(video_id):
    """Run a pre-specified, research-only ordinal gait-score evaluation."""
    payload = request.get_json(silent=True) or {}
    if payload.get("research_external_processing_confirmed") is not True:
        return jsonify({
            "success": False,
            "error": "External OpenAI processing requires explicit research confirmation."
        }), 400
    if payload.get("research_ordinal_score_evaluation_confirmed") is not True:
        return jsonify({
            "success": False,
            "error": "Research ordinal-score evaluation requires explicit confirmation."
        }), 400

    analysis, error, status = _load_completed_analysis(video_id)
    if error:
        return jsonify(error), status
    if analysis["task_type"] != "gait":
        return jsonify({
            "success": False,
            "error": "GPT-5.6 Terra score evaluation is currently limited to gait."
        }), 400

    result = gpt56_terra_scorer.analyze_video(
        analysis["video_path"], analysis["task_type"], include_research_score=True
    )
    return jsonify(result), 200 if result.get("success") else 502


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
