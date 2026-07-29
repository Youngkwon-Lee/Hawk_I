"""
Video Analysis Route
ROI Detection + Task Classification
"""

from flask import Blueprint, request, jsonify, current_app, send_from_directory
import os
import cv2
from werkzeug.utils import secure_filename
import sys
import threading
import json
import time
import re
from uuid import UUID

# Add services directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from services.roi_detector import MovementBasedROI, ROIResult
from services.task_classifier import TaskClassifier, TaskClassificationResult
from services.mediapipe_processor import MediaPipeProcessor
from services.metrics_calculator import MetricsCalculator
from services.finger_performability import get_finger_performability_gate
from services.updrs_scorer import UPDRSScorer
from services.interpretation_agent import InterpretationAgent
from services.progress_tracker import init_analysis, update_step, complete_analysis, fail_analysis
from services.supabase_observations import persist_analysis_observation, SupabaseObservationResult
from services.supabase_observations import get_supabase_observation_config
from services.physio_context import (
    PhysioContextError,
    authorize_parkicheck_session,
    authorize_physio_subject,
)
from services.analysis_media import (
    MEDIA_ASSETS,
    load_analysis_result,
    resolve_media_path,
    write_analysis_access_record,
)
from services.supabase_auth import (
    SupabaseAuthUnavailable,
    SupabaseClinicianForbidden,
    SupabaseInvalidToken,
    authenticate_clinician,
    extract_bearer_token,
)
from services.visualization_data_generator import generate_visualization_data, detect_events
from agents.orchestrator import OrchestratorAgent
from domain.context import AnalysisContext
# asdict no longer needed - clinical_scores already converted in ClinicalAgent

bp = Blueprint('analyze', __name__, url_prefix='/api')
ANALYSIS_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,179}$")
PHYSIO_IDENTITY_FIELDS = (
    "physio_subject_person_id",
    "physio_organization_id",
    "physio_created_by_person_id",
    "physio_performer_person_id",
    "physio_subject_display_name",
    "physio_organization_display_name",
)
PHYSIO_PERSISTENCE_OWNERS = frozenset({"hawk_i", "parkicheck"})


class InvalidAnalysisContext(ValueError):
    """Raised when request metadata does not match the integration contract."""


def _optional_text(name: str, max_length: int = 160) -> str | None:
    value = request.form.get(name)
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    if len(value) > max_length:
        raise InvalidAnalysisContext(f"invalid {name}")
    return value


def _require_uuid(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    try:
        return str(UUID(value))
    except (ValueError, AttributeError) as exc:
        raise InvalidAnalysisContext(f"invalid {name}") from exc


def _build_physio_context() -> dict | None:
    """Authorize patient-linked context and canonicalize privileged fields."""
    subject_id = _require_uuid(
        _optional_text("physio_subject_person_id"),
        "physio_subject_person_id",
    )
    supplied_org_id = _require_uuid(
        _optional_text("physio_organization_id"),
        "physio_organization_id",
    )
    activity_session_id = _require_uuid(
        _optional_text("physio_activity_session_id")
        or _optional_text("assessment_session_id"),
        "physio_activity_session_id",
    )
    contract_version = _optional_text("physio_contract_version", max_length=80)
    persistence_owner = _optional_text("physio_persistence_owner", max_length=32)
    if persistence_owner:
        persistence_owner = persistence_owner.lower()
        if persistence_owner not in PHYSIO_PERSISTENCE_OWNERS:
            raise InvalidAnalysisContext("invalid physio_persistence_owner")

    if persistence_owner == "parkicheck":
        access_token = extract_bearer_token(request.headers.get("Authorization"))
        if contract_version != "parkicheck-hawk-i/v1":
            raise InvalidAnalysisContext("invalid ParkiCheck contract version")
        if not subject_id or not supplied_org_id or not activity_session_id:
            raise InvalidAnalysisContext(
                "ParkiCheck analysis requires subject, organization, and activity session"
            )
        config = get_supabase_observation_config()
        if config is None:
            raise SupabaseAuthUnavailable("Supabase authentication is not configured")
        context = authorize_parkicheck_session(
            access_token,
            activity_session_id,
            subject_person_id=subject_id,
            organization_id=supplied_org_id,
            config=config,
        )
        context["contract_version"] = contract_version
        context["persistence_owner"] = persistence_owner
        return context

    has_identity_context = any(_optional_text(name) for name in PHYSIO_IDENTITY_FIELDS)
    if not has_identity_context:
        delegated = {
            "contract_version": contract_version,
            "activity_session_id": activity_session_id,
            "persistence_owner": persistence_owner,
        }
        return {key: value for key, value in delegated.items() if value} or None

    if not subject_id or not supplied_org_id:
        raise InvalidAnalysisContext("patient-linked analysis requires subject and organization")

    access_token = extract_bearer_token(request.headers.get("Authorization"))
    config = get_supabase_observation_config()
    if config is None:
        raise SupabaseAuthUnavailable("Supabase authentication is not configured")
    if supplied_org_id != config.organization_id:
        raise SupabaseClinicianForbidden("organization access denied")

    clinician = authenticate_clinician(access_token, config=config)
    context = authorize_physio_subject(
        clinician,
        subject_id,
        activity_session_id=activity_session_id,
        config=config,
    )
    if contract_version:
        context["contract_version"] = contract_version
    if persistence_owner:
        context["persistence_owner"] = persistence_owner
    return context


def _valid_analysis_id(video_id: str) -> bool:
    return bool(ANALYSIS_ID_PATTERN.fullmatch(video_id)) and ".." not in video_id


def _authorize_result_context(result: dict) -> tuple[dict | None, int | None]:
    context = result.get("physio_context")
    context = context if isinstance(context, dict) else {}
    subject_id = context.get("subject_person_id")
    persistence_owner = context.get("persistence_owner")
    activity_session_id = context.get("activity_session_id")
    is_parkicheck = persistence_owner == "parkicheck"
    if not is_parkicheck and (not isinstance(subject_id, str) or not subject_id.strip()):
        return None, None

    try:
        access_token = extract_bearer_token(request.headers.get("Authorization"))
        config = get_supabase_observation_config()
        if is_parkicheck:
            activity_session_id = str(UUID(str(activity_session_id).strip()))
            canonical_subject_id = (
                str(UUID(subject_id.strip()))
                if isinstance(subject_id, str) and subject_id.strip()
                else None
            )
            canonical_organization_id = context.get("organization_id")
            canonical_organization_id = (
                str(UUID(canonical_organization_id.strip()))
                if isinstance(canonical_organization_id, str)
                and canonical_organization_id.strip()
                else None
            )
            authorize_parkicheck_session(
                access_token,
                activity_session_id,
                subject_person_id=canonical_subject_id,
                organization_id=canonical_organization_id,
                config=config,
            )
        else:
            subject_id = str(UUID(subject_id.strip()))
            clinician = authenticate_clinician(access_token, config=config)
            authorize_physio_subject(clinician, subject_id, config=config)
    except SupabaseInvalidToken:
        return {"success": False, "error": "authentication required"}, 401
    except (SupabaseClinicianForbidden, ValueError):
        return {"success": False, "error": "result access denied"}, 403
    except (SupabaseAuthUnavailable, PhysioContextError):
        return {"success": False, "error": "authentication unavailable"}, 503
    return None, None


def build_score_advisory(video_type: str, performability: dict | None) -> dict | None:
    """Build a simple score interpretation advisory for frontend rendering."""
    if video_type != "finger_tapping" or not performability:
        return None

    status = performability.get("status")
    if status == "performable":
        return {
            "level": "standard",
            "summary": "자동 점수를 표준 해석 흐름으로 사용할 수 있습니다.",
        }
    if status == "uncertain":
        return {
            "level": "review_recommended",
            "summary": "자동 점수는 사용할 수 있지만, 경계 케이스라 수기 검토를 함께 권장합니다.",
        }
    return {
        "level": "reference_only",
        "summary": "자동 점수는 참고용으로만 보고, 수기 판정 또는 재촬영을 우선 권장합니다.",
    }


def build_analysis_trace(
    video_id: str,
    response: dict,
    observation_result: SupabaseObservationResult,
) -> dict:
    activity_session_id = (
        observation_result.activity_session_id
        or response.get("assessment_session_id")
    )
    observation_fhir_id = None
    if observation_result.persistence_owner == "parkicheck" and activity_session_id:
        observation_fhir_id = f"parkicheck-{activity_session_id}"
    elif observation_result.saved:
        observation_fhir_id = f"hawkeye-{video_id}"
    return {
        key: value
        for key, value in {
            "analysis_id": video_id,
            "activity_session_id": activity_session_id,
            "observation_id": observation_result.observation_id,
            "observation_fhir_id": observation_fhir_id,
            "persistence_owner": observation_result.persistence_owner or "hawk_i",
        }.items()
        if value is not None
    }


def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm', 'mkv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_video_background(
    video_path,
    video_id,
    patient_id,
    manual_test_type,
    app_config,
    scoring_method='ensemble',
    ml_model_type='rf',
    physio_context=None,
    medication_context=None,
):
    """
    Background task for video analysis using Multi-Agent Orchestrator
    """
    try:
        print(f"\n{'='*50}")
        print(f"Processing video: {os.path.basename(video_path)} (ID: {video_id})")
        print(f"{'='*50}\n")

        # Initialize Orchestrator
        orchestrator = OrchestratorAgent()
        
        # Define progress callback
        def progress_callback(step_name, status, **kwargs):
            update_step(video_id, step_name, status, **kwargs)

        # Create Context with scoring configuration
        ctx = AnalysisContext(
            video_path=video_path, 
            video_id=video_id,
            scoring_method=scoring_method,
            ml_model_type=ml_model_type,
            manual_test_type=manual_test_type,
        )
        
        # Run Analysis
        ctx = orchestrator.process(ctx, on_progress_update=progress_callback)
        
        if ctx.error:
            raise Exception(ctx.error)

        # Post-processing for Visualizations & Response Formatting
        print("\nStep 4: Generating visualizations...")
        
        # Extract data from context
        landmarks = ctx.skeleton_data.get("landmarks", [])
        metrics = ctx.kinematic_metrics
        updrs_result = ctx.clinical_scores # This is UPDRSScore object
        ai_result = ctx.report # This is Report object
        
        # Convert landmarks to dict for visualization
        # Note: landmarks are already dicts (converted by VisionAgent using asdict)
        frames_data = []
        for lf in landmarks:
            frames_data.append({
                "frame": lf.get("frame_number", 0),
                "timestamp": lf.get("timestamp", 0.0),
                "keypoints": lf.get("landmarks", [])
            })
            
        # Heatmap (Already generated by VisionAgent)
        update_step(video_id, "heatmap", "in_progress")
        heatmap_path = ctx.vision_meta.get("heatmap_path")
        update_step(video_id, "heatmap", "completed")
        
        # Temporal Map (Already generated by VisionAgent)
        update_step(video_id, "temporal_map", "in_progress")
        temporal_path = ctx.vision_meta.get("trajectory_map_path")
        update_step(video_id, "temporal_map", "completed")
        
        # Attention Map (Not yet in VisionAgent, keep or skip?)
        # For now, we skip generating it to save time if it's not critical, 
        # or we could move it to VisionAgent later. 
        # Let's keep it if it was there, but we need viz_service.
        # Since we removed viz_service init, let's skip it for now or init it just for this.
        # Actually, let's remove it to streamline.
        attention_path = None

        # Overlay Video - Use skeleton video generated by VisionAgent
        skeleton_video_path = ctx.vision_meta.get("skeleton_video_path")
        if skeleton_video_path and os.path.exists(skeleton_video_path):
            skeleton_video_url = f"/files/{os.path.basename(skeleton_video_path)}"
        else:
            # Fallback to original video if skeleton not available
            skeleton_video_url = f"/files/{os.path.basename(video_path)}"

        # Original video URL (for canvas overlay mode)
        original_video_url = f"/files/{os.path.basename(video_path)}"

        update_step(video_id, "overlay_video", "completed", result_url=skeleton_video_url)

        # Prepare Response
        
        # clinical_scores is already a dict (converted in ClinicalAgent)
        updrs_dict = updrs_result if updrs_result else None
        
        ai_interpretation = None
        if ai_result:
            ai_interpretation = {
                "summary": ai_result.summary_for_patient,
                "explanation": ai_result.summary_for_clinician,
                "recommendations": ai_result.recommendations
            }

        # Reasoning Log conversion - transform to frontend format
        reasoning_log = []
        for step in ctx.reasoning_log:
            step_dict = step.model_dump(mode='json')
            # Map backend fields to frontend fields
            reasoning_log.append({
                "agent": step_dict.get("step", "unknown"),  # Backend 'step' -> Frontend 'agent'
                "step": step_dict.get("step", ""),  # Keep step for display
                "content": step_dict.get("message", ""),  # Backend 'message' -> Frontend 'content'
                "timestamp": step_dict.get("timestamp", ""),
                "meta": step_dict.get("meta")
            })

        # Generate visualization data for charts
        fps = ctx.vision_meta.get("fps", 30.0)
        # Get gait analysis from gait_cycle_data (populated by GaitCycleAgent)
        gait_analysis = getattr(ctx, 'gait_cycle_data', None)

        # Determine task type for visualization
        viz_task_type = "finger_tapping" if "finger" in ctx.task_type.lower() or "tapping" in ctx.task_type.lower() else "gait"

        visualization_data = generate_visualization_data(
            landmark_frames=frames_data,
            gait_analysis=gait_analysis,
            fps=fps,
            task_type=viz_task_type
        )

        # Detect clinically relevant events
        event_task_type = viz_task_type  # Reuse the same task type
        detected_events = detect_events(
            landmark_frames=frames_data,
            gait_analysis=gait_analysis,
            fps=fps,
            task_type=event_task_type
        )

        performability = None
        if viz_task_type == "finger_tapping" and metrics:
            try:
                detection_rate = None
                total_frames = max(1, ctx.vision_meta.get("frame_count", 0))
                if landmarks:
                    detection_rate = len(landmarks) / total_frames
                performability = get_finger_performability_gate().assess(
                    metrics,
                    detection_rate=detection_rate,
                ).__dict__
            except Exception as perf_exc:
                print(f"[WARN] Finger performability assessment failed: {perf_exc}")

        score_advisory = build_score_advisory(viz_task_type, performability)

        response = {
            "success": True,
            "id": video_id,
            "patient_id": patient_id,
            "physio_context": physio_context or None,
            "assessment_session_id": (physio_context or {}).get("activity_session_id"),
            "timeline_contract_version": (physio_context or {}).get("contract_version"),
            "medication_context": medication_context or None,
            "video_type": ctx.task_type,
            "auto_detected": manual_test_type is None,
            "confidence": ctx.vision_meta.get("confidence", 0.0),
            "roi": {
                "x": ctx.vision_meta.get("roi", (0,0,0,0))[0],
                "y": ctx.vision_meta.get("roi", (0,0,0,0))[1],
                "w": ctx.vision_meta.get("roi", (0,0,0,0))[2],
                "h": ctx.vision_meta.get("roi", (0,0,0,0))[3]
            },
            "motion_analysis": {
                "motion_pattern": ctx.vision_meta.get("motion_pattern"),
                "motion_area_ratio": ctx.vision_meta.get("motion_area_ratio"),
                "body_part": ctx.vision_meta.get("body_part")
            },
            "reasoning": ctx.vision_meta.get("reasoning", ""), # Legacy single string
            "reasoning_log": reasoning_log, # NEW: Full log
            "video_metadata": {
                "width": 0, # TODO: Get from meta
                "height": 0,
                "fps": ctx.vision_meta.get("fps", 0),
                "duration": 0,
                "total_frames": ctx.vision_meta.get("frame_count", 0)
            },
            "metrics": metrics,
            "skeleton_data": {
                "total_frames": len(landmarks),
                "detection_rate": 0, # Calculate if needed
                "mode": "pose" if ctx.task_type in ["gait", "leg_agility"] else "hand",
                "skeleton_video_url": skeleton_video_url,
                "original_video_url": original_video_url,  # For canvas overlay mode
                "keypoints": frames_data,  # For frontend canvas overlay
                "fps": ctx.vision_meta.get("fps", 30.0)  # For video sync
            },
            "scoring_method": ctx.scoring_method,
            "ml_model_type": ctx.ml_model_type,
            "performability_assessment": performability,
            "score_advisory": score_advisory,
            "updrs_score": updrs_dict,
            "ai_interpretation": ai_interpretation,
            "events": detected_events,
            "visualization_urls": {
                "heatmap": f"/files/{os.path.basename(heatmap_path)}" if heatmap_path else None,
                "temporal_map": f"/files/{os.path.basename(temporal_path)}" if temporal_path else None,
                "attention_map": None
            },
            "visualization_data": visualization_data,
            "gait_cycle_analysis": gait_analysis  # Include raw gait cycle data
        }

        observation_result = persist_analysis_observation(response)
        response["integrations"] = {
            "supabase_observation": observation_result.as_public_dict()
        }
        response["analysis_trace"] = build_analysis_trace(
            video_id,
            response,
            observation_result,
        )

        # Save result
        result_path = os.path.join(app_config['UPLOAD_FOLDER'], f"{video_id}_result.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)

        # Mark analysis as completed
        update_step(video_id, "updrs_calculation", "completed")
        update_step(video_id, "ai_interpretation", "completed")
        complete_analysis(video_id)

        print(f"\n{'='*50}")
        print(f"Analysis Complete! (ID: {video_id})")

    except Exception as e:
        print(f"\nError during analysis:")
        print(f"  {str(e)}\n")
        import traceback
        with open('error.log', 'w') as f:
            f.write(f"Error: {str(e)}\n")
            traceback.print_exc(file=f)
        traceback.print_exc()
        fail_analysis(video_id, str(e))


@bp.route('/analyze', methods=['POST'])
def start_analysis():
    """
    Start asynchronous video analysis
    """
    try:
        # Check if video file is present
        if 'video_file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No video file provided"
            }), 400

        video_file = request.files['video_file']

        if video_file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400

        if not allowed_file(video_file.filename):
            return jsonify({
                "success": False,
                "error": "Invalid file type. Allowed: mp4, avi, mov, webm, mkv"
            }), 400

        try:
            physio_context = _build_physio_context()
        except InvalidAnalysisContext as exc:
            return jsonify({"success": False, "error": str(exc)}), 400
        except SupabaseInvalidToken:
            return jsonify({"success": False, "error": "authentication required"}), 401
        except SupabaseClinicianForbidden:
            return jsonify({"success": False, "error": "patient access denied"}), 403
        except SupabaseAuthUnavailable:
            return jsonify({"success": False, "error": "authentication unavailable"}), 503
        except PhysioContextError:
            return jsonify({"success": False, "error": "patient authorization unavailable"}), 502

        try:
            patient_id = (
                (physio_context or {}).get("subject_person_id")
                or _optional_text("patient_id")
                or 'unknown'
            )
        except InvalidAnalysisContext as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

        # Generate unique video_id for progress tracking
        filename = secure_filename(video_file.filename)
        from uuid import uuid4

        video_id = f"{os.path.splitext(filename)[0]}_{int(time.time())}_{uuid4().hex[:12]}"

        # Persist the authorization boundary before the original video is saved.
        # This closes the processing-time window where /files could otherwise
        # serve a patient upload before the final result JSON exists.
        write_analysis_access_record(
            current_app.config['UPLOAD_FOLDER'],
            video_id,
            physio_context or None,
        )
        
        # Save video file
        # Note: We need to save it here before starting the thread
        video_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{video_id}_{filename}")
        video_file.save(video_path)

        # Initialize progress tracking
        init_analysis(video_id, task_type="auto_detect")

        # Get optional parameters
        medication_context = None
        medication_context_raw = request.form.get("medication_context")
        if medication_context_raw:
            try:
                parsed_medication_context = json.loads(medication_context_raw)
                if isinstance(parsed_medication_context, dict):
                    medication_context = parsed_medication_context
            except (TypeError, ValueError, json.JSONDecodeError):
                medication_context = None
        manual_test_type = request.form.get('test_type', None)
        # Scoring methods:
        # - 'coral': CORAL Ordinal Regression with Mamba (Best: Gait 0.790, Finger 0.553, Hand 0.598)
        # - 'rule': Rule-based scoring with PD4T-calibrated thresholds
        # - 'ml': ML scoring (RF/XGBoost/feature_baseline/feature_baseline_seq on kinematic features)
        # - 'ensemble': Rule + ML average
        scoring_method = request.form.get('scoring_method', 'coral')  # coral (default), rule, ml, ensemble
        ml_model_type = request.form.get('ml_model_type', 'rf')  # rf, xgb, ordinal, feature_baseline, or feature_baseline_seq
        
        # Start background thread
        thread = threading.Thread(
            target=process_video_background,
            args=(
                video_path,
                video_id,
                patient_id,
                manual_test_type,
                current_app.config.copy(),
                scoring_method,
                ml_model_type,
                physio_context or None,
                medication_context,
            )
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            "success": True,
            "message": "Analysis started",
            "id": video_id,
            "status": "in_progress"
        }), 202

    except Exception as e:
        print(f"Error starting analysis: {e}")
        return jsonify({
            "success": False,
            "error": "Unable to start analysis"
        }), 500


@bp.route('/analysis/result/<video_id>', methods=['GET'])
def get_analysis_result(video_id):
    """
    Get the final result of an analysis
    """
    if not _valid_analysis_id(video_id):
        return jsonify({"success": False, "error": "Invalid analysis ID"}), 400

    upload_folder = os.path.abspath(current_app.config['UPLOAD_FOLDER'])
    result_path = os.path.abspath(os.path.join(upload_folder, f"{video_id}_result.json"))
    if os.path.commonpath([upload_folder, result_path]) != upload_folder:
        return jsonify({"success": False, "error": "Invalid analysis ID"}), 400
    
    if not os.path.exists(result_path):
        return jsonify({
            "success": False,
            "error": "Result not found or analysis not complete"
        }), 404
        
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        auth_error, status = _authorize_result_context(result)
        if auth_error:
            return jsonify(auth_error), status
        response = jsonify(result)
        response.headers["Cache-Control"] = "no-store, private"
        return response
    except Exception as e:
        print(f"Error reading analysis result: {e}")
        return jsonify({
            "success": False,
            "error": "Error reading result"
        }), 500


@bp.route('/analysis/media/<video_id>/<asset>', methods=['GET', 'HEAD'])
def get_analysis_media(video_id, asset):
    """Serve one allowlisted analysis asset after result authorization."""
    if not _valid_analysis_id(video_id) or asset not in MEDIA_ASSETS:
        return jsonify({"success": False, "error": "Invalid media request"}), 400

    upload_folder = os.path.abspath(current_app.config['UPLOAD_FOLDER'])
    result = load_analysis_result(upload_folder, video_id)
    if result is None:
        return jsonify({"success": False, "error": "Result not found"}), 404

    auth_error, status = _authorize_result_context(result)
    if auth_error:
        return jsonify(auth_error), status

    media_path = resolve_media_path(upload_folder, result, asset)
    if media_path is None:
        return jsonify({"success": False, "error": "Media not found"}), 404

    response = send_from_directory(
        upload_folder,
        media_path.name,
        conditional=True,
    )
    response.headers["Accept-Ranges"] = "bytes"
    response.headers["Cache-Control"] = "no-store, private"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@bp.route('/analyze-status/<analysis_id>', methods=['GET'])
def get_analysis_status(analysis_id):
    """
    Get analysis status (for async processing - future implementation)
    """
    # This is now redundant with progress_tracker but kept for compatibility
    return jsonify({
        "success": False,
        "error": "Use /api/analysis/progress/<video_id> instead"
    }), 301
