"""
Video Analysis Route
ROI Detection + Task Classification
"""

from flask import Blueprint, request, jsonify, current_app
import os
import cv2
from werkzeug.utils import secure_filename
import sys
import threading
import json
import time

# Add services directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from services.roi_detector import MovementBasedROI, ROIResult
from services.task_classifier import TaskClassifier, TaskClassificationResult
from services.mediapipe_processor import MediaPipeProcessor
from services.metrics_calculator import MetricsCalculator
from services.updrs_scorer import UPDRSScorer
from services.interpretation_agent import InterpretationAgent
from services.progress_tracker import init_analysis, update_step, complete_analysis, fail_analysis

bp = Blueprint('analyze', __name__, url_prefix='/api')


def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm', 'mkv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_video_background(video_path, video_id, patient_id, manual_test_type, app_config):
    """
    Background task for video analysis
    """
    # Create a new app context for the background thread
    # (Not strictly needed here since we passed config, but good practice if using Flask extensions)
    
    try:
        print(f"\n{'='*50}")
        print(f"📹 Processing video: {os.path.basename(video_path)} (ID: {video_id})")
        print(f"{'='*50}\n")

        # Get video metadata
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        print(f"Video Info:")
        print(f"  Resolution: {frame_width}x{frame_height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Total Frames: {total_frames}\n")

        # Initialize ROI detector and task classifier
        roi_detector = MovementBasedROI(fps=int(fps))
        task_classifier = TaskClassifier()

        # Step 1: Detect ROI
        print("Step 1: Detecting ROI...")
        update_step(video_id, "roi_detection", "in_progress")
        roi_result = roi_detector.detect(video_path, num_frames=30)
        update_step(video_id, "roi_detection", "completed")

        # Step 2: Classify task
        print("\nStep 2: Classifying task...")
        task_result = task_classifier.classify(roi_result, (frame_width, frame_height))

        # Use manual override if provided, otherwise use auto-detected
        final_video_type = manual_test_type if manual_test_type else task_result.task_type

        # Step 3: MediaPipe Skeleton Extraction and Metrics Calculation
        print("\nStep 3: Extracting skeleton and calculating metrics...")
        update_step(video_id, "skeleton", "in_progress")

        metrics = None
        skeleton_data = None
        updrs_score_data = None
        ai_interpretation = None

        try:
            # Determine mode based on video type
            if final_video_type in ["finger_tapping", "hand_movement", "pronation_supination"]:
                mode = "hand"
            elif final_video_type in ["gait", "leg_agility"]:
                mode = "pose"
            else:
                mode = "hand"  # default

            # Extract skeleton landmarks
            processor = MediaPipeProcessor(mode=mode)

            # Apply ROI for better detection
            roi_tuple = (task_result.roi[0], task_result.roi[1],
                        task_result.roi[2], task_result.roi[3])

            # Generate output video path for skeleton overlay
            filename = os.path.basename(video_path)
            base_name = os.path.splitext(filename)[0]
            skeleton_video_path = os.path.join(app_config['UPLOAD_FOLDER'], f"{base_name}_skeleton.mp4")

            landmark_frames = processor.process_video(video_path, roi=roi_tuple, output_video_path=skeleton_video_path)

            print(f"  Extracted landmarks from {len(landmark_frames)} frames")
            update_step(video_id, "skeleton", "completed")
            
            # Step 4: Visualizations (Heatmap, Temporal, Attention)
            print("\nStep 4: Generating visualizations...")
            
            from services.visualization import VisualizationService
            viz_service = VisualizationService()
            
            # Convert landmark frames to dict format
            frames_data = []
            for lf in landmark_frames:
                frames_data.append({
                    "frame": lf.frame_number,
                    "timestamp": lf.timestamp,
                    "keypoints": lf.landmarks
                })
            
            # Heatmap with overlay
            update_step(video_id, "heatmap", "in_progress")
            heatmap_path = os.path.join(app_config['UPLOAD_FOLDER'], f"{video_id}_heatmap.png")
            viz_service.generate_heatmap(frames_data, heatmap_path, video_path=video_path)
            update_step(video_id, "heatmap", "completed")
            
            # Temporal Map with overlay
            update_step(video_id, "temporal_map", "in_progress")
            temporal_path = os.path.join(app_config['UPLOAD_FOLDER'], f"{video_id}_temporal.png")
            viz_service.generate_temporal_map(frames_data, temporal_path, mode=mode, video_path=video_path)
            update_step(video_id, "temporal_map", "completed")
            
            # Attention Map
            update_step(video_id, "attention_map", "in_progress")
            attention_path = os.path.join(app_config['UPLOAD_FOLDER'], f"{video_id}_attention.png")
            viz_service.generate_attention_map(frames_data, attention_path)
            update_step(video_id, "attention_map", "completed")

            # Overlay Video (already generated by MediaPipeProcessor)
            update_step(video_id, "overlay_video", "completed", result_url=f"/files/{os.path.basename(skeleton_video_path)}")

            # Set skeleton_data (even if no landmarks detected)
            skeleton_data = {
            "total_frames": len(landmark_frames),
            "detection_rate": len(landmark_frames) / total_frames if total_frames > 0 else 0,
            "mode": mode,
            "skeleton_video_url": f"/files/{os.path.basename(skeleton_video_path)}"
            }
            # Calculate metrics
            if len(landmark_frames) > 0:
                print("\nStep 5: Calculating metrics...")
                update_step(video_id, "metrics", "in_progress")

                from services.metrics_calculator import MetricsCalculator
                metrics_calc = MetricsCalculator()

                metrics = metrics_calc.calculate_metrics(
                    video_type=final_video_type,
                    landmark_frames=landmark_frames,
                    fps=fps
                )

                update_step(video_id, "metrics", "completed")
                print(f"  ✓ Metrics calculated: {list(metrics.keys())}")

                # Detect timeline events
                print("\nDetecting timeline events...")
                from services.event_detector import EventDetector
                event_detector = EventDetector()
                events = event_detector.detect_events(
                    video_type=final_video_type,
                    metrics=metrics,
                    landmark_frames=landmark_frames,
                    fps=fps
                )
                print(f"  ✓ Detected {len(events)} events")
            else:
                events = []

                
            # UPDRS Scoring and AI Interpretation
            if metrics:
                from services.updrs_scorer import UPDRSScorer
                scorer = UPDRSScorer()
                
                if final_video_type == "finger_tapping":
                    # Reconstruct metrics object for UPDRS
                    from services.metrics_calculator import FingerTappingMetrics
                    metrics_obj = FingerTappingMetrics(
                        tapping_speed=metrics.get("tapping_speed", 0),
                        amplitude_mean=metrics.get("amplitude_mean", 0),
                        amplitude_std=metrics.get("amplitude_std", 0),
                        rhythm_variability=metrics.get("rhythm_variability", 0),
                        fatigue_rate=metrics.get("fatigue_rate", 0),
                        hesitation_count=metrics.get("hesitation_count", 0),
                        total_taps=metrics.get("total_taps", 0),
                        duration=metrics.get("duration", 0)
                    )

                    # Calculate UPDRS score
                    update_step(video_id, "updrs_calculation", "in_progress")
                    updrs_result = scorer.score_finger_tapping(metrics_obj)
                    updrs_score_data = {
                        "score": updrs_result.total_score,
                        "severity": updrs_result.severity,
                        "base_score": updrs_result.base_score,
                        "penalties": updrs_result.penalties,
                        "details": updrs_result.details
                    }
                    update_step(video_id, "updrs_calculation", "completed")

                    print(f"  Tapping Speed: {metrics['tapping_speed']} Hz")
                    print(f"  Amplitude: {metrics['amplitude_mean']} px")
                    print(f"  Fatigue Rate: {metrics['fatigue_rate']}%")
                    print(f"  UPDRS Score: {updrs_result.total_score} ({updrs_result.severity})")

                    # Step 5: AI Interpretation (if API key available)
                    print("\nStep 5: Generating AI interpretation...")
                    update_step(video_id, "ai_interpretation", "in_progress")
                    try:
                        interpreter = InterpretationAgent()
                        print(f"  InterpretationAgent initialized")
                        interpretation_result = interpreter.interpret_finger_tapping(
                            updrs_score=updrs_result.total_score,
                            severity=updrs_result.severity,
                            metrics=metrics,
                            details=updrs_result.details
                        )
                        ai_interpretation = {
                            "summary": interpretation_result.summary,
                            "explanation": interpretation_result.explanation,
                            "recommendations": interpretation_result.recommendations
                        }
                        update_step(video_id, "ai_interpretation", "completed")
                        print(f"  ✓ AI Interpretation generated successfully")
                        print(f"  Summary: {interpretation_result.summary[:50]}...")
                    except Exception as interp_error:
                        import traceback
                        print(f"  ⚠️ AI Interpretation ERROR: {str(interp_error)}")
                        print(f"  Error type: {type(interp_error).__name__}")
                        traceback.print_exc()
                        update_step(video_id, "ai_interpretation", "error")
                        ai_interpretation = None

                elif mode == "pose":
                    metrics_obj = calculator.calculate_gait_metrics(frames_dict)
                    metrics = {
                        "walking_speed": round(metrics_obj.walking_speed, 2),
                        "stride_length": round(metrics_obj.stride_length, 2),
                        "cadence": round(metrics_obj.cadence, 1),
                        "stride_variability": round(metrics_obj.stride_variability, 1),
                        "arm_swing_asymmetry": round(metrics_obj.arm_swing_asymmetry, 1),
                        "step_count": metrics_obj.step_count,
                        "duration": round(metrics_obj.duration, 2)
                    }

                    # Calculate UPDRS score
                    update_step(video_id, "updrs_calculation", "in_progress")
                    updrs_result = scorer.score_gait(metrics_obj)
                    updrs_score_data = {
                        "score": updrs_result.total_score,
                        "severity": updrs_result.severity,
                        "base_score": updrs_result.base_score,
                        "penalties": updrs_result.penalties,
                        "details": updrs_result.details
                    }
                    update_step(video_id, "updrs_calculation", "completed")

                    print(f"  Walking Speed: {metrics['walking_speed']} m/s")
                    print(f"  Cadence: {metrics['cadence']} steps/min")
                    print(f"  Stride Length: {metrics['stride_length']} m")
                    print(f"  UPDRS Score: {updrs_result.total_score} ({updrs_result.severity})")

                    # Step 5: AI Interpretation (if API key available)
                    print("\nStep 5: Generating AI interpretation...")
                    update_step(video_id, "ai_interpretation", "in_progress")
                    try:
                        interpreter = InterpretationAgent()
                        interpretation_result = interpreter.interpret_gait(
                            updrs_score=updrs_result.total_score,
                            severity=updrs_result.severity,
                            metrics=metrics,
                            details=updrs_result.details
                        )
                        ai_interpretation = {
                            "summary": interpretation_result.summary,
                            "explanation": interpretation_result.explanation,
                            "recommendations": interpretation_result.recommendations
                        }
                        update_step(video_id, "ai_interpretation", "completed")
                        print(f"  ✓ AI Interpretation generated")
                    except Exception as interp_error:
                        print(f"  ⚠️ AI Interpretation unavailable: {str(interp_error)}")
                        update_step(video_id, "ai_interpretation", "error")
                        ai_interpretation = None

                skeleton_data = {
                    "total_frames": len(landmark_frames),
                    "detection_rate": len(landmark_frames) / total_frames if total_frames > 0 else 0,
                    "mode": mode,
                    "skeleton_video_url": f"/files/{os.path.basename(skeleton_video_path)}"
                }
            else:
                print("  ⚠️ No landmarks detected - metrics unavailable")

        except Exception as e:
            print(f"  ⚠️ MediaPipe processing failed: {str(e)}")
            import traceback
            traceback.print_exc()
            # Continue without metrics

        print(f"\n{'='*50}")
        print(f"✅ Analysis Complete!")
        print(f"{'='*50}")
        print(f"Video Type (Auto-detected): {task_result.task_type}")
        print(f"Video Type (Final): {final_video_type}")
        print(f"Confidence: {task_result.confidence:.1%}")
        print(f"{'='*50}\n")

        # Prepare response
        response = {
            "success": True,
            "id": video_id,  # Add video_id for progress tracking
            "patient_id": patient_id,
            "video_type": final_video_type,  # AUTO-DETECTED!
            "auto_detected": manual_test_type is None,
            "confidence": task_result.confidence,
            "roi": {
                "x": task_result.roi[0],
                "y": task_result.roi[1],
                "w": task_result.roi[2],
                "h": task_result.roi[3]
            },
            "motion_analysis": {
                "motion_pattern": task_result.motion_pattern,
                "motion_area_ratio": task_result.motion_area_ratio,
                "body_part": roi_result.body_part
            },
            "reasoning": task_result.reasoning,
            "video_metadata": {
                "width": frame_width,
                "height": frame_height,
                "fps": fps,
                "duration": duration,
                "total_frames": total_frames
            },
            "metrics": metrics,  # Clinical kinematic metrics
            "skeleton_data": skeleton_data,  # Skeleton extraction info
            "updrs_score": updrs_score_data,  # UPDRS scoring (rule-based)
            "ai_interpretation": ai_interpretation,  # AI-generated patient-friendly explanation
            "events": events if 'events' in locals() else [],  # Timeline events
            "visualization_urls": {
                "heatmap": f"/files/{video_id}_heatmap.png" if os.path.exists(os.path.join(app_config['UPLOAD_FOLDER'], f"{video_id}_heatmap.png")) else None,
                "temporal_map": f"/files/{video_id}_temporal.png" if os.path.exists(os.path.join(app_config['UPLOAD_FOLDER'], f"{video_id}_temporal.png")) else None,
                "attention_map": f"/files/{video_id}_attention.png" if os.path.exists(os.path.join(app_config['UPLOAD_FOLDER'], f"{video_id}_attention.png")) else None
            }
        }

        # Save result to file
        result_path = os.path.join(app_config['UPLOAD_FOLDER'], f"{video_id}_result.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
        
        # Mark analysis as completed
        complete_analysis(video_id)

    except Exception as e:
        print(f"\n❌ Error during analysis:")
        print(f"  {str(e)}\n")
        import traceback
        traceback.print_exc()

        # Mark analysis as failed
        fail_analysis(video_id, str(e))

        # Clean up on error
        if os.path.exists(video_path):
            try:
                # os.remove(video_path) # Keep for debugging
                pass
            except:
                pass


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

        # Generate unique video_id for progress tracking
        import time
        filename = secure_filename(video_file.filename)
        video_id = f"{os.path.splitext(filename)[0]}_{int(time.time())}"
        
        # Save video file
        # Note: We need to save it here before starting the thread
        video_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{video_id}_{filename}")
        video_file.save(video_path)

        # Initialize progress tracking
        init_analysis(video_id, task_type="auto_detect")

        # Get optional parameters
        patient_id = request.form.get('patient_id', 'unknown')
        manual_test_type = request.form.get('test_type', None)
        
        # Start background thread
        thread = threading.Thread(
            target=process_video_background,
            args=(
                video_path, 
                video_id, 
                patient_id, 
                manual_test_type, 
                current_app.config.copy() # Pass config copy to thread
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
            "error": str(e)
        }), 500


@bp.route('/analysis/result/<video_id>', methods=['GET'])
def get_analysis_result(video_id):
    """
    Get the final result of an analysis
    """
    result_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{video_id}_result.json")
    
    if not os.path.exists(result_path):
        return jsonify({
            "success": False,
            "error": "Result not found or analysis not complete"
        }), 404
        
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error reading result: {str(e)}"
        }), 500


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

