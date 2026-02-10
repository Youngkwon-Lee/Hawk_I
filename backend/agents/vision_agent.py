import os
import cv2
import time
from agents.base_agent import BaseAgent
from domain.context import AnalysisContext
from services.roi_detector import MovementBasedROI
from services.mediapipe_processor import MediaPipeProcessor
from services.task_classifier import TaskClassifier
from services.visualization import VisualizationService

class VisionAgent(BaseAgent):
    def __init__(self):
        self.roi_detector = MovementBasedROI()
        self.mp_processor_hand = MediaPipeProcessor(mode="hand")
        self.mp_processor_pose = MediaPipeProcessor(mode="pose")
        self.task_classifier = TaskClassifier()
        self.visualization_service = VisualizationService()

    def _log(self, msg: str, elapsed: float = None):
        """Print timestamped log"""
        elapsed_str = f" ({elapsed:.2f}s)" if elapsed else ""
        print(f"  [Vision] {msg}{elapsed_str}")

    def process(self, ctx: AnalysisContext) -> AnalysisContext:
        try:
            ctx.log("vision", "Vision Agent started analysis.")

            video_path = ctx.video_path
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # 1. Get Video Metadata
            t0 = time.time()
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            ctx.vision_meta = {
                "fps": fps,
                "frame_count": frame_count,
                "quality_flag": "ok" # Placeholder for quality check
            }
            self._log(f"Video loaded: {frame_count} frames at {fps:.2f} fps", time.time() - t0)

            # 2. ROI Detection
            t0 = time.time()
            self._log("Starting ROI detection...")
            roi_result = self.roi_detector.detect(video_path)
            self._log("ROI detection complete", time.time() - t0)
            
            # 3. Task Classification
            # We pass the ROI result to the classifier
            # We need frame size for classification
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            task_result_obj = self.task_classifier.classify(roi_result, (frame_width, frame_height))
            
            ctx.task_type = task_result_obj.task_type
            
            # --- Heatmap Generation ---
            try:
                # roi_result.movement_map is already available (H x W float32 0..1)
                movement_map = roi_result.movement_map
                
                # Generate path
                video_dir = os.path.dirname(video_path)
                video_basename = os.path.splitext(os.path.basename(video_path))[0]
                heatmap_path = os.path.join(video_dir, f"{video_basename}_heatmap.jpg")
                
                # Use VisualizationService
                self.visualization_service.generate_motion_heatmap(
                    movement_map=movement_map,
                    output_path=heatmap_path,
                    video_path=video_path # Pass video path for overlay
                )
                
                ctx.log("vision", f"Heatmap generated: {heatmap_path}")
            except Exception as e:
                ctx.log("vision", f"Failed to generate heatmap: {e}")
                heatmap_path = None


            # Store comprehensive metadata
            ctx.vision_meta.update({
                "roi": task_result_obj.roi,
                "body_part": roi_result.body_part,
                "motion_pattern": task_result_obj.motion_pattern,
                "motion_area_ratio": task_result_obj.motion_area_ratio,
                "confidence": task_result_obj.confidence,
                "reasoning": task_result_obj.reasoning,
                "heatmap_path": heatmap_path
            })

            ctx.log("vision", f"Task classified as: {ctx.task_type}", meta=dict(ctx.vision_meta))

            # 3. Skeleton Extraction (MediaPipe)
            t0 = time.time()
            self._log("Starting skeleton extraction (MediaPipe)...")

            # Select processor based on task
            if ctx.task_type == "gait" or ctx.task_type == "leg_agility":
                processor = self.mp_processor_pose
                self._log("Using POSE mode for gait/leg_agility")
            else:
                processor = self.mp_processor_hand
                self._log("Using HAND mode for finger tapping")

            # Generate skeleton overlay video path
            video_dir = os.path.dirname(video_path)
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            skeleton_video_path = os.path.join(video_dir, f"{video_basename}_skeleton.mp4")

            # Get video FPS for frame_skip optimization (cv2 already imported at top)
            cap = cv2.VideoCapture(video_path)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            # Task-specific optimization settings - Balance speed and accuracy
            resize_width = 480  # Higher resolution for better skeleton accuracy
            use_square = False  # Keep original aspect ratio
            skip_video = False  # Generate skeleton video on backend (reliable)

            # Frame skip for high FPS videos (60fps -> ~20fps processing)
            frame_skip = 3 if video_fps > 35 else 1

            self._log(f"Processing: {resize_width}px, fps={video_fps:.0f}, skip={frame_skip}, no_video={skip_video}")

            # Process full video - skip_video=True makes this fast enough
            # (no video encoding, only keypoint extraction)
            landmarks = processor.process_video(
                video_path,
                output_video_path=skeleton_video_path if not skip_video else None,
                max_duration=None,  # Process full video for complete canvas overlay
                resize_width=resize_width,
                use_mediapipe_optimal=use_square,
                frame_skip=frame_skip,
                skip_video_generation=skip_video
            )
            self._log(f"Skeleton extraction complete: {len(landmarks)} frames", time.time() - t0)

            # Store skeleton video path in context
            ctx.vision_meta["skeleton_video_path"] = skeleton_video_path

            # Convert LandmarkFrame objects to dicts for context storage
            t0 = time.time()
            from dataclasses import asdict
            landmarks_dicts = [asdict(lf) for lf in landmarks]

            ctx.skeleton_data = {
                "landmarks": landmarks_dicts, # This is List[Dict]
            }
            ctx.log("vision", f"Skeleton extraction complete. {len(landmarks)} frames processed.")

            # 4. Trajectory Map Generation
            try:
                t0 = time.time()
                self._log("Generating trajectory map...")

                # Prepare data for visualization (needs 'keypoints' key)
                viz_data = []
                for lf in landmarks:
                    viz_data.append({
                        "keypoints": lf.landmarks,
                        "timestamp": lf.timestamp
                    })

                # Determine mode for visualization
                viz_mode = 'pose' if ctx.task_type in ["gait", "leg_agility"] else 'hand'

                # Generate path
                trajectory_path = os.path.join(video_dir, f"{video_basename}_trajectory.jpg")

                # Generate map
                self.visualization_service.generate_temporal_map(
                    frames_data=viz_data,
                    output_path=trajectory_path,
                    mode=viz_mode,
                    resolution=(frame_width, frame_height),
                    video_path=video_path
                )

                ctx.vision_meta["trajectory_map_path"] = trajectory_path
                self._log("Trajectory map generated", time.time() - t0)

            except Exception as e:
                ctx.log("vision", f"Failed to generate trajectory map: {e}")

            
            ctx.status = "vision_done"
            return ctx

        except Exception as e:
            ctx.error = str(e)
            ctx.status = "error"
            ctx.log("vision", f"Error in Vision Agent: {e}")
            raise e
