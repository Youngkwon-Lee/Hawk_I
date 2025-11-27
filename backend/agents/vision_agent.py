import os
import cv2
from agents.base_agent import BaseAgent
from domain.context import AnalysisContext
from services.roi_detector import MovementBasedROI
from services.mediapipe_processor import MediaPipeProcessor
from services.task_classifier import TaskClassifier

class VisionAgent(BaseAgent):
    def __init__(self):
        self.roi_detector = MovementBasedROI()
        self.mp_processor_hand = MediaPipeProcessor(mode="hand")
        self.mp_processor_pose = MediaPipeProcessor(mode="pose")
        self.task_classifier = TaskClassifier()

    def process(self, ctx: AnalysisContext) -> AnalysisContext:
        try:
            ctx.log("vision", "Vision Agent started analysis.")
            
            video_path = ctx.video_path
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # 1. Get Video Metadata
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            ctx.vision_meta = {
                "fps": fps,
                "frame_count": frame_count,
                "quality_flag": "ok" # Placeholder for quality check
            }
            ctx.log("vision", f"Video loaded: {frame_count} frames at {fps:.2f} fps.")

            # 2. ROI Detection
            # We call detect directly to get the ROIResult which contains body_part
            roi_result = self.roi_detector.detect(video_path)
            
            # 3. Task Classification
            # We pass the ROI result to the classifier
            # We need frame size for classification
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            task_result_obj = self.task_classifier.classify(roi_result, (frame_width, frame_height))
            
            ctx.task_type = task_result_obj.task_type
            
            # Store comprehensive metadata
            ctx.vision_meta.update({
                "roi": task_result_obj.roi,
                "body_part": roi_result.body_part,
                "motion_pattern": task_result_obj.motion_pattern,
                "motion_area_ratio": task_result_obj.motion_area_ratio,
                "confidence": task_result_obj.confidence,
                "reasoning": task_result_obj.reasoning
            })

            ctx.log("vision", f"Task classified as: {ctx.task_type}", meta=dict(ctx.vision_meta))

            # 3. Skeleton Extraction (MediaPipe)
            # Select processor based on task
            if ctx.task_type == "gait" or ctx.task_type == "leg_agility":
                processor = self.mp_processor_pose
            else:
                processor = self.mp_processor_hand

            # Generate skeleton overlay video path
            video_dir = os.path.dirname(video_path)
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            skeleton_video_path = os.path.join(video_dir, f"{video_basename}_skeleton.mp4")

            landmarks = processor.process_video(video_path, output_video_path=skeleton_video_path)

            # Store skeleton video path in context
            ctx.vision_meta["skeleton_video_path"] = skeleton_video_path
            
            # Convert LandmarkFrame objects to dicts for context storage
            # The context expects serializable data usually, but objects are fine for in-memory
            # Let's keep objects or convert if needed. 
            # Existing services might expect objects.
            # But context.skeleton_data is Dict. Let's store the list of frames there.
            
            # Convert LandmarkFrame objects to dicts for context storage
            from dataclasses import asdict
            landmarks_dicts = [asdict(lf) for lf in landmarks]
            
            ctx.skeleton_data = {
                "landmarks": landmarks_dicts, # This is List[Dict]
            }
            ctx.log("vision", f"Skeleton extraction complete. {len(landmarks)} frames processed.")
            
            ctx.status = "vision_done"
            return ctx

        except Exception as e:
            ctx.error = str(e)
            ctx.status = "error"
            ctx.log("vision", f"Error in Vision Agent: {e}")
            raise e
