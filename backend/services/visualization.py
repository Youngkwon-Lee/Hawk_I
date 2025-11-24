import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import io

class VisualizationService:
    def __init__(self):
        pass

    def generate_heatmap(self, frames_data, output_path, resolution=(856, 480), video_path=None):
        """
        Generate a heatmap of movement density from skeleton frames.
        If video_path is provided, overlays the heatmap on a representative frame.
        """
        try:
            # Initialize empty density map
            density_map = np.zeros((resolution[1], resolution[0]), dtype=np.float32)
            
            # Accumulate keypoint positions
            for frame in frames_data:
                if not frame.get('keypoints'):
                    continue
                    
                for kp in frame['keypoints']:
                    # Keypoints are normalized [0, 1], scale to resolution
                    x = int(kp['x'] * resolution[0])
                    y = int(kp['y'] * resolution[1])
                    
                    # Add Gaussian blob at position
                    if 0 <= x < resolution[0] and 0 <= y < resolution[1]:
                        # Accumulate density
                        cv2.circle(density_map, (x, y), 15, 1, -1)

            # Normalize density map
            if np.max(density_map) > 0:
                density_map = density_map / np.max(density_map)
            
            # Apply Gaussian blur for smoother appearance
            density_map = cv2.GaussianBlur(density_map, (31, 31), 0)
            
            # Create colormap (Jet)
            heatmap_color = cv2.applyColorMap((density_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            if video_path:
                # Extract representative frame
                frame = self._extract_representative_frame(video_path)
                if frame is not None:
                    # Resize frame to match resolution
                    frame = cv2.resize(frame, resolution)
                    
                    # Create mask where density is low to make it transparent
                    # We want high density areas to show the heatmap color, low density to show original frame
                    # Or we can just blend them.
                    
                    # Better approach:
                    # 1. Create a mask from density map
                    mask = density_map
                    mask = np.stack([mask]*3, axis=2) # Make 3 channel
                    
                    # 2. Blend: Result = Frame * (1-Mask) + Heatmap * Mask
                    # Adjust mask intensity for better visibility
                    mask = np.clip(mask * 1.5, 0, 1) # Boost intensity
                    
                    # Convert to float for blending
                    frame_float = frame.astype(np.float32) / 255.0
                    heatmap_float = heatmap_color.astype(np.float32) / 255.0
                    
                    # Blend
                    blended = frame_float * (1 - mask * 0.6) + heatmap_float * (mask * 0.6)
                    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
                    
                    cv2.imwrite(output_path, blended)
                    return True

            # If no video path, save just the heatmap on white background or black
            # For consistency with previous behavior if no video, maybe just save heatmap
            cv2.imwrite(output_path, heatmap_color)
            
            return True
        except Exception as e:
            print(f"Error generating heatmap: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_temporal_map(self, frames_data, output_path, mode='hand', resolution=(856, 480), video_path=None):
        """
        Generate a temporal trajectory map of key joints.
        """
        try:
            # Create a blank white image
            img = np.ones((resolution[1], resolution[0], 3), dtype=np.uint8) * 255
            
            # Define key joints to track based on mode
            # Hand: Wrist(0), Index Tip(8), Thumb Tip(4)
            # Pose: Left Ankle(27), Right Ankle(28)
            indices_to_track = [4, 8] if mode == 'hand' else [27, 28]
            #Medical-appropriate colors: Teal and Purple (BGR)
            colors = [(180, 119, 31), (128, 0, 128)] # Teal #1F77B4, Purple #800080
            
            points_history = {idx: [] for idx in indices_to_track}
            
            for frame in frames_data:
                if not frame.get('keypoints'):
                    continue
                
                keypoints = frame['keypoints']
                for i, idx in enumerate(indices_to_track):
                    if idx < len(keypoints):
                        kp = keypoints[idx]
                        x = int(kp['x'] * resolution[0])
                        y = int(kp['y'] * resolution[1])
                        points_history[idx].append((x, y))

            # Draw trajectories
            for i, idx in enumerate(indices_to_track):
                points = points_history[idx]
                if len(points) < 2:
                    continue
                    
                # Draw lines connecting points
                # Fade color over time? For now, solid color
                for j in range(1, len(points)):
                    pt1 = points[j-1]
                    pt2 = points[j]
                    
                    # Gradient alpha could be cool, but simple line for now
                    cv2.line(img, pt1, pt2, colors[i % len(colors)], 2, cv2.LINE_AA)
                    
            # Save
            cv2.imwrite(output_path, img)
            
            # If video_path provided, overlay on representative frame
            if video_path:
                self._create_overlay(output_path, video_path, resolution)
            
            return True
            
        except Exception as e:
            print(f"Error generating temporal map: {e}")
            return False

    def generate_attention_map(self, frames_data, output_path, resolution=(856, 480)):
        """
        Generate a simulated attention map (placeholder for now as we don't have attention weights from model yet).
        For now, highlight the center of motion.
        """
        try:
              # Create movement density map
              movement_map = np.zeros((resolution[1], resolution[0]), dtype=np.float32)

              # Calculate movement for each keypoint across frames
              for i in range(1, len(frames_data)):
                  prev_frame = frames_data[i-1]
                  curr_frame = frames_data[i]

                  if not prev_frame.get('keypoints') or not curr_frame.get('keypoints'):
                      continue

                  prev_kps = prev_frame['keypoints']
                  curr_kps = curr_frame['keypoints']

                  # Calculate movement for each keypoint
                  for j in range(min(len(prev_kps), len(curr_kps))):
                      prev_kp = prev_kps[j]
                      curr_kp = curr_kps[j]

                      # Calculate movement distance
                      dx = (curr_kp['x'] - prev_kp['x']) * resolution[0]
                      dy = (curr_kp['y'] - prev_kp['y']) * resolution[1]
                      movement = np.sqrt(dx*dx + dy*dy)

                      # Add movement to map at current position
                      x = int(curr_kp['x'] * resolution[0])
                      y = int(curr_kp['y'] * resolution[1])

                      if 0 <= x < resolution[0] and 0 <= y < resolution[1]:
                          # Add Gaussian blob weighted by movement
                          cv2.circle(movement_map, (x, y), 30, float(movement), -1)

              # Normalize movement map
              if np.max(movement_map) > 0:
                  movement_map = movement_map / np.max(movement_map)

              # Apply Gaussian blur for smoother appearance
              movement_map = cv2.GaussianBlur(movement_map, (51, 51), 0)

              # Create colored heatmap (medical gradient: blue -> green -> yellow -> red)
              heatmap_color = cv2.applyColorMap((movement_map * 255).astype(np.uint8), cv2.COLORMAP_JET)

              # Save
              cv2.imwrite(output_path, heatmap_color)

              # Overlay on video frame if provided
              if video_path:
                  self._create_overlay(output_path, video_path, resolution)

              return True
        except Exception as e:
              print(f"Error generating attention map: {e}")
              import traceback
              traceback.print_exc()
              return False
    
    def _extract_representative_frame(self, video_path, frame_number=None):
        """
        Extract a representative frame from video.
        If frame_number not specified, use middle frame.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_number is None:
                frame_number = total_frames // 2
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                return frame
            return None
        except Exception as e:
            print(f"Error extracting frame: {e}")
            return None
    
    def _create_overlay(self, viz_path, video_path, resolution):
        """
        Blend visualization with original video frame.
        Overwrites the visualization file with overlaid version.
        """
        try:
            # Extract representative frame
            frame = self._extract_representative_frame(video_path)
            if frame is None:
                return
            
            # Resize frame to match resolution
            frame = cv2.resize(frame, resolution)
            
            # Read visualization
            viz = cv2.imread(viz_path)
            if viz is None:
                return
            
            # Ensure same size
            viz = cv2.resize(viz, resolution)
            
            # Alpha blend: 60% visualization, 40% original frame
            alpha = 0.6
            blended = cv2.addWeighted(viz, alpha, frame, 1 - alpha, 0)
            
            # Save overlaid version
            cv2.imwrite(viz_path, blended)
            
        except Exception as e:
            print(f"Error creating overlay: {e}")
