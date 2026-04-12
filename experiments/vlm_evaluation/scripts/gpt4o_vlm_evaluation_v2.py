"""
GPT-5.1 VLM Evaluation for User-Labeled PD4T Test Set
OpenAI GPT-5.1 with vision capabilities for Parkinson's Disease assessment
Uses custom test set from data_mapping folder (485 samples, 18 subjects)
"""

import os
import argparse
import pandas as pd
import json
import base64
import cv2
from tqdm import tqdm
from openai import OpenAI

# Configuration
DEFAULT_BASE_DIR = "C:/Users/YK/tulip/PD4T/PD4T/PD4T"
DEFAULT_TEST_CSV_DIR = "C:/Users/YK/tulip/VLM_commercial/data_mapping"
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY")

def get_prompt(task_name):
    """Generate task-specific prompt for GPT-5.1 (same as Qwen)"""
    criteria = {
        "Finger tapping": "Assess the speed, amplitude, hesitations, halts, and decrement in amplitude of the finger tapping.",
        "Hand movement": "Assess the speed, amplitude, hesitations, halts, and decrement in amplitude of the hand opening and closing.",
        "Leg agility": "Assess the speed, amplitude, hesitations, halts, and decrement in amplitude of the leg agility (heel stomping).",
        "Gait": "Assess the stride amplitude, stride speed, height of foot lift, heel strike, turning, and arm swing."
    }

    task_criteria = criteria.get(task_name, "Assess the movement quality and signs of Parkinson's disease.")

    prompt = f"""You are an expert neurologist specializing in Parkinson's Disease.
Analyze the video of a patient performing the '{task_name}' task.
{task_criteria}

Rate the severity of the motor impairment on the MDS-UPDRS scale:
0: Normal (No problems)
1: Slight (Slight slowness/small amplitude, no decrement)
2: Mild (Mild slowness/amplitude, some decrement or hesitations)
3: Moderate (Moderate slowness/amplitude, frequent hesitations/halts)
4: Severe (Severe impairment, barely performs the task)

Output ONLY a JSON object with the following format:
{{
  "score": <int, 0-4>,
  "reasoning": "<string, brief explanation>"
}}
"""
    return prompt

def get_video_path(task, filename, video_dir):
    """Helper to resolve video path with nested structure"""
    parts = filename.split('_')
    if task == "Gait":
        if len(parts) != 2:
            return None
        visit_id, patient_id = parts[0], parts[1]
        rel_path = os.path.join(task, patient_id, f"{visit_id}.mp4")
    else:
        if len(parts) != 3:
            return None
        visit_id, pos, patient_id = parts[0], parts[1], parts[2]
        rel_path = os.path.join(task, patient_id, f"{visit_id}_{pos}.mp4")
    return os.path.join(video_dir, rel_path)

def extract_frames(video_path, max_frames=16):
    """Extract frames from video for GPT-5.1 Vision API"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame sampling interval
    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = [int(i * total_frames / max_frames) for i in range(max_frames)]

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Resize to 512x512 to reduce token usage
            frame = cv2.resize(frame, (512, 512))
            frames.append(frame)

    cap.release()
    return frames

def encode_frame_to_base64(frame):
    """Encode OpenCV frame to base64 for GPT-5.1 API"""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def evaluate_video_gpt51(client, video_path, task_name, max_frames=16):
    """Evaluate single video using GPT-5.1 Vision API"""
    # Extract frames
    frames = extract_frames(video_path, max_frames=max_frames)

    if not frames:
        return -1, "Failed to extract frames", ""

    # Prepare messages with frames
    prompt = get_prompt(task_name)

    content = [{"type": "text", "text": prompt}]

    # Add frames as images
    for frame in frames:
        base64_image = encode_frame_to_base64(frame)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    messages = [{"role": "user", "content": content}]

    try:
        # Call GPT-5.1 Vision API (temperature 0.0 not supported, using default 1.0)
        response = client.chat.completions.create(
            model="gpt-5.1-chat-latest",
            messages=messages,
            max_completion_tokens=300,
        )

        output_text = response.choices[0].message.content

        # Parse JSON response
        try:
            start = output_text.find('{')
            end = output_text.rfind('}') + 1
            json_str = output_text[start:end]
            data = json.loads(json_str)
            pred_score = data.get('score', -1)
            reason = data.get('reasoning', '')
        except:
            pred_score = -1
            reason = output_text

        return pred_score, reason, output_text

    except Exception as e:
        print(f"API Error: {e}")
        return -1, f"API Error: {str(e)}", ""

def main(args):
    # Initialize OpenAI client
    api_key = args.api_key or DEFAULT_API_KEY

    if not api_key:
        print("Error: OpenAI API key not provided. Set OPENAI_API_KEY environment variable or use --api_key")
        return

    client = OpenAI(api_key=api_key)
    print("OpenAI client initialized")
    print(f"Using test CSV directory: {args.test_csv_dir}")
    print(f"Using video directory: {args.base_dir}/Videos")

    # Setup paths
    video_dir = os.path.join(args.base_dir, "Videos")

    # Task-specific configurations (based on Qwen reference)
    frame_config = {
        "Gait": {"max_frames": 16},
        "Finger tapping": {"max_frames": 12},
        "Hand movement": {"max_frames": 12},
        "Leg agility": {"max_frames": 12}
    }

    # Task name mapping (CSV filename -> Display name)
    task_mapping = {
        "gait_test.csv.txt": "Gait",
        "fingertapping_test.csv.txt": "Finger tapping",
        "handmovement_test.csv.txt": "Hand movement",
        "legagility_test.csv.txt": "Leg agility"
    }

    results = []
    total_samples = 0

    for csv_filename, task in task_mapping.items():
        csv_path = os.path.join(args.test_csv_dir, csv_filename)

        if not os.path.exists(csv_path):
            print(f"Warning: CSV file not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path, header=None, names=['filename', 'frames', 'gt_score'])
        print(f"\nProcessing {task}: {len(df)} samples")
        total_samples += len(df)

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=task):
            filename = row['filename']
            gt_score = int(row['gt_score'])

            vid_path = get_video_path(task, filename, video_dir)
            if not vid_path or not os.path.exists(vid_path):
                print(f"\nVideo not found: {vid_path}")
                results.append({
                    'task': task,
                    'filename': filename,
                    'gt_score': gt_score,
                    'pred_score': -1,
                    'reason': 'Video file not found',
                    'raw_output': ''
                })
                continue

            # Get task-specific config
            config = frame_config.get(task, {"max_frames": 16})

            # Evaluate with GPT-5.1
            pred_score, reason, raw_output = evaluate_video_gpt51(
                client, vid_path, task, max_frames=config["max_frames"]
            )

            results.append({
                'task': task,
                'filename': filename,
                'gt_score': gt_score,
                'pred_score': pred_score,
                'reason': reason,
                'raw_output': raw_output
            })

            # Save intermediate results every 10 samples
            if (len(results)) % 10 == 0:
                save_dir = "C:/Users/YK/tulip/VLM_commercial/results"
                os.makedirs(save_dir, exist_ok=True)
                temp_path = os.path.join(save_dir, "gpt51_results_temp.csv")
                pd.DataFrame(results).to_csv(temp_path, index=False)

    # Final save
    save_dir = "C:/Users/YK/tulip/VLM_commercial/results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "gpt51_results.csv")
    pd.DataFrame(results).to_csv(save_path, index=False)
    print(f"\nResults saved to {save_path}")

    # Compute metrics
    df_results = pd.DataFrame(results)
    df_valid = df_results[df_results['pred_score'] >= 0]

    if len(df_valid) > 0:
        accuracy = (df_valid['pred_score'] == df_valid['gt_score']).mean()
        mae = (df_valid['pred_score'] - df_valid['gt_score']).abs().mean()

        print(f"\n{'='*80}")
        print("Evaluation Results")
        print(f"{'='*80}")
        print(f"Total samples processed: {len(results)}")
        print(f"Valid predictions: {len(df_valid)}")
        print(f"Failed predictions: {len(results) - len(df_valid)}")
        print(f"\nAccuracy: {accuracy:.3f}")
        print(f"MAE: {mae:.3f}")

        # Per-task breakdown
        print(f"\nPer-task breakdown:")
        for task in df_valid['task'].unique():
            task_df = df_valid[df_valid['task'] == task]
            task_acc = (task_df['pred_score'] == task_df['gt_score']).mean()
            task_mae = (task_df['pred_score'] - task_df['gt_score']).abs().mean()
            print(f"  {task}: Acc={task_acc:.3f}, MAE={task_mae:.3f}, N={len(task_df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-5.1 VLM Evaluation for User-Labeled PD4T Test Set")
    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR, help="Base directory of PD4T dataset (where Videos folder is)")
    parser.add_argument("--test_csv_dir", type=str, default=DEFAULT_TEST_CSV_DIR, help="Directory containing test CSV files")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    args = parser.parse_args()

    main(args)
