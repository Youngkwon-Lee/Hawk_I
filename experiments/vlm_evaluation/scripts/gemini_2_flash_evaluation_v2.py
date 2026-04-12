"""
Gemini 2.0 Flash Experimental VLM Evaluation for User-Labeled PD4T Test Set
Google Gemini 2.0 Flash Experimental with multimodal capabilities for Parkinson's Disease assessment
Uses custom test set from data_mapping folder (485 samples, 18 subjects)
"""

import os
import argparse
import pandas as pd
import json
import time
from tqdm import tqdm
import google.generativeai as genai

# Configuration
DEFAULT_BASE_DIR = "C:/Users/YK/tulip/PD4T/PD4T/PD4T"
DEFAULT_TEST_CSV_DIR = "C:/Users/YK/tulip/VLM_commercial/data_mapping"
DEFAULT_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_prompt(task_name):
    """Generate task-specific prompt for Gemini (same as Qwen)"""
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

def evaluate_video_gemini(model, video_path, task_name):
    """Evaluate single video using Gemini 2.0 Flash Experimental"""
    prompt = get_prompt(task_name)

    try:
        # Upload video file
        video_file = genai.upload_file(path=video_path)

        # Wait for file to be processed
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            return -1, "Video processing failed", ""

        # Generate content with video
        response = model.generate_content(
            [video_file, prompt],
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 300,
            }
        )

        # Check if response has valid content
        if not response.candidates or not response.candidates[0].content.parts:
            return -1, f"Empty response: {response.prompt_feedback}", ""

        output_text = response.text

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

        # Delete uploaded file to free storage
        genai.delete_file(video_file.name)

        return pred_score, reason, output_text

    except Exception as e:
        print(f"API Error: {e}")
        return -1, f"API Error: {str(e)}", ""

def main(args):
    # Initialize Gemini client
    api_key = args.api_key or DEFAULT_API_KEY

    if not api_key:
        print("Error: Google API key not provided. Set GOOGLE_API_KEY environment variable or use --api_key")
        return

    genai.configure(api_key=api_key)

    # Initialize Gemini 2.0 Flash Experimental (3.0 Pro not available yet)
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    print("Gemini 2.0 Flash Experimental model initialized")
    print(f"Using test CSV directory: {args.test_csv_dir}")
    print(f"Using video directory: {args.base_dir}/Videos")

    # Setup paths
    video_dir = os.path.join(args.base_dir, "Videos")

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

            # Evaluate with Gemini 2.0 Flash Experimental
            pred_score, reason, raw_output = evaluate_video_gemini(model, vid_path, task)

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
                temp_path = os.path.join(save_dir, "gemini2_flash_results_temp.csv")
                pd.DataFrame(results).to_csv(temp_path, index=False)

            # Rate limiting: Gemini API has limits
            time.sleep(1)

    # Final save
    save_dir = "C:/Users/YK/tulip/VLM_commercial/results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "gemini2_flash_results.csv")
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
    parser = argparse.ArgumentParser(description="Gemini 2.0 Flash Experimental VLM Evaluation for User-Labeled PD4T Test Set")
    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR, help="Base directory of PD4T dataset (where Videos folder is)")
    parser.add_argument("--test_csv_dir", type=str, default=DEFAULT_TEST_CSV_DIR, help="Directory containing test CSV files")
    parser.add_argument("--api_key", type=str, default=None, help="Google API key")
    args = parser.parse_args()

    main(args)
