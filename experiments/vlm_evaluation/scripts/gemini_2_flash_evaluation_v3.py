"""
Gemini 2.0 Flash Experimental VLM Evaluation - Version 3
Updated prompt without medical terminology (2025-11-25)
Uses custom test set from data_mapping folder (485 samples)
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
    """Generate task-specific prompt - updated without medical terms"""
    criteria = {
        "Finger tapping": "Assess the speed, amplitude, hesitations, halts, and decrement in amplitude of the finger tapping.",
        "Hand movement": "Assess the speed, amplitude, hesitations, halts, and decrement in amplitude of the hand opening and closing.",
        "Leg agility": "Assess the speed, amplitude, hesitations, halts, and decrement in amplitude of the leg agility (heel stomping).",
        "Gait": "Assess the stride amplitude, stride speed, height of foot lift, heel strike, turning, and arm swing."
    }

    task_criteria = criteria.get(task_name, "Assess the movement quality and motor performance.")

    prompt = f"""You are an expert in movement analysis.
Analyze the video of a person performing the '{task_name}' task.
{task_criteria}

Rate the severity of the motor performance on a 0-4 scale:
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

        # Clean up uploaded file
        try:
            genai.delete_file(video_file.name)
        except:
            pass

        return pred_score, reason, output_text

    except Exception as e:
        return -1, f"API Error: {str(e)}", ""

def main(args):
    # Initialize Gemini API
    api_key = args.api_key or DEFAULT_API_KEY

    if not api_key:
        print("Error: Google API key not provided. Set GOOGLE_API_KEY environment variable or use --api_key")
        return

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    print("Gemini 2.0 Flash Experimental initialized")

    # Setup paths
    video_dir = os.path.join(args.base_dir, "Videos")
    test_csv_dir = args.test_csv_dir

    print(f"Using video directory: {video_dir}")
    print(f"Using test CSV directory: {test_csv_dir}")

    # Task mapping to CSV filenames
    task_to_csv = {
        "Gait": "gait_test.csv.txt",
        "Finger tapping": "fingertapping_test.csv.txt",
        "Hand movement": "handmovement_test.csv.txt",
        "Leg agility": "legagility_test.csv.txt"
    }

    tasks = [args.task] if args.task else ["Gait", "Finger tapping", "Hand movement", "Leg agility"]

    results = []

    for task in tasks:
        csv_filename = task_to_csv.get(task)
        if not csv_filename:
            print(f"Unknown task: {task}")
            continue

        csv_path = os.path.join(test_csv_dir, csv_filename)
        if not os.path.exists(csv_path):
            print(f"Test CSV not found: {csv_path}")
            continue

        # Read test CSV (no header: filename,frame_count,gt_score)
        df = pd.read_csv(csv_path, header=None, names=['filename', 'frame_count', 'gt_score'])
        print(f"\nProcessing task: {task} ({len(df)} samples)")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=task):
            filename = row['filename']
            gt_score = float(row['gt_score'])

            vid_path = get_video_path(task, filename, video_dir)
            if not vid_path or not os.path.exists(vid_path):
                print(f"Video not found: {vid_path}")
                continue

            # Evaluate with Gemini
            pred_score, reason, raw_output = evaluate_video_gemini(
                model, vid_path, task
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
            if idx % 10 == 0 and idx > 0:
                temp_path = os.path.join("../results", "gemini2_flash_v3_results_temp.csv")
                pd.DataFrame(results).to_csv(temp_path, index=False)
                print(f"\nSaved intermediate results ({len(results)} samples)")

            # Rate limiting
            time.sleep(1)

    # Final save
    save_dir = "../results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "gemini2_flash_v3_results.csv")
    pd.DataFrame(results).to_csv(save_path, index=False)
    print(f"\nResults saved to {save_path}")

    # Compute metrics
    df_results = pd.DataFrame(results)
    df_valid = df_results[df_results['pred_score'] >= 0]

    if len(df_valid) > 0:
        accuracy = (df_valid['pred_score'] == df_valid['gt_score']).mean()
        mae = (df_valid['pred_score'] - df_valid['gt_score']).abs().mean()
        print(f"\nOverall Performance:")
        print(f"  Valid predictions: {len(df_valid)}/{len(df_results)}")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  MAE: {mae:.3f}")

        # Per-task metrics
        print(f"\nPer-Task Performance:")
        for task in df_valid['task'].unique():
            task_df = df_valid[df_valid['task'] == task]
            task_acc = (task_df['pred_score'] == task_df['gt_score']).mean()
            task_mae = (task_df['pred_score'] - task_df['gt_score']).abs().mean()
            print(f"  {task}: Acc={task_acc:.3f}, MAE={task_mae:.3f} (n={len(task_df)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini 2.0 Flash VLM Evaluation v3")
    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR, help="Base directory of PD4T dataset")
    parser.add_argument("--test_csv_dir", type=str, default=DEFAULT_TEST_CSV_DIR, help="Directory with test CSV files")
    parser.add_argument("--api_key", type=str, default=None, help="Google API key")
    parser.add_argument("--task", type=str, default=None, help="Specific task to evaluate")
    args = parser.parse_args()

    main(args)
