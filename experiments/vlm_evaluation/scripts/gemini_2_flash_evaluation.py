"""
Gemini 2.0 Flash VLM Evaluation for PD4T Dataset
Google Gemini 2.0 Flash with multimodal capabilities for Parkinson's Disease assessment
"""

import os
import argparse
import pandas as pd
import json
import time
from tqdm import tqdm
from pathlib import Path
import google.generativeai as genai

# Configuration
DEFAULT_BASE_DIR = "PD4T/PD4T/PD4T"
DEFAULT_API_KEY = os.getenv("GOOGLE_API_KEY")  # Set via environment variable

def get_prompt(task_name):
    """Generate task-specific prompt for Gemini"""
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
    """
    Evaluate single video using Gemini 2.0 Flash
    """
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
                "temperature": 0.0,  # Deterministic for medical assessment
                "max_output_tokens": 300,
            }
        )

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

    # Initialize Gemini 2.0 Flash model
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    print("Gemini 2.0 Flash model initialized")

    # Setup paths
    annotations_dir = os.path.join(args.base_dir, "Annotations")
    video_dir = os.path.join(args.base_dir, "Videos")

    tasks = [args.task] if args.task else ["Finger tapping", "Gait", "Hand movement", "Leg agility"]

    results = []

    for task in tasks:
        csv_path = os.path.join(annotations_dir, task, "test.csv")
        if not os.path.exists(csv_path):
            print(f"Annotation file not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path, header=None, names=['filename', 'frames', 'gt_score'])
        print(f"\nProcessing task: {task} ({len(df)} samples)")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=task):
            filename = row['filename']
            gt_score = float(row['gt_score'])

            vid_path = get_video_path(task, filename, video_dir)
            if not vid_path or not os.path.exists(vid_path):
                print(f"Video not found: {vid_path}")
                continue

            # Evaluate with Gemini 2.0 Flash
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
            if idx % 10 == 0 and idx > 0:
                temp_path = os.path.join(args.base_dir, "gemini_results_temp.csv")
                pd.DataFrame(results).to_csv(temp_path, index=False)
                print(f"Saved intermediate results ({len(results)} samples)")

            # Rate limiting: Gemini API has limits
            time.sleep(1)

    # Final save
    save_dir = os.path.join(args.base_dir, "Results_VLM")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "gemini_2_flash_results.csv")
    pd.DataFrame(results).to_csv(save_path, index=False)
    print(f"\nResults saved to {save_path}")

    # Compute metrics
    df_results = pd.DataFrame(results)
    df_valid = df_results[df_results['pred_score'] >= 0]

    if len(df_valid) > 0:
        accuracy = (df_valid['pred_score'] == df_valid['gt_score']).mean()
        mae = (df_valid['pred_score'] - df_valid['gt_score']).abs().mean()
        print(f"\nAccuracy: {accuracy:.3f}")
        print(f"MAE: {mae:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini 2.0 Flash VLM Evaluation for PD4T")
    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR, help="Base directory of dataset")
    parser.add_argument("--api_key", type=str, default=None, help="Google API key")
    parser.add_argument("--task", type=str, default=None, help="Specific task to evaluate")
    args = parser.parse_args()

    main(args)
