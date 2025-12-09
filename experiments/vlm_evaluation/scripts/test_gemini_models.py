"""
Test Gemini Models (2.0 Flash, 2.5 Flash, 3.0 Pro) on Small Sample
2025-11-25
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
import google.generativeai as genai

# Configuration
GOOGLE_API_KEY = "AIzaSyBtEoiwynmJK84ARDZJUnSCUY4U9izij9w"
BASE_DIR = "C:/Users/YK/tulip/PD4T/PD4T/PD4T"

# Models to test
MODELS_TO_TEST = [
    "gemini-2.0-flash-exp",
    "gemini-2.5-flash",
    "gemini-3-pro-preview"
]

def get_prompt(task_name):
    """Generate task-specific prompt"""
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
    """Helper to resolve video path"""
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

def evaluate_video(model, video_path, task_name, model_name):
    """Evaluate single video with specified Gemini model"""
    prompt = get_prompt(task_name)

    try:
        # Upload video file
        print(f"  Uploading video... ", end='', flush=True)
        video_file = genai.upload_file(path=video_path)

        # Wait for processing
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            print("FAILED (processing)")
            return -1, "Video processing failed", "", 0

        print("OK", flush=True)

        # Generate content
        start_time = time.time()
        response = model.generate_content(
            [video_file, prompt],
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 300,
            }
        )
        inference_time = time.time() - start_time

        output_text = response.text

        # Parse JSON
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

        return pred_score, reason, output_text, inference_time

    except Exception as e:
        print(f"ERROR: {e}")
        return -1, f"API Error: {str(e)}", "", 0

def main():
    # Initialize API
    genai.configure(api_key=GOOGLE_API_KEY)

    print("=" * 80)
    print("Gemini Models Comparison Test (2025-11-25)")
    print(f"Models: {', '.join(MODELS_TO_TEST)}")
    print("=" * 80)

    # Get test samples (2 from each task)
    annotations_dir = os.path.join(BASE_DIR, "Annotations")
    video_dir = os.path.join(BASE_DIR, "Videos")

    test_samples = []
    for task in ["Gait", "Finger tapping"]:
        csv_path = os.path.join(annotations_dir, task, "test.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, header=None, names=['filename', 'frames', 'gt_score'])
            # Take first 2 samples
            for idx, row in df.head(2).iterrows():
                filename = row['filename']
                gt_score = float(row['gt_score'])
                vid_path = get_video_path(task, filename, video_dir)
                if vid_path and os.path.exists(vid_path):
                    test_samples.append({
                        'task': task,
                        'filename': filename,
                        'gt_score': gt_score,
                        'video_path': vid_path
                    })

    print(f"\nTest samples: {len(test_samples)}")
    for s in test_samples:
        print(f"  - {s['task']}: {s['filename']} (GT={s['gt_score']})")

    # Test each model
    all_results = []

    for model_name in MODELS_TO_TEST:
        print(f"\n{'=' * 80}")
        print(f"Testing Model: {model_name}")
        print(f"{'=' * 80}")

        try:
            model = genai.GenerativeModel(model_name)
            print(f"Model initialized: {model_name}")
        except Exception as e:
            print(f"ERROR: Failed to initialize model - {e}")
            continue

        for i, sample in enumerate(test_samples, 1):
            print(f"\n[{i}/{len(test_samples)}] {sample['task']}: {sample['filename']}")

            pred_score, reason, raw_output, inference_time = evaluate_video(
                model, sample['video_path'], sample['task'], model_name
            )

            result = {
                'model': model_name,
                'task': sample['task'],
                'filename': sample['filename'],
                'gt_score': sample['gt_score'],
                'pred_score': pred_score,
                'reason': reason,
                'inference_time': inference_time,
                'raw_output': raw_output
            }
            all_results.append(result)

            print(f"  GT: {sample['gt_score']}, Pred: {pred_score}, Time: {inference_time:.2f}s")
            if pred_score >= 0:
                print(f"  Reason: {reason[:100]}...")

            # Rate limiting
            time.sleep(2)

    # Save results
    results_df = pd.DataFrame(all_results)
    output_path = "../results/gemini_models_comparison_test.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_path}")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    for model_name in MODELS_TO_TEST:
        model_results = results_df[results_df['model'] == model_name]
        valid = model_results[model_results['pred_score'] >= 0]

        if len(valid) > 0:
            accuracy = (valid['pred_score'] == valid['gt_score']).mean()
            mae = (valid['pred_score'] - valid['gt_score']).abs().mean()
            avg_time = valid['inference_time'].mean()

            print(f"\n{model_name}:")
            print(f"  Valid predictions: {len(valid)}/{len(model_results)}")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  MAE: {mae:.3f}")
            print(f"  Avg inference time: {avg_time:.2f}s")
        else:
            print(f"\n{model_name}: All predictions failed")

if __name__ == "__main__":
    main()
