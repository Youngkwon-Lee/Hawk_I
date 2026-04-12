"""
Compare VLM Results: GPT-4o vs Gemini 2.0 Flash
Comprehensive comparison and analysis of VLM predictions on PD4T dataset
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    cohen_kappa_score,
    confusion_matrix,
    classification_report
)
import json

def load_results(gpt4o_path, gemini_path):
    """Load results from both models"""
    df_gpt4o = pd.read_csv(gpt4o_path)
    df_gemini = pd.read_csv(gemini_path)

    # Filter valid predictions
    df_gpt4o_valid = df_gpt4o[df_gpt4o['pred_score'] >= 0].copy()
    df_gemini_valid = df_gemini[df_gemini['pred_score'] >= 0].copy()

    return df_gpt4o_valid, df_gemini_valid

def compute_metrics(df, model_name):
    """Compute evaluation metrics for a model"""
    gt = df['gt_score'].values
    pred = df['pred_score'].values

    accuracy = accuracy_score(gt, pred)
    mae = mean_absolute_error(gt, pred)
    kappa = cohen_kappa_score(gt, pred, weights='quadratic')

    # Confusion matrix
    cm = confusion_matrix(gt, pred, labels=[0, 1, 2, 3, 4])

    # Classification report
    report = classification_report(gt, pred, labels=[0, 1, 2, 3, 4],
                                   target_names=['0', '1', '2', '3', '4'],
                                   output_dict=True, zero_division=0)

    return {
        'model': model_name,
        'accuracy': accuracy,
        'mae': mae,
        'weighted_kappa': kappa,
        'confusion_matrix': cm,
        'classification_report': report,
        'num_samples': len(df)
    }

def compute_task_metrics(df, model_name):
    """Compute per-task metrics"""
    tasks = df['task'].unique()
    task_metrics = {}

    for task in tasks:
        df_task = df[df['task'] == task]
        gt = df_task['gt_score'].values
        pred = df_task['pred_score'].values

        if len(gt) == 0:
            continue

        accuracy = accuracy_score(gt, pred)
        mae = mean_absolute_error(gt, pred)
        kappa = cohen_kappa_score(gt, pred, weights='quadratic')

        task_metrics[task] = {
            'accuracy': accuracy,
            'mae': mae,
            'weighted_kappa': kappa,
            'num_samples': len(df_task)
        }

    return task_metrics

def generate_markdown_report(metrics_gpt4o, metrics_gemini,
                             task_metrics_gpt4o, task_metrics_gemini,
                             output_path):
    """Generate comprehensive markdown report"""

    report = f"""# VLM Comparison Report: GPT-4o vs Gemini 2.0 Flash

**Dataset**: PD4T (Parkinson's Disease 4 Tasks)
**Report Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 Overall Performance

| Metric | GPT-4o | Gemini 2.0 Flash | Winner |
|--------|--------|------------------|--------|
| **Accuracy** | {metrics_gpt4o['accuracy']:.3f} | {metrics_gemini['accuracy']:.3f} | {'GPT-4o' if metrics_gpt4o['accuracy'] > metrics_gemini['accuracy'] else 'Gemini' if metrics_gemini['accuracy'] > metrics_gpt4o['accuracy'] else 'Tie'} |
| **MAE** | {metrics_gpt4o['mae']:.3f} | {metrics_gemini['mae']:.3f} | {'GPT-4o' if metrics_gpt4o['mae'] < metrics_gemini['mae'] else 'Gemini' if metrics_gemini['mae'] < metrics_gpt4o['mae'] else 'Tie'} |
| **Weighted Kappa** | {metrics_gpt4o['weighted_kappa']:.3f} | {metrics_gemini['weighted_kappa']:.3f} | {'GPT-4o' if metrics_gpt4o['weighted_kappa'] > metrics_gemini['weighted_kappa'] else 'Gemini' if metrics_gemini['weighted_kappa'] > metrics_gpt4o['weighted_kappa'] else 'Tie'} |
| **Samples** | {metrics_gpt4o['num_samples']} | {metrics_gemini['num_samples']} | - |

---

## 📈 Task-wise Performance

### GPT-4o

| Task | Accuracy | MAE | Weighted Kappa | Samples |
|------|----------|-----|----------------|---------|
"""

    for task, m in task_metrics_gpt4o.items():
        report += f"| {task} | {m['accuracy']:.3f} | {m['mae']:.3f} | {m['weighted_kappa']:.3f} | {m['num_samples']} |\n"

    report += "\n### Gemini 2.0 Flash\n\n"
    report += "| Task | Accuracy | MAE | Weighted Kappa | Samples |\n"
    report += "|------|----------|-----|----------------|---------|\\n"

    for task, m in task_metrics_gemini.items():
        report += f"| {task} | {m['accuracy']:.3f} | {m['mae']:.3f} | {m['weighted_kappa']:.3f} | {m['num_samples']} |\n"

    report += """
---

## 🔍 Confusion Matrix

### GPT-4o

```
"""
    report += str(metrics_gpt4o['confusion_matrix'])
    report += """
```

### Gemini 2.0 Flash

```
"""
    report += str(metrics_gemini['confusion_matrix'])
    report += """
```

---

## 📊 Classification Report

### GPT-4o

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
"""

    for cls in ['0', '1', '2', '3', '4']:
        if cls in metrics_gpt4o['classification_report']:
            rep = metrics_gpt4o['classification_report'][cls]
            report += f"| {cls} | {rep['precision']:.3f} | {rep['recall']:.3f} | {rep['f1-score']:.3f} | {int(rep['support'])} |\n"

    report += "\n### Gemini 2.0 Flash\n\n"
    report += "| Class | Precision | Recall | F1-Score | Support |\n"
    report += "|-------|-----------|--------|----------|---------|\\n"

    for cls in ['0', '1', '2', '3', '4']:
        if cls in metrics_gemini['classification_report']:
            rep = metrics_gemini['classification_report'][cls]
            report += f"| {cls} | {rep['precision']:.3f} | {rep['recall']:.3f} | {rep['f1-score']:.3f} | {int(rep['support'])} |\n"

    report += """
---

## 💡 Key Insights

"""

    # Winner determination
    gpt_wins = 0
    gemini_wins = 0

    if metrics_gpt4o['accuracy'] > metrics_gemini['accuracy']:
        gpt_wins += 1
        report += "- **Accuracy**: GPT-4o outperforms Gemini\n"
    elif metrics_gemini['accuracy'] > metrics_gpt4o['accuracy']:
        gemini_wins += 1
        report += "- **Accuracy**: Gemini outperforms GPT-4o\n"
    else:
        report += "- **Accuracy**: Tie between models\n"

    if metrics_gpt4o['mae'] < metrics_gemini['mae']:
        gpt_wins += 1
        report += "- **MAE**: GPT-4o has lower error\n"
    elif metrics_gemini['mae'] < metrics_gpt4o['mae']:
        gemini_wins += 1
        report += "- **MAE**: Gemini has lower error\n"
    else:
        report += "- **MAE**: Tie between models\n"

    if metrics_gpt4o['weighted_kappa'] > metrics_gemini['weighted_kappa']:
        gpt_wins += 1
        report += "- **Weighted Kappa**: GPT-4o shows better agreement\n"
    elif metrics_gemini['weighted_kappa'] > metrics_gpt4o['weighted_kappa']:
        gemini_wins += 1
        report += "- **Weighted Kappa**: Gemini shows better agreement\n"
    else:
        report += "- **Weighted Kappa**: Tie between models\n"

    report += f"\n**Overall Winner**: {'GPT-4o' if gpt_wins > gemini_wins else 'Gemini 2.0 Flash' if gemini_wins > gpt_wins else 'Tie'} ({gpt_wins} vs {gemini_wins})\n"

    report += """
---

## 📝 Recommendations

1. **Model Selection**:
   - For highest accuracy: Choose the model with better accuracy on your target task
   - For cost-effectiveness: Gemini 2.0 Flash offers free tier
   - For production: Consider ensemble of both models

2. **Task-Specific Insights**:
   - Compare task-wise metrics to identify model strengths
   - Consider task-specific model routing in production

3. **Future Work**:
   - Analyze per-class performance for targeted improvements
   - Investigate failure cases and reasoning quality
   - Explore prompt engineering for better results

---

**Contact**: For questions about this analysis or the PD4T dataset, contact a.dadashzadeh@bristol.ac.uk
"""

    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Report saved to: {output_path}")

def main(args):
    print("Loading results...")
    df_gpt4o, df_gemini = load_results(args.gpt4o_results, args.gemini_results)

    print(f"GPT-4o valid samples: {len(df_gpt4o)}")
    print(f"Gemini valid samples: {len(df_gemini)}")

    print("\nComputing overall metrics...")
    metrics_gpt4o = compute_metrics(df_gpt4o, "GPT-4o")
    metrics_gemini = compute_metrics(df_gemini, "Gemini 2.0 Flash")

    print("\nComputing task-wise metrics...")
    task_metrics_gpt4o = compute_task_metrics(df_gpt4o, "GPT-4o")
    task_metrics_gemini = compute_task_metrics(df_gemini, "Gemini 2.0 Flash")

    print("\nGenerating comparison report...")
    generate_markdown_report(
        metrics_gpt4o, metrics_gemini,
        task_metrics_gpt4o, task_metrics_gemini,
        args.output
    )

    # Print summary to console
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"GPT-4o Accuracy: {metrics_gpt4o['accuracy']:.3f} | MAE: {metrics_gpt4o['mae']:.3f} | Kappa: {metrics_gpt4o['weighted_kappa']:.3f}")
    print(f"Gemini Accuracy: {metrics_gemini['accuracy']:.3f} | MAE: {metrics_gemini['mae']:.3f} | Kappa: {metrics_gemini['weighted_kappa']:.3f}")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare VLM Results")
    parser.add_argument("--gpt4o_results", type=str, required=True, help="Path to GPT-4o results CSV")
    parser.add_argument("--gemini_results", type=str, required=True, help="Path to Gemini results CSV")
    parser.add_argument("--output", type=str, default="results/comparison_report.md", help="Output path for report")
    args = parser.parse_args()

    main(args)
