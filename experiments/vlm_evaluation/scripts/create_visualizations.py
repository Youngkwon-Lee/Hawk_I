"""
Create publication-quality visualizations for GPT-5.1 VLM evaluation results
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Set style for publication quality
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Load results
results_path = "C:/Users/YK/tulip/VLM_commercial/results/gpt51_results.csv"
df = pd.read_csv(results_path)

# Create output directory
output_dir = "C:/Users/YK/tulip/VLM_commercial/results/figures"
os.makedirs(output_dir, exist_ok=True)

# Calculate metrics per task
tasks = df['task'].unique()
metrics_data = []

for task in tasks:
    task_df = df[df['task'] == task]
    acc = (task_df['pred_score'] == task_df['gt_score']).mean()

    # Calculate F1, Recall, Precision (Macro)
    from sklearn.metrics import f1_score, recall_score, precision_score
    f1 = f1_score(task_df['gt_score'], task_df['pred_score'], average='macro', zero_division=0)
    recall = recall_score(task_df['gt_score'], task_df['pred_score'], average='macro', zero_division=0)
    precision = precision_score(task_df['gt_score'], task_df['pred_score'], average='macro', zero_division=0)

    metrics_data.append({
        'Task': task,
        'Accuracy': acc,
        'F1-Score': f1,
        'Recall': recall,
        'Precision': precision,
        'N': len(task_df)
    })

metrics_df = pd.DataFrame(metrics_data)

# Figure 1: Task performance comparison (Bar chart)
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(tasks))
width = 0.2

bars1 = ax.bar(x - 1.5*width, metrics_df['Accuracy'], width, label='Accuracy', alpha=0.8)
bars2 = ax.bar(x - 0.5*width, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8)
bars3 = ax.bar(x + 0.5*width, metrics_df['Recall'], width, label='Recall', alpha=0.8)
bars4 = ax.bar(x + 1.5*width, metrics_df['Precision'], width, label='Precision', alpha=0.8)

ax.set_xlabel('Task')
ax.set_ylabel('Score')
ax.set_title('GPT-5.1 Performance Metrics by Task')
ax.set_xticks(x)
ax.set_xticklabels(tasks, rotation=15, ha='right')
ax.legend()
ax.set_ylim([0, 0.8])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure1_task_performance.png'), bbox_inches='tight')
print("[OK] Figure 1 saved: Task performance comparison")
plt.close()

# Figure 2: Confusion matrices (2x2 grid for 4 tasks)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, task in enumerate(tasks):
    task_df = df[df['task'] == task]
    cm = confusion_matrix(task_df['gt_score'], task_df['pred_score'], labels=[0, 1, 2, 3, 4])

    # Normalize by row (GT)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                ax=axes[idx], cbar=True, vmin=0, vmax=1,
                xticklabels=[0, 1, 2, 3, 4], yticklabels=[0, 1, 2, 3, 4])
    axes[idx].set_title(f'{task} (N={len(task_df)})')
    axes[idx].set_xlabel('Predicted Score')
    axes[idx].set_ylabel('Ground Truth Score')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure2_confusion_matrices.png'), bbox_inches='tight')
print("[OK] Figure 2 saved: Confusion matrices")
plt.close()

# Figure 3: GT vs Pred score distribution comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()

for idx, task in enumerate(tasks):
    task_df = df[df['task'] == task]

    gt_counts = task_df['gt_score'].value_counts().sort_index()
    pred_counts = task_df['pred_score'].value_counts().sort_index()

    # Ensure all scores 0-4 are present
    for score in range(5):
        if score not in gt_counts:
            gt_counts[score] = 0
        if score not in pred_counts:
            pred_counts[score] = 0

    gt_counts = gt_counts.sort_index()
    pred_counts = pred_counts.sort_index()

    x = np.arange(5)
    width = 0.35

    axes[idx].bar(x - width/2, gt_counts, width, label='Ground Truth', alpha=0.8)
    axes[idx].bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)

    axes[idx].set_xlabel('UPDRS Score')
    axes[idx].set_ylabel('Count')
    axes[idx].set_title(f'{task} (N={len(task_df)})')
    axes[idx].set_xticks(x)
    axes[idx].legend()
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure3_score_distribution.png'), bbox_inches='tight')
print("[OK] Figure 3 saved: GT vs Pred distribution")
plt.close()

# Figure 4: Gait task model comparison
# Data from CSV (skeleton, kinematic, both, Transformer, VideoMamba, GPT-5.1)
gait_models = {
    'Skeleton\nXGBoost': 0.384,
    'Skeleton\nSVM': 0.480,
    'Kinematic\nSVM': 0.562,
    'Both\nMLP': 0.603,
    'Skeleton\nTransformer': 0.507,
    'Kinematic\nTransformer': 0.507,
    'VideoMamba\n16f': 0.534,
    'VideoMamba\n32f': 0.507,
    'GPT-5.1': 0.644
}

fig, ax = plt.subplots(figsize=(10, 6))
models = list(gait_models.keys())
accs = list(gait_models.values())
colors = ['#1f77b4']*8 + ['#d62728']  # Highlight GPT-5.1 in red

bars = ax.bar(models, accs, color=colors, alpha=0.8)
ax.set_xlabel('Model')
ax.set_ylabel('Accuracy')
ax.set_title('Gait Task: Model Comparison')
ax.set_ylim([0, 0.7])
ax.grid(axis='y', alpha=0.3)
ax.tick_params(axis='x', rotation=45)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure4_gait_model_comparison.png'), bbox_inches='tight')
print("[OK] Figure 4 saved: Gait model comparison")
plt.close()

# Figure 5: Per-class accuracy heatmap
per_class_acc = []
class_labels = [0, 1, 2, 3, 4]

for task in tasks:
    task_df = df[df['task'] == task]
    task_acc = []
    for score in class_labels:
        score_df = task_df[task_df['gt_score'] == score]
        if len(score_df) > 0:
            acc = (score_df['pred_score'] == score_df['gt_score']).mean()
        else:
            acc = 0
        task_acc.append(acc)
    per_class_acc.append(task_acc)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(per_class_acc, annot=True, fmt='.2f', cmap='RdYlGn',
            ax=ax, cbar=True, vmin=0, vmax=1,
            xticklabels=class_labels, yticklabels=tasks)
ax.set_title('Per-Class Accuracy by Task')
ax.set_xlabel('UPDRS Score')
ax.set_ylabel('Task')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure5_per_class_accuracy.png'), bbox_inches='tight')
print("[OK] Figure 5 saved: Per-class accuracy heatmap")
plt.close()

# Figure 6: Overall confusion matrix
cm_overall = confusion_matrix(df['gt_score'], df['pred_score'], labels=[0, 1, 2, 3, 4])
cm_normalized = cm_overall.astype('float') / cm_overall.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            ax=ax, cbar=True, vmin=0, vmax=1,
            xticklabels=[0, 1, 2, 3, 4], yticklabels=[0, 1, 2, 3, 4])
ax.set_title('Overall Confusion Matrix (N=485, Normalized by GT)')
ax.set_xlabel('Predicted Score')
ax.set_ylabel('Ground Truth Score')

# Add counts in parentheses
for i in range(5):
    for j in range(5):
        count = cm_overall[i, j]
        if count > 0:
            text = ax.texts[i*5 + j]
            current_text = text.get_text()
            text.set_text(f'{current_text}\n({count})')
            text.set_fontsize(8)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure6_overall_confusion_matrix.png'), bbox_inches='tight')
print("[OK] Figure 6 saved: Overall confusion matrix")
plt.close()

print("\n" + "="*60)
print("All 6 publication-quality PNG figures generated successfully!")
print("="*60)
print(f"\nOutput directory: {output_dir}")
print("\nGenerated figures:")
print("  1. figure1_task_performance.png - Task-wise performance metrics")
print("  2. figure2_confusion_matrices.png - 2x2 grid of confusion matrices")
print("  3. figure3_score_distribution.png - GT vs Pred distribution")
print("  4. figure4_gait_model_comparison.png - Gait task model comparison")
print("  5. figure5_per_class_accuracy.png - Per-class accuracy heatmap")
print("  6. figure6_overall_confusion_matrix.png - Overall confusion matrix")
