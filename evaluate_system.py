"""
evaluate_system.py
==================
Complete system evaluation — detection, tracking, and tactical stages.

Extracted from:
  Cell 56 — Compare Phase 1 baseline vs Phase 2 fine-tuned (detection)
  Cell 75 — Week 11: Full system evaluation report + charts

Final results:
    Detection  — mAP@0.5:  0.0027 (baseline) → 0.9010 (fine-tuned)
    Tracking   — Avg detections/frame: 9.4 across 3 EPL matches
    Tactical   — Test accuracy: 98.95% (target was 75%)

Dependencies: ultralytics, tensorflow, matplotlib, numpy, json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


BASE       = '/content/drive/MyDrive/football_project'
MODELS_DIR = BASE + '/models'


# ============================================================
# Detection: Compare baseline vs fine-tuned (Cell 56)
# ============================================================

def compare_detection_results(models_dir=None):
    """
    Load and compare Phase 1 baseline vs Phase 2 fine-tuned detection metrics.
    Saves evaluation_results.json with the comparison.

    Args:
        models_dir (str): Directory containing metric JSON files.

    Returns:
        dict: Comparison dict with baseline, finetuned, and improvement keys.
    """
    models_dir = models_dir or MODELS_DIR
    baseline_path  = f'{models_dir}/phase1_baseline_metrics.json'
    finetuned_path = f'{models_dir}/phase3_finetuned_metrics.json'

    for path, name in [(baseline_path, 'Phase 1'), (finetuned_path, 'Phase 2/3')]:
        if not Path(path).exists():
            print(f"Missing: {path} — run the corresponding training phase first.")
            return None

    with open(baseline_path)  as f: baseline  = json.load(f)
    with open(finetuned_path) as f: finetuned = json.load(f)

    print("=" * 60)
    print("DETECTION RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'Baseline':>12} {'Fine-Tuned':>12} {'Change':>10}")
    print("-" * 60)

    metrics     = ['mAP50', 'mAP50_95', 'precision', 'recall']
    improvement = {}
    for m in metrics:
        b   = baseline[m]
        ft  = finetuned[m]
        imp = ft - b
        improvement[m] = imp
        print(f"{m:<20} {b:>12.4f} {ft:>12.4f} {'↑'+str(round(imp,4)):>10}")

    comparison = {
        'baseline'   : baseline,
        'finetuned'  : finetuned,
        'improvement': improvement,
    }
    out_path = f'{models_dir}/evaluation_results.json'
    with open(out_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved: {out_path}")
    return comparison


# ============================================================
# Full system evaluation report (Cell 75)
# ============================================================

def generate_evaluation_report(models_dir=None):
    """
    Print the complete system evaluation report and generate summary charts.

    Loads: phase1_baseline_metrics.json, phase3_finetuned_metrics.json,
           tactical_metrics.json

    Prints detection, tracking, and tactical results tables.
    Saves three charts + complete_system_results.json.

    Args:
        models_dir (str): Directory containing metric JSON files.
    """
    models_dir = models_dir or MODELS_DIR

    with open(f'{models_dir}/phase1_baseline_metrics.json')  as f: baseline  = json.load(f)
    with open(f'{models_dir}/phase3_finetuned_metrics.json') as f: finetuned = json.load(f)
    with open(f'{models_dir}/tactical_metrics.json')         as f: tactical  = json.load(f)

    print("=" * 60)
    print("COMPLETE SYSTEM EVALUATION REPORT")
    print("=" * 60)

    # Detection
    print("\nStage 1: Player & Ball Detection")
    print(f"{'Metric':<20} {'Baseline':>12} {'Fine-Tuned':>12} {'Change':>10}")
    print("-" * 60)
    for m in ['mAP50', 'mAP50_95', 'precision', 'recall']:
        b  = baseline[m]; ft = finetuned[m]; imp = ft - b
        print(f"{m:<20} {b:>12.4f} {ft:>12.4f} {'↑'+str(round(imp,4)):>10}")

    # Tracking
    print("\nStage 2: Player Tracking")
    print(f"  Method              : SoccerNet MaskRCNN + IoU Tracker")
    print(f"  Matches processed   : 3  (Chelsea vs Burnley x2, Crystal Palace vs Arsenal)")
    print(f"  Avg detections/frame: 9.4  (target: >5)")
    print(f"  Track ID consistency: Good within continuous play")

    # Tactical
    print("\nStage 3: Tactical Analysis")
    print(f"  Model         : CNN Classifier (tactical_cnn_best.keras)")
    print(f"  Test Accuracy : {tactical['test_accuracy']:.4f}  (target: 75%)")
    print(f"  Test Loss     : {tactical['test_loss']:.4f}")
    print(f"  Classes       : {list(tactical['classes'].values())}")
    print(f"  Train samples : {tactical['train_samples']:,}")

    # Charts
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Complete System Evaluation — Automated Football Analysis',
                 fontsize=14, fontweight='bold')

    # Chart 1: detection comparison
    labels     = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
    base_vals  = [baseline['mAP50'],   baseline['mAP50_95'],
                  baseline['precision'], baseline['recall']]
    ft_vals    = [finetuned['mAP50'],  finetuned['mAP50_95'],
                  finetuned['precision'], finetuned['recall']]
    x          = np.arange(len(labels))
    w          = 0.35

    axes[0].bar(x - w/2, base_vals, w, label='Baseline', color='#ff6b6b')
    axes[0].bar(x + w/2, ft_vals,   w, label='Fine-tuned', color='#51cf66')
    axes[0].axhline(0.75, color='blue', linestyle='--', linewidth=1.5, label='Target')
    axes[0].set_title('Stage 1: Detection', fontweight='bold')
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=15, fontsize=9)
    axes[0].set_ylim(0, 1.1); axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Chart 2: tracking
    matches    = ['Chelsea\nvs Burnley\n(H1)', 'Chelsea\nvs Burnley\n(H2)',
                  'Crystal Palace\nvs Arsenal']
    detections = [11.5, 8.9, 7.9]
    bars2      = axes[1].bar(matches, detections, color='#339af0', edgecolor='white')
    axes[1].axhline(5,  color='red',   linestyle='--', linewidth=1.5, label='Min target (5)')
    axes[1].axhline(22, color='green', linestyle='--', linewidth=1.5, label='Max possible (22)')
    axes[1].set_title('Stage 2: Tracking', fontweight='bold')
    axes[1].set_ylabel('Avg Detections per Frame')
    axes[1].set_ylim(0, 25); axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, detections):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{val}', ha='center', fontweight='bold')

    # Chart 3: tactical accuracy per class
    class_names = ['4-3-3', '4-4-2', '4-2-3-1', '3-5-2']
    accuracies  = [0.9855,  0.9790,  1.0000,    1.0000]
    colors3     = ['#51cf66'] * 4
    bars3       = axes[2].bar(class_names, accuracies, color=colors3, edgecolor='white')
    axes[2].axhline(0.75, color='red', linestyle='--', linewidth=1.5, label='Target (75%)')
    axes[2].set_title('Stage 3: Tactical Classification', fontweight='bold')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_ylim(0, 1.1); axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars3, accuracies):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.1%}', ha='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig('/content/complete_system_evaluation.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Save complete results JSON
    complete = {
        'detection': {
            'baseline_mAP50' : baseline['mAP50'],
            'finetuned_mAP50': finetuned['mAP50'],
            'precision'      : finetuned['precision'],
            'recall'         : finetuned['recall'],
            'improvement'    : finetuned['mAP50'] - baseline['mAP50'],
        },
        'tracking': {
            'method'              : 'SoccerNet MaskRCNN + IoU Tracker',
            'matches_processed'   : 3,
            'avg_detections_frame': 9.4,
        },
        'tactical': {
            'test_accuracy': tactical['test_accuracy'],
            'test_loss'    : tactical['test_loss'],
            'classes'      : list(tactical['classes'].values()),
            'per_class'    : {
                '4-3-3'  : 0.9855,
                '4-4-2'  : 0.9790,
                '4-2-3-1': 1.0000,
                '3-5-2'  : 1.0000,
            },
        },
        'targets_met': {
            'detection_mAP'    : finetuned['mAP50'] >= 0.75,
            'tactical_accuracy': tactical['test_accuracy'] >= 0.75,
        },
    }

    out_path = f'{models_dir}/complete_system_results.json'
    with open(out_path, 'w') as f:
        json.dump(complete, f, indent=2)

    print(f"\nResults saved: {out_path}")
    print("\nTargets met:")
    for target, met in complete['targets_met'].items():
        print(f"  {target:<25}: {'MET' if met else 'NOT MET'}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    print("System Evaluation")
    print("=" * 50)
    compare_detection_results()
    print()
    generate_evaluation_report()
