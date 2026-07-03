"""
Evaluate SETrack ablation configurations on GOT-10k val.
Compute AO, SR0.5, SR0.75, and compare with baseline.
"""
import sys, os
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.insert(0, prj_path)

import argparse
import numpy as np
from lib.test.evaluation.environment import env_settings
from lib.test.utils.load_text import load_text


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x, y, w, h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    return inter / (w1*h1 + w2*h2 - inter + 1e-10)


def evaluate_config(config_name, max_seqs=None):
    """Evaluate a config on GOT-10k val."""
    env = env_settings()
    result_dir = os.path.join(env.results_path, 'setrack', config_name, 'got10k')
    val_dir = os.path.join(env.got10k_path, 'val')

    if not os.path.isdir(result_dir):
        print(f"ERROR: Result dir not found: {result_dir}")
        return None

    result_files = sorted([f for f in os.listdir(result_dir) if f.endswith('.txt') and 'time' not in f])
    if not result_files:
        print(f"  No result files found in {result_dir}")
        return None

    if max_seqs:
        result_files = result_files[:max_seqs]

    all_ious = []
    seq_ious = {}
    nan_count = 0

    for rf in result_files:
        pred_path = os.path.join(result_dir, rf)
        seq_name = rf.replace('.txt', '')
        gt_path = os.path.join(val_dir, seq_name, 'groundtruth.txt')

        if not os.path.exists(gt_path):
            print(f"  WARN: GT missing for {seq_name}")
            continue

        try:
            pred = load_text(pred_path, delimiter='\t', dtype=np.float64)
            gt = load_text(gt_path, delimiter=',', dtype=np.float64)
        except Exception as e:
            print(f"  ERROR reading {seq_name}: {e}")
            continue

        n_frames = min(len(pred), len(gt))
        seq_iou = []
        for i in range(n_frames):
            if np.any(~np.isfinite(pred[i])):
                nan_count += 1
                continue
            iou = compute_iou(pred[i], gt[i])
            seq_iou.append(iou)

        if seq_iou:
            seq_mean = np.mean(seq_iou)
            seq_ious[seq_name] = (seq_mean, len(seq_iou))
            all_ious.extend(seq_iou)
        else:
            seq_ious[seq_name] = (0.0, 0)

    # Compute metrics
    if not all_ious:
        print(f"  ERROR: No valid IoUs found!")
        return None

    ious_arr = np.array(all_ious)
    ao = np.mean(ious_arr)
    sr50 = np.mean(ious_arr > 0.5) * 100
    sr75 = np.mean(ious_arr > 0.75) * 100

    result = {
        'config': config_name,
        'sequences_evaluated': len(seq_ious),
        'total_frames': len(all_ious),
        'nan_frames': nan_count,
        'ao': ao,
        'sr0.5': sr50,
        'sr0.75': sr75,
        'seq_ious': seq_ious,
    }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, nargs='+',
                       default=['vitb_256_mae_setrack_base_got10k_10ep',
                               'vitb_256_mae_setrack_cross_got10k_10ep',
                               'vitb_256_mae_setrack_prune_got10k_10ep',
                               'vitb_256_mae_setrack_got10k_10ep'])
    parser.add_argument('--max_seqs', type=int, default=None, help='Max sequences to evaluate')
    args = parser.parse_args()

    print("\n=== SETrack Ablation Evaluation (GOT-10k Val) ===\n")

    results = {}
    for cfg in args.configs:
        print(f"Evaluating {cfg}...")
        result = evaluate_config(cfg, args.max_seqs)
        if result:
            results[cfg] = result
            print(f"  AO:    {result['ao']:.4f}")
            print(f"  SR0.5: {result['sr0.5']:.2f}%")
            print(f"  SR0.75: {result['sr0.75']:.2f}%")
            print(f"  Seqs:  {result['sequences_evaluated']}")
        else:
            print(f"  FAILED to evaluate")
        print()

    # Print comparison table
    if len(results) > 1:
        print("\n=== COMPARISON TABLE ===\n")
        print(f"{'Config':<50} {'AO':<10} {'SR0.5':<10} {'SR0.75':<10} {'Seqs':<8}")
        print("-" * 88)
        for cfg in args.configs:
            if cfg in results:
                r = results[cfg]
                print(f"{cfg:<50} {r['ao']:<10.4f} {r['sr0.5']:<10.2f} {r['sr0.75']:<10.2f} {r['sequences_evaluated']:<8}")

        # Compute deltas relative to full
        if 'vitb_256_mae_setrack_got10k_10ep' in results:
            full_ao = results['vitb_256_mae_setrack_got10k_10ep']['ao']
            print(f"\n(relative to full config: AO={full_ao:.4f})")
            print(f"{'Config':<50} {'ΔAO':<10} {'ΔSR0.5':<10} {'ΔSR0.75':<10}")
            print("-" * 80)
            for cfg in args.configs:
                if cfg in results and cfg != 'vitb_256_mae_setrack_got10k_10ep':
                    r = results[cfg]
                    delta_ao = r['ao'] - full_ao
                    full_sr50 = results['vitb_256_mae_setrack_got10k_10ep']['sr0.5']
                    full_sr75 = results['vitb_256_mae_setrack_got10k_10ep']['sr0.75']
                    delta_sr50 = r['sr0.5'] - full_sr50
                    delta_sr75 = r['sr0.75'] - full_sr75
                    print(f"{cfg:<50} {delta_ao:<10.4f} {delta_sr50:<10.2f} {delta_sr75:<10.2f}")


if __name__ == '__main__':
    main()
