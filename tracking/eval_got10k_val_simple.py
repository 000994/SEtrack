"""
Simple GOT-10k val evaluation script.
Computes AO, SR0.5, SR0.75 from tracking result files.
Usage: python tracking/eval_got10k_val_simple.py --tracker setrack --cfg vitb_256_mae_setrack_got10k_10ep
"""
import sys, os; prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path: sys.path.insert(0, prj_path)

import argparse, numpy as np
from lib.test.evaluation.environment import env_settings
from lib.test.utils.load_text import load_text


def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1; x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    return inter / (w1*h1 + w2*h2 - inter + 1e-10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracker', type=str, default='setrack')
    parser.add_argument('--cfg', type=str, default='vitb_256_mae_setrack_got10k_10ep')
    args = parser.parse_args()

    env = env_settings()
    result_dir = os.path.join(env.results_path, args.tracker, args.cfg, 'got10k')
    val_dir = os.path.join(env.got10k_path, 'val')

    assert os.path.isdir(result_dir), f"Result dir not found: {result_dir}"
    assert os.path.isdir(val_dir), f"Val dir not found: {val_dir}"

    result_files = sorted([f for f in os.listdir(result_dir) if f.endswith('.txt') and 'time' not in f])
    if not result_files:
        print("No result files found!")
        return

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
    ious_arr = np.array(all_ious)
    ao = np.mean(ious_arr) if len(ious_arr) > 0 else 0
    sr50 = np.mean(ious_arr > 0.5) * 100 if len(ious_arr) > 0 else 0
    sr75 = np.mean(ious_arr > 0.75) * 100 if len(ious_arr) > 0 else 0

    print(f"\n=== GOT-10k Val Evaluation ===")
    print(f"Sequences evaluated: {len(seq_ious)}")
    print(f"Total frames:        {len(all_ious)}")
    print(f"NaN frames:          {nan_count}")
    print(f"AO:                  {ao:.4f}")
    print(f"SR0.5:               {sr50:.2f}%")
    print(f"SR0.75:              {sr75:.2f}%")
    print(f"\nTop 5 sequences by IoU:")
    for name, (iou_val, frames) in sorted(seq_ious.items(), key=lambda x: -x[1][0])[:5]:
        print(f"  {name}: AO={iou_val:.4f} ({frames} frames)")
    print(f"\nBottom 5 sequences by IoU:")
    for name, (iou_val, frames) in sorted(seq_ious.items(), key=lambda x: x[1][0])[:5]:
        print(f"  {name}: AO={iou_val:.4f} ({frames} frames)")
    print(f"\nNOTE: Local evaluation, not official GOT-10k test server result.")


if __name__ == '__main__':
    main()
