"""
RIM parameter diagnosis: test multiple RIM configs on first N val sequences.
Usage: python tracking/debug_rim_diagnosis.py [--num_seqs 5]
"""
import sys, os
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.insert(0, prj_path)

import torch, numpy as np, argparse
from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_sequence
from lib.test.evaluation.tracker import Tracker


def compute_iou(box1, box2):
    x1,y1,w1,h1 = box1; x2,y2,w2,h2 = box2
    xi1,yi1 = max(x1,x2), max(y1,y2)
    xi2,yi2 = min(x1+w1,x2+w2), min(y1+h1,y2+h2)
    inter = max(0,xi2-xi1)*max(0,yi2-yi1)
    return inter/(w1*h1+w2*h2-inter+1e-10)


def eval_seq(config_name, config_label):
    """Run tracking on first few sequences and compute AO."""
    tracker = Tracker('setrack', config_name, 'got10k_val', None)
    dataset = get_dataset('got10k_val')
    # First N sequences
    num = min(len(dataset), 5)
    ious_all = []

    for i in range(num):
        seq = dataset[i]
        try:
            output = tracker.run_sequence(seq, debug=False)
        except Exception as e:
            print(f'  [{config_label}] seq {seq.name}: ERROR: {e}')
            continue

        # Compute per-frame IoU
        pred = output.get('target_bbox', [])
        gt = seq.ground_truth_rect
        seq_iou = []
        for j in range(min(len(pred), len(gt))):
            if np.any(~np.isfinite(pred[j])):
                continue
            iou = compute_iou(pred[j], gt[j])
            seq_iou.append(iou)
        if seq_iou:
            ious_all.extend(seq_iou)
            print(f'  [{config_label}] {seq.name}: AO={np.mean(seq_iou):.4f} ({len(seq_iou)} frames)')
        else:
            print(f'  [{config_label}] {seq.name}: NO VALID FRAMES')

    if ious_all:
        ao = np.mean(ious_all)
        sr50 = np.mean(np.array(ious_all) > 0.5) * 100
        return {'ao': ao, 'sr50': sr50, 'n_frames': len(ious_all)}
    return {'ao': 0, 'sr50': 0, 'n_frames': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_seqs', type=int, default=5)
    args = parser.parse_args()

    # Config order: RIM OFF (baseline), original, keep075, keep0875
    configs = [
        ('vitb_256_mae_setrack_full_fixmae_got10k_10ep_eval_noprune',          'RIM OFF'),
        ('vitb_256_mae_setrack_full_fixmae_got10k_10ep_eval_rim_original',      'RIM zero (e=0.7,k=0.50)'),
        ('vitb_256_mae_setrack_full_fixmae_got10k_10ep_eval_rim_originalfill',  'RIM original-fill'),
        ('vitb_256_mae_setrack_full_fixmae_got10k_10ep_eval_rim_soft01',        'RIM soft01 (s=0.1)'),
        ('vitb_256_mae_setrack_full_fixmae_got10k_10ep_eval_rim_soft025',       'RIM soft025 (s=0.25)'),
        ('vitb_256_mae_setrack_full_fixmae_got10k_10ep_eval_rim_soft05',        'RIM soft05 (s=0.5)'),
    ]

    print(f'=== RIM Parameter Diagnosis (first {args.num_seqs} sequences) ===\n')
    results = {}

    for cfg_name, label in configs:
        print(f'Testing: {label}')
        try:
            r = eval_seq(cfg_name, label)
            results[label] = r
            print(f'  => Overall AO={r["ao"]:.4f}, SR50={r["sr50"]:.1f}%, frames={r["n_frames"]}')
        except Exception as e:
            print(f'  => FAILED: {e}')
            import traceback; traceback.print_exc()
        print()

    # Summary table
    print('='*70)
    print(f'{"Config":<45} {"AO":>8} {"SR50":>8}')
    print('-'*70)
    for label, r in results.items():
        print(f'{label:<45} {r["ao"]:>8.4f} {r["sr50"]:>7.1f}%')

    # Analysis
    rim_off = results.get('RIM OFF', {})
    rim_off_ao = rim_off.get('ao', 0)
    print(f'\n=== Analysis ===')
    print(f'PRUNING_FILL_VALUE="original" not supported by current code (only "zero" mode).')
    for label, r in results.items():
        if label == 'RIM OFF': continue
        if rim_off_ao > 0:
            ratio = r['ao'] / rim_off_ao * 100
            print(f'{label}: {ratio:.1f}% of RIM OFF')

    # If any RIM config is close to RIM OFF (>90%), recommend full 180 test
    for label, r in results.items():
        if label == 'RIM OFF': continue
        if rim_off_ao > 0 and r['ao'] / rim_off_ao > 0.9:
            print(f'\n>>> {label} is within 90% of RIM OFF - recommend full 180-seq test!')


if __name__ == '__main__':
    main()
