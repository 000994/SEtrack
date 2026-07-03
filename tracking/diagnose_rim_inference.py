"""
Diagnose RIM (Redundant Information Pruning) inference behavior.
Compare: RIM on vs RIM off using same checkpoint, collect statistics.
"""
import sys, os
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.insert(0, prj_path)

import torch
import numpy as np
import argparse
from lib.config.setrack.config import cfg, update_config_from_file
from lib.models.setrack.setrack import build_setrack
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


def diagnose_rim_on_single_seq(config_name, seq_name='GOT-10k_Val_000001'):
    """Test RIM-on inference on a single sequence."""
    env = env_settings()
    update_config_from_file(os.path.join(env.prj_dir, f'experiments/setrack/{config_name}.yaml'))

    print(f"\n=== RIM ON: {config_name} ===")
    print(f"USE_REDUNDANT_PRUNING: {cfg.MODEL.BACKBONE.USE_REDUNDANT_PRUNING}")
    print(f"USE_CROSS_SEMANTIC: {cfg.MODEL.BACKBONE.USE_CROSS_SEMANTIC}")

    # Build model
    model = build_setrack(cfg)
    model = model.cuda()
    model.eval()

    # Load checkpoint
    checkpoint_path = os.path.join(env.save_dir, f"checkpoints/train/setrack/{config_name}/SETrack_ep0010.pth.tar")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['net'], strict=True)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"WARNING: Checkpoint not found: {checkpoint_path}")
        return None

    # Load sequence data
    val_dir = os.path.join(env.got10k_path, 'val', seq_name)
    frames_dir = os.path.join(val_dir, 'img')
    gt_path = os.path.join(val_dir, 'groundtruth.txt')

    if not os.path.exists(gt_path):
        print(f"ERROR: GT not found: {gt_path}")
        return None

    gt = load_text(gt_path, delimiter=',', dtype=np.float64)
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])

    print(f"Sequence: {seq_name}, frames: {len(frame_files)}, GT boxes: {len(gt)}")

    # Test inference on first 5 frames
    from lib.test.tracker.setrack import SETRACKTracker
    from lib.test.parameter.setrack import parameters

    params = parameters(config_name)
    tracker = SETRACKTracker(params)

    ious = []
    pruning_stats = {'keep_nums': [], 'remove_nums': [], 'scores_mean': []}

    for frame_idx in range(min(5, len(frame_files))):
        frame_path = os.path.join(frames_dir, frame_files[frame_idx])
        from lib.utils.load_text import imread
        image = imread(frame_path)

        # Initialize on first frame
        if frame_idx == 0:
            bbox = gt[0]  # [x, y, w, h]
            tracker.initialize(image, {'init_bbox': bbox})
        else:
            # Track
            output = tracker.track(image)
            pred_bbox = output['target_bbox']
            gt_bbox = gt[frame_idx]
            iou = compute_iou(pred_bbox, gt_bbox)
            ious.append(iou)
            print(f"  Frame {frame_idx}: IoU={iou:.4f}, pred={pred_bbox}, gt={gt_bbox}")

    if ious:
        print(f"Mean IoU (first 5 frames): {np.mean(ious):.4f}")

    return {'mean_iou': np.mean(ious) if ious else 0, 'ious': ious}


def diagnose_rim_off_single_seq(config_name_base, seq_name='GOT-10k_Val_000001'):
    """Test RIM-off inference by loading full checkpoint with RIM disabled."""
    env = env_settings()
    config_name_noprune = config_name_base.replace('10ep', '10ep_eval_noprune')
    update_config_from_file(os.path.join(env.prj_dir, f'experiments/setrack/{config_name_noprune}.yaml'))

    print(f"\n=== RIM OFF (eval config): {config_name_noprune} ===")
    print(f"USE_REDUNDANT_PRUNING: {cfg.MODEL.BACKBONE.USE_REDUNDANT_PRUNING}")
    print(f"USE_CROSS_SEMANTIC: {cfg.MODEL.BACKBONE.USE_CROSS_SEMANTIC}")

    # Build model
    model = build_setrack(cfg)
    model = model.cuda()
    model.eval()

    # Load checkpoint from full config
    checkpoint_path = os.path.join(env.save_dir, f"checkpoints/train/setrack/{config_name_base}/SETrack_ep0010.pth.tar")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['net'], strict=False)  # strict=False for config mismatch
        print(f"Loaded checkpoint (from full config): {checkpoint_path}")
    else:
        print(f"WARNING: Checkpoint not found: {checkpoint_path}")
        return None

    # Load sequence data
    val_dir = os.path.join(env.got10k_path, 'val', seq_name)
    frames_dir = os.path.join(val_dir, 'img')
    gt_path = os.path.join(val_dir, 'groundtruth.txt')

    if not os.path.exists(gt_path):
        print(f"ERROR: GT not found: {gt_path}")
        return None

    gt = load_text(gt_path, delimiter=',', dtype=np.float64)
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])

    print(f"Sequence: {seq_name}, frames: {len(frame_files)}, GT boxes: {len(gt)}")

    # Test inference on first 5 frames
    from lib.test.tracker.setrack import SETRACKTracker
    from lib.test.parameter.setrack import parameters

    params = parameters(config_name_noprune)
    tracker = SETRACKTracker(params)

    ious = []

    for frame_idx in range(min(5, len(frame_files))):
        frame_path = os.path.join(frames_dir, frame_files[frame_idx])
        from lib.utils.load_text import imread
        image = imread(frame_path)

        # Initialize on first frame
        if frame_idx == 0:
            bbox = gt[0]  # [x, y, w, h]
            tracker.initialize(image, {'init_bbox': bbox})
        else:
            # Track
            output = tracker.track(image)
            pred_bbox = output['target_bbox']
            gt_bbox = gt[frame_idx]
            iou = compute_iou(pred_bbox, gt_bbox)
            ious.append(iou)
            print(f"  Frame {frame_idx}: IoU={iou:.4f}, pred={pred_bbox}, gt={gt_bbox}")

    if ious:
        print(f"Mean IoU (first 5 frames): {np.mean(ious):.4f}")

    return {'mean_iou': np.mean(ious) if ious else 0, 'ious': ious}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='vitb_256_mae_setrack_got10k_10ep')
    parser.add_argument('--seq', type=str, default='GOT-10k_Val_000001')
    args = parser.parse_args()

    print(f"RIM Inference Diagnosis")
    print(f"Config: {args.config}")
    print(f"Sequence: {args.seq}")

    try:
        result_rim_on = diagnose_rim_on_single_seq(args.config, args.seq)
    except Exception as e:
        print(f"ERROR in RIM-on test: {e}")
        import traceback
        traceback.print_exc()
        result_rim_on = None

    try:
        result_rim_off = diagnose_rim_off_single_seq(args.config, args.seq)
    except Exception as e:
        print(f"ERROR in RIM-off test: {e}")
        import traceback
        traceback.print_exc()
        result_rim_off = None

    # Compare results
    if result_rim_on and result_rim_off:
        print(f"\n=== COMPARISON ===")
        print(f"RIM ON  - Mean IoU: {result_rim_on['mean_iou']:.4f}")
        print(f"RIM OFF - Mean IoU: {result_rim_off['mean_iou']:.4f}")
        diff = result_rim_off['mean_iou'] - result_rim_on['mean_iou']
        print(f"Difference (OFF - ON): {diff:+.4f}")
        if diff > 0:
            print(f"→ RIM OFF is better by {abs(diff):.4f}")
        elif diff < 0:
            print(f"→ RIM ON is better by {abs(diff):.4f}")
        else:
            print(f"→ No difference")


if __name__ == '__main__':
    main()
