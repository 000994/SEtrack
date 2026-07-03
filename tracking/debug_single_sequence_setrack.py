"""
SETrack single-sequence debug script (Phase 9).
Tests loading smoke checkpoint and running tracker on one GOT-10k val sequence.
"""
import sys, os
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.insert(0, prj_path)

import argparse
import torch
import numpy as np
import cv2 as cv

from lib.test.tracker.setrack import SETrack
from lib.test.utils import TrackerParams
from lib.config.setrack.config import cfg, update_config_from_file


def compute_iou(box1, box2):
    """Compute IoU between two XYWH boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    area1, area2 = w1*h1, w2*h2
    return inter / (area1 + area2 - inter + 1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='vitb_256_mae_setrack_smoke_got10k')
    parser.add_argument('--checkpoint', type=str,
                        default='output/checkpoints/train/setrack/vitb_256_mae_setrack_smoke_got10k/SETrack_ep0001.pth.tar')
    parser.add_argument('--sequence', type=str, default='data/got10k/val/GOT-10k_Val_000001')
    parser.add_argument('--num_frames', type=int, default=0, help='0=all frames')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = args.device
    torch.cuda.set_device(device)

    # ---- 1. Load config ----
    yaml_file = os.path.join(prj_path, 'experiments/setrack/%s.yaml' % args.config)
    assert os.path.exists(yaml_file), f"Config not found: {yaml_file}"
    update_config_from_file(yaml_file)
    print("Config loaded:", args.config)

    # ---- 2. Build params ----
    params = TrackerParams()
    params.cfg = cfg
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.checkpoint = os.path.join(prj_path, args.checkpoint)
    params.debug = 0
    params.save_all_boxes = False
    print("Checkpoint:", params.checkpoint)
    assert os.path.exists(params.checkpoint), f"Checkpoint not found: {params.checkpoint}"

    # ---- 3. Load sequence info ----
    seq_path = os.path.join(prj_path, args.sequence)
    assert os.path.isdir(seq_path), f"Sequence dir not found: {seq_path}"
    frames = sorted([f for f in os.listdir(seq_path) if f.endswith('.jpg')],
                    key=lambda f: int(f[:-4]))
    gt_file = os.path.join(seq_path, 'groundtruth.txt')
    gt = np.loadtxt(gt_file, delimiter=',')
    print(f"Sequence: {os.path.basename(seq_path)}")
    print(f"Frames: {len(frames)}, GT shape: {gt.shape}")

    # ---- 4. Initialize tracker ----
    tracker = SETrack(params, dataset_name='got10k_val')
    init_bbox = gt[0].tolist()  # [x, y, w, h]
    first_img = cv.imread(os.path.join(seq_path, frames[0]))
    first_img_rgb = cv.cvtColor(first_img, cv.COLOR_BGR2RGB)
    init_info = {'init_bbox': init_bbox}
    tracker.initialize(first_img_rgb, init_info)
    print(f"Tracker initialized with bbox: {init_bbox}")

    # ---- 5. Track frame by frame ----
    num_frames = args.num_frames if args.num_frames > 0 else len(frames)
    num_frames = min(num_frames, len(frames))
    ious = []
    pred_boxes = [init_bbox]
    times = []

    for i in range(1, num_frames):
        img = cv.imread(os.path.join(seq_path, frames[i]))
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        out = tracker.track(img_rgb)
        t1.record()
        torch.cuda.synchronize()
        elapsed = t0.elapsed_time(t1)

        pred_box = out['target_bbox']
        pred_boxes.append(pred_box)
        times.append(elapsed)

        gt_box = gt[i].tolist()
        iou = compute_iou(pred_box, gt_box)
        ious.append(iou)

        if i <= 5:
            print(f"  Frame {i:3d}: pred={[round(v,1) for v in pred_box]}, "
                  f"gt={[round(v,1) for v in gt_box]}, IoU={iou:.4f}, time={elapsed:.1f}ms")

        # Check for NaN/Inf
        if np.any(~np.isfinite(pred_box)):
            print(f"  ERROR Frame {i}: NaN/Inf in pred_box! {pred_box}")
            break

    # ---- 6. Summary ----
    print()
    print("=== Summary ===")
    print(f"Frames tracked: {num_frames - 1}")
    print(f"NaN/Inf: NONE" if all(np.all(np.isfinite(b)) for b in pred_boxes) else "FOUND!")
    in_range = all(0 <= b[0] and 0 <= b[1] and b[2] > 0 and b[3] > 0 for b in pred_boxes if np.any(np.isfinite(b)))
    print(f"Bbox in valid range: {'YES' if in_range else 'NO'}")
    mean_iou = np.mean(ious) if ious else 0
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Avg track time: {np.mean(times):.1f}ms")
    print(f"Tracking FPS: {1000.0/np.mean(times):.1f}" if times else "N/A")
    print()
    print("NOTE: This is smoke checkpoint (1 epoch). Results are NOT representative of final performance.")


if __name__ == '__main__':
    main()
