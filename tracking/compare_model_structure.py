"""
Compare SETrack vs OSTrack model structure and parameter counts.
"""
import sys, os
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.insert(0, prj_path)

import torch
from lib.config.setrack.config import cfg as setrack_cfg, update_config_from_file as setrack_update
from lib.config.ostrack.config import cfg as ostrack_cfg, update_config_from_file as ostrack_update
from lib.models.setrack.setrack import build_setrack
from lib.models.ostrack.ostrack import build_ostrack
from lib.test.evaluation.environment import env_settings


def count_params(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_setrack():
    env = env_settings()
    setrack_update(os.path.join(env.prj_dir, 'experiments/setrack/vitb_256_mae_setrack_got10k_10ep.yaml'))
    model = build_setrack(setrack_cfg)

    print("\n=== SETrack Model Structure ===")
    print(f"Total params: {count_params(model) / 1e6:.2f}M")
    print(f"Backbone type: {setrack_cfg.MODEL.BACKBONE.TYPE}")
    print(f"Head type: {setrack_cfg.MODEL.HEAD.TYPE}")

    # Check key components
    if hasattr(model, 'backbone'):
        bb = model.backbone
        print(f"\nBackbone components:")
        if hasattr(bb, 'shared_self_blocks'):
            print(f"  shared_self_blocks: {len(bb.shared_self_blocks)} blocks")
            print(f"    Params: {sum(p.numel() for p in bb.shared_self_blocks.parameters()) / 1e6:.2f}M")
        if hasattr(bb, 'cross_semantic_blocks'):
            print(f"  cross_semantic_blocks: {len(bb.cross_semantic_blocks)} blocks")
            print(f"    Params: {sum(p.numel() for p in bb.cross_semantic_blocks.parameters()) / 1e6:.2f}M")
        if hasattr(bb, 'redundant_pruning'):
            print(f"  redundant_pruning: {type(bb.redundant_pruning).__name__}")
            print(f"    Params: {sum(p.numel() for p in bb.redundant_pruning.parameters()) / 1e6:.2f}M")
        print(f"  patch_embed:")
        print(f"    Params: {sum(p.numel() for p in bb.patch_embed.parameters()) / 1e6:.2f}M")

    if hasattr(model, 'box_head'):
        print(f"\nBox head: {type(model.box_head).__name__}")
        print(f"  Params: {sum(p.numel() for p in model.box_head.parameters()) / 1e6:.2f}M")

    return model


def analyze_ostrack():
    env = env_settings()
    ostrack_update(os.path.join(env.prj_dir, 'experiments/ostrack/vitb_256_mae_ce_got10k_10ep.yaml'))
    model = build_ostrack(ostrack_cfg)

    print("\n=== OSTrack Model Structure ===")
    print(f"Total params: {count_params(model) / 1e6:.2f}M")
    print(f"Backbone type: {ostrack_cfg.MODEL.BACKBONE.TYPE}")
    print(f"Head type: {ostrack_cfg.MODEL.HEAD.TYPE}")

    # Check key components
    if hasattr(model, 'backbone'):
        bb = model.backbone
        print(f"\nBackbone components:")
        if hasattr(bb, 'blocks'):
            print(f"  blocks: {len(bb.blocks)} blocks")
            print(f"    Params: {sum(p.numel() for p in bb.blocks.parameters()) / 1e6:.2f}M")
        if hasattr(bb, 'patch_embed'):
            print(f"  patch_embed:")
            print(f"    Params: {sum(p.numel() for p in bb.patch_embed.parameters()) / 1e6:.2f}M")

    if hasattr(model, 'box_head'):
        print(f"\nBox head: {type(model.box_head).__name__}")
        print(f"  Params: {sum(p.numel() for p in model.box_head.parameters()) / 1e6:.2f}M")

    return model


def main():
    setrack = analyze_setrack()
    ostrack = analyze_ostrack()

    setrack_params = count_params(setrack) / 1e6
    ostrack_params = count_params(ostrack) / 1e6

    print(f"\n=== COMPARISON ===")
    print(f"SETrack params:  {setrack_params:.2f}M")
    print(f"OSTrack params:  {ostrack_params:.2f}M")
    print(f"Difference:      {setrack_params - ostrack_params:+.2f}M")


if __name__ == '__main__':
    main()
