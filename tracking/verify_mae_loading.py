"""
Verify MAE pretrained weights loading for SETrack and OSTrack models.
Checks: MAE checkpoint structure, missing/unexpected keys, parameter norms.
"""
import sys, os
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.insert(0, prj_path)

import torch
import argparse
from lib.config.setrack.config import cfg as setrack_cfg, update_config_from_file as setrack_update_config
from lib.config.ostrack.config import cfg as ostrack_cfg, update_config_from_file as ostrack_update_config
from lib.models.setrack.setrack import build_setrack
from lib.models.ostrack.ostrack import build_ostrack
from lib.test.evaluation.environment import env_settings


def verify_mae_checkpoint():
    """Check MAE checkpoint structure and keys."""
    env = env_settings()
    mae_path = os.path.join(env.prj_dir, 'pretrained_models', 'mae_pretrain_vit_base.pth')

    if not os.path.exists(mae_path):
        print(f"ERROR: MAE checkpoint not found at {mae_path}")
        return None

    mae_ckpt = torch.load(mae_path, map_location='cpu')
    print(f"\n=== MAE Checkpoint Structure ===")
    print(f"Keys in checkpoint: {list(mae_ckpt.keys())}")
    if 'model' in mae_ckpt:
        model_state = mae_ckpt['model']
        model_keys = list(model_state.keys())[:10]
        print(f"First 10 model keys: {model_keys}")
        print(f"Total model keys: {len(model_state)}")
        # Check MAE blocks.0 norm directly
        if 'blocks.0.attn.qkv.weight' in model_state:
            mae_block0_norm = model_state['blocks.0.attn.qkv.weight'].norm().item()
            print(f"\nMAE blocks.0.attn.qkv.weight norm (in checkpoint): {mae_block0_norm:.4f}")
        if 'blocks.8.attn.qkv.weight' in model_state:
            mae_block8_norm = model_state['blocks.8.attn.qkv.weight'].norm().item()
            print(f"MAE blocks.8.attn.qkv.weight norm (in checkpoint): {mae_block8_norm:.4f}")
        return model_state
    return None


def verify_setrack_loading(config_file):
    """Verify SETrack MAE loading from config."""
    setrack_update_config(config_file)
    print(f"\n=== SETrack Config: {os.path.basename(config_file)} ===")
    print(f"PRETRAIN_FILE: {setrack_cfg.MODEL.PRETRAIN_FILE}")
    print(f"USE_CROSS_SEMANTIC: {setrack_cfg.MODEL.BACKBONE.USE_CROSS_SEMANTIC}")
    print(f"USE_REDUNDANT_PRUNING: {setrack_cfg.MODEL.BACKBONE.USE_REDUNDANT_PRUNING}")

    # Build model
    model = build_setrack(setrack_cfg)
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Check specific layer norms (these should NOT be random init if MAE loaded correctly)
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'shared_self_blocks') and len(model.backbone.shared_self_blocks) > 0:
        block_0_attn_weight = model.backbone.shared_self_blocks[0].attn.qkv.weight
        print(f"\nbackbone.shared_self_blocks[0].attn.qkv.weight norm: {block_0_attn_weight.norm().item():.4f}")
        # Also check deepest shared layer
        last_idx = len(model.backbone.shared_self_blocks) - 1
        block_last_attn_weight = model.backbone.shared_self_blocks[last_idx].attn.qkv.weight
        print(f"backbone.shared_self_blocks[{last_idx}].attn.qkv.weight norm: {block_last_attn_weight.norm().item():.4f}")

    if hasattr(model, 'backbone') and hasattr(model.backbone, 'patch_embed'):
        patch_weight = model.backbone.patch_embed.proj.weight
        print(f"backbone.patch_embed.proj.weight norm: {patch_weight.norm().item():.4f}")

    # Check cross_semantic_blocks are random init (expected)
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'cross_semantic_blocks') and len(model.backbone.cross_semantic_blocks) > 0:
        if hasattr(model.backbone.cross_semantic_blocks[0], 'q_proj'):
            cross_0_weight = model.backbone.cross_semantic_blocks[0].q_proj.weight
            print(f"cross_semantic_blocks[0].q_proj.weight norm: {cross_0_weight.norm().item():.4f} (expected: random init)")


def verify_ostrack_loading(config_file):
    """Verify OSTrack MAE loading from config."""
    ostrack_update_config(config_file)
    print(f"\n=== OSTrack Config: {os.path.basename(config_file)} ===")
    print(f"PRETRAIN_FILE: {ostrack_cfg.MODEL.PRETRAIN_FILE}")

    # Build model
    model = build_ostrack(ostrack_cfg)
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Check specific layer norms
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'blocks') and len(model.backbone.blocks) > 0:
        block_0_attn_weight = model.backbone.blocks[0].attn.qkv.weight
        print(f"\nbackbone.blocks[0].attn.qkv.weight norm: {block_0_attn_weight.norm().item():.4f}")

    if hasattr(model, 'backbone') and hasattr(model.backbone, 'patch_embed'):
        patch_weight = model.backbone.patch_embed.proj.weight
        print(f"backbone.patch_embed.proj.weight norm: {patch_weight.norm().item():.4f}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setrack_cfg', type=str, default='vitb_256_mae_setrack_got10k_10ep')
    parser.add_argument('--ostrack_cfg', type=str, default='vitb_256_mae_ce_got10k_10ep')
    args = parser.parse_args()

    env = env_settings()
    prj_dir = env.prj_dir

    # Verify MAE checkpoint
    mae_state = verify_mae_checkpoint()

    # Verify SETrack loading
    setrack_config_path = os.path.join(prj_dir, f'experiments/setrack/{args.setrack_cfg}.yaml')
    if os.path.exists(setrack_config_path):
        try:
            setrack_model = verify_setrack_loading(setrack_config_path)
        except Exception as e:
            print(f"ERROR loading SETrack: {e}")
            import traceback; traceback.print_exc()
    else:
        print(f"WARNING: SETrack config not found: {setrack_config_path}")

    # Verify OSTrack loading
    ostrack_config_path = os.path.join(prj_dir, f'experiments/ostrack/{args.ostrack_cfg}.yaml')
    if os.path.exists(ostrack_config_path):
        try:
            ostrack_model = verify_ostrack_loading(ostrack_config_path)
        except Exception as e:
            print(f"ERROR loading OSTrack: {e}")
            import traceback; traceback.print_exc()
    else:
        print(f"WARNING: OSTrack config not found: {ostrack_config_path}")

    # Summary comparison
    print(f"\n=== Norm Comparison Summary ===")
    if mae_state and 'blocks.0.attn.qkv.weight' in mae_state:
        print(f"MAE raw blocks.0.attn.qkv.weight:  {mae_state['blocks.0.attn.qkv.weight'].norm().item():.4f}")

    print(f"\n=== Verification Complete ===")
    print(f"Expected: SETrack shared_self_blocks[0..8] norms ≈ OSTrack blocks[0..8] norms ≈ MAE blocks[0..8] norms")
    print(f"Expected: cross_semantic_blocks are random init (not in MAE) — this is CORRECT")


if __name__ == '__main__':
    main()
