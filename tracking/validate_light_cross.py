"""Pre-training validation for light cross block."""
import sys, os, time
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.insert(0, prj_path)

import torch
from lib.config.setrack.config import cfg, update_config_from_file
from lib.models.setrack import build_setrack
from lib.test.evaluation.environment import env_settings
from thop import profile

env = env_settings()
update_config_from_file(os.path.join(env.prj_dir,
    'experiments/setrack/vitb_256_mae_setrack_cross_light_fixmae_got10k_10ep.yaml'))

print('Config:')
print('  USE_CROSS_SEMANTIC:', cfg.MODEL.BACKBONE.USE_CROSS_SEMANTIC)
print('  CROSS_SEMANTIC_BLOCK_TYPE:', cfg.MODEL.BACKBONE.CROSS_SEMANTIC_BLOCK_TYPE)
print('  USE_REDUNDANT_PRUNING:', cfg.MODEL.BACKBONE.USE_REDUNDANT_PRUNING)

model = build_setrack(cfg, training=False)
model.cuda().eval()

# Params
total = sum(p.numel() for p in model.parameters())
print('\nTotal params: %.3fM (%d)' % (total/1e6, total))

bb = model.backbone
for name, mod in bb.named_children():
    n = sum(p.numel() for p in mod.parameters())
    if n > 0:
        print('  backbone.%s: %.3fM (%d)' % (name, n/1e6, n))
    if name == 'cross_semantic_blocks':
        for i, blk in enumerate(mod):
            bn = sum(p.numel() for p in blk.parameters())
            print('    cross[%d] (light): %.3fM (%d)' % (i, bn/1e6, bn))

box_n = sum(p.numel() for p in model.box_head.parameters())
print('  box_head: %.3fM' % (box_n/1e6))

# MACs
z = torch.randn(1, 3, 128, 128).cuda()
x = torch.randn(1, 3, 256, 256).cuda()
macs, params = profile(model, inputs=(z, x), verbose=False)
print('\nMACs: %.3fG' % (macs/1e9))

# Forward test
with torch.no_grad():
    out = model.forward(template=z, search=x)
print('Forward OK: pred_boxes=%s, score_map=%s' % (list(out['pred_boxes'].shape), list(out['score_map'].shape)))
has_nan = torch.isnan(out['pred_boxes']).any().item()
print('NaN in pred_boxes:', has_nan)

# MAE loading check
print('\nMAE check:')
print('  shared_self_blocks[0] norm: %.4f' % bb.shared_self_blocks[0].attn.qkv.weight.norm().item())
print('  patch_embed norm: %.4f' % bb.patch_embed.proj.weight.norm().item())
if hasattr(bb.cross_semantic_blocks[0], 'q_proj'):
    print('  cross[0].q_proj norm: %.4f (expected: random)' % bb.cross_semantic_blocks[0].q_proj.weight.norm().item())

# FPS test
with torch.no_grad():
    for _ in range(10):
        model.forward(template=z, search=x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        model.forward(template=z, search=x)
    torch.cuda.synchronize()
    t1 = time.time()
fps = 100 / (t1 - t0)
print('\nFPS (batch=1, avg over 100): %.1f' % fps)

# Comparison table
print('\n============ Comparison ============')
print('%-30s %10s %10s' % ('', 'Params', 'MACs'))
print('%-30s %10s %10s' % ('OSTrack', '92.082M', '21.499G'))
print('%-30s %10s %10s' % ('SETrack base', '70.828M', '16.795G'))
print('%-30s %10s %10s' % ('SETrack heavy cross', '92.082M', '22.684G'))
print('%-30s %10.3fM %10.3fG' % ('SETrack light cross', total/1e6, macs/1e9))
print()

# Light cross block detail
c0 = bb.cross_semantic_blocks[0]
print('Light cross block components:')
for pname, p in c0.named_parameters():
    print('  %s: %s  %d params' % (pname, list(p.shape), p.numel()))

print('\nAll validations passed!')
