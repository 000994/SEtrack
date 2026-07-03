"""
SETrack model (Phase 2 skeleton — minimal runnable version).

This module provides:
  - SETrack: the main tracking model class
  - build_setrack(): factory function to construct the model

Phase 2 behavior:
  - Reuses CenterPredictor head from OSTrack
  - Uses VisionTransformerSETrack backbone (currently similar to OSTrack CE backbone)
  - Forward interface is fully compatible with OSTrack actor/tracker

Phase 3 will add:
  - Semantic Self-Association Blocks in the backbone
  - Cross-layer Semantic Association Module
  - Redundant Information Pruning Module
  - Any necessary changes to the forward pass routing
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.setrack.vit_setrack import (vit_base_patch16_224_setrack,
                                              vit_large_patch16_224_setrack)
from lib.utils.box_ops import box_xyxy_to_cxcywh


class SETrack(nn.Module):
    """ SETrack tracking model (Phase 2 skeleton).

    This is the base class for SETrack. Currently structurally similar to OSTrack.
    The backbone will be upgraded in Phase 3 with the full SETrack architecture.
    """

    def __init__(self, backbone, box_head, aux_loss=False, head_type="CENTER"):
        """ Initializes the model.

        Parameters:
            backbone: torch module of the transformer backbone (VisionTransformerSETrack).
            box_head: torch module for bounding box prediction (CenterPredictor).
            aux_loss: True if auxiliary decoding losses are to be used.
            head_type: "CORNER" or "CENTER".
        """
        super().__init__()
        self.backbone = backbone
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        """Forward pass for SETrack.

        Args:
            template: [B, 3, H_t, W_t] template image patch
            search:   [B, 3, H_s, W_s] search region image patch
            ce_template_mask: mask for candidate elimination on template side
            ce_keep_rate: keep rate for candidate elimination
            return_last_attn: whether to return last attention weights

        Returns:
            out: dict with keys:
                'pred_boxes': [B, Nq, 4] predicted bounding boxes (cx, cy, w, h) in [0,1]
                'score_map':  [B, Nq, H, W] score map
                'size_map':   [B, Nq, 2, H, W] size map
                'offset_map': [B, Nq, 2, H, W] offset map
                'backbone_feat': backbone features
        """
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, [B, L_z+L_x, C]
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for search region
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        elif self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out
        else:
            raise NotImplementedError("Unknown head type: %s" % self.head_type)


def build_setrack(cfg, training=True):
    """Build the SETrack model.

    Args:
        cfg: EasyDict configuration object.
        training: whether in training mode.

    Returns:
        model: SETrack instance.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')

    # Determine pretrained weight path
    if cfg.MODEL.PRETRAIN_FILE and ('SETrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        if not os.path.exists(pretrained):
            print(f"SETrack: Pretrained weight not found at {pretrained}, using random init.")
            pretrained = ''
    else:
        pretrained = ''

    # Build backbone based on config type
    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        # OSTrack-compatible basic ViT (no CE)
        from lib.models.ostrack.vit import vit_base_patch16_224
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_setrack':
        # SETrack backbone with cross-layer semantic + redundant pruning
        backbone = vit_base_patch16_224_setrack(
            pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
            use_cross_semantic=getattr(cfg.MODEL.BACKBONE, 'USE_CROSS_SEMANTIC', True),
            cross_semantic_block_type=getattr(cfg.MODEL.BACKBONE, 'CROSS_SEMANTIC_BLOCK_TYPE', 'light'),
            use_redundant_pruning=getattr(cfg.MODEL.BACKBONE, 'USE_REDUNDANT_PRUNING', True),
            pruning_center_ratio=getattr(cfg.MODEL.BACKBONE, 'PRUNING_CENTER_RATIO', 0.5),
            pruning_energy_ratio=getattr(cfg.MODEL.BACKBONE, 'PRUNING_ENERGY_RATIO', 0.7),
            pruning_min_keep_ratio=getattr(cfg.MODEL.BACKBONE, 'PRUNING_MIN_KEEP_RATIO', 0.5),
            pruning_sim_aggregation=getattr(cfg.MODEL.BACKBONE, 'PRUNING_SIM_AGGREGATION', 'mean'),
            pruning_fill_value=getattr(cfg.MODEL.BACKBONE, 'PRUNING_FILL_VALUE', 'zero'),
            pruning_soft_scale=getattr(cfg.MODEL.BACKBONE, 'PRUNING_SOFT_SCALE', 0.1),
        )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_setrack':
        backbone = vit_large_patch16_224_setrack(
            pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
            use_cross_semantic=getattr(cfg.MODEL.BACKBONE, 'USE_CROSS_SEMANTIC', True),
            use_redundant_pruning=getattr(cfg.MODEL.BACKBONE, 'USE_REDUNDANT_PRUNING', True),
        )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        # Allow using the OSTrack CE backbone via SETrack builder
        from lib.models.ostrack.vit_ce import vit_base_patch16_224_ce
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                             ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                             ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                             )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError("Unsupported backbone type: %s" % cfg.MODEL.BACKBONE.TYPE)

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = SETrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    # Load SETrack-specific checkpoint if specified
    if 'SETrack' in cfg.MODEL.PRETRAIN_FILE and training:
        if os.path.exists(cfg.MODEL.PRETRAIN_FILE):
            checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
