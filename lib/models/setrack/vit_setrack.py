"""
SETrack Vision Transformer Backbone (Phase 6 — redundant information pruning).

Architecture:
  - shared_self_blocks: 9× SemanticSelfAssociationBlock (shared between branches)
    * Template branch uses all 9 (indices 0-8)
    * Search front branch reuses first 6 (indices 0-5)
    * Template features from layers 6,7,8 are cached
  - cross_semantic_blocks: 3× CrossLayerSemanticAssociationBlock
    * MHCA: search (Q) attends to [search + current_t + deep_t] (K/V)
    * Configurable: USE_CROSS_SEMANTIC=True/False
  - redundant_pruning: RedundantInformationPruning (Phase 6)
    * Energy-based adaptive pruning via search↔template-center cosine similarity
    * Prune-and-restore: keeps 256-token grid for CenterPredictor compat
    * Configurable: USE_REDUNDANT_PRUNING=True/False
  - Candidate Elimination: DISABLED
"""
import math
import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple, trunc_normal_

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.layers.setrack_blocks import (
    SemanticSelfAssociationBlock,
    NoParamCrossSemanticPlaceholder,
    CrossLayerSemanticAssociationBlock,
    LightCrossLayerSemanticAssociationBlock,
    RedundantPruningPlaceholder,
    RedundantInformationPruning,
)
from lib.models.ostrack.vit import resize_pos_embed, _init_vit_weights

_logger = logging.getLogger(__name__)


class VisionTransformerSETrack(nn.Module):
    """ SETrack backbone with redundant information pruning (Phase 6).

    Phase 6 status:
      - shared_self_blocks: 9× SemanticSelfAssociationBlock (shared)  ✅
      - Cross-semantic: 3× CrossLayerSemanticAssociationBlock (MHCA)  ✅
      - Redundant pruning: RedundantInformationPruning (energy-based)  ✅
      - Candidate Elimination: DISABLED                                ✅
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None, weight_init='',
                 # SETrack branch depths
                 template_branch_depth=9,
                 search_self_depth=6,
                 cross_semantic_depth=3,
                 # Whether to share self-attention blocks
                 share_self_blocks=True,
                 # Cross-semantic config
                 use_cross_semantic=True,
                 cross_placeholder_type="no_param",
                 cross_semantic_block_type="heavy",  # "heavy" or "light"
                 # Redundant pruning config (Phase 6)
                 use_redundant_pruning=True,
                 pruning_center_ratio=0.5,
                 pruning_energy_ratio=0.7,
                 pruning_min_keep_ratio=0.5,
                 pruning_sim_aggregation="mean",
                 pruning_fill_value="zero",
                 pruning_soft_scale=0.1,
                 # Kept for config compatibility, IGNORED
                 ce_loc=None, ce_keep_ratio=None):
        super().__init__()

        # ---- Basic ViT attributes ----
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size

        # ---- Patch embedding (shared) ----
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # ---- Class / Distillation tokens (kept for pretrained weight compat) ----
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # ---- Tracking-specific attributes (set in finetune_track) ----
        self.cat_mode = 'direct'
        self.pos_embed_z = None
        self.pos_embed_x = None
        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None
        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]
        self.add_cls_token = False
        self.add_sep_seg = False
        self.cls_pos_embed = None

        self.template_branch_depth = template_branch_depth
        self.search_self_depth = search_self_depth
        self.cross_semantic_depth = cross_semantic_depth
        self.share_self_blocks = share_self_blocks

        # ---- Stochastic depth (only for the 9 shared blocks) ----
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, template_branch_depth)]

        # ================================================================
        #  shared_self_blocks: 9× SemanticSelfAssociationBlock
        #  SHARED between template and search branches.
        #
        #  Template branch: uses shared_self_blocks[0:9] (all 9)
        #  Search front:    uses shared_self_blocks[0:6] (first 6, SAME weights)
        #
        #  This is the KEY difference from Phase 3's 18 independent blocks.
        # ================================================================
        self.shared_self_blocks = nn.ModuleList([
            SemanticSelfAssociationBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer)
            for i in range(template_branch_depth)
        ])
        # Indices of layers whose output features are cached for cross-layer association
        self.template_cache_indices = [6, 7, 8]  # last 3 of 9 layers (0-indexed)

        # ================================================================
        #  cross_semantic_blocks: 3× CrossLayerSemanticAssociationBlock
        #  Phase 5: Multi-Head Cross-Attention with cached template features.
        #  When USE_CROSS_SEMANTIC=False, falls back to 0-param placeholder.
        # ================================================================
        self.use_cross_semantic = use_cross_semantic
        self.cross_semantic_depth = cross_semantic_depth
        self.cross_semantic_block_type = cross_semantic_block_type
        if use_cross_semantic:
            # Real cross-attention modules (heavy or light)
            if cross_semantic_block_type == "light":
                CrossBlockCls = LightCrossLayerSemanticAssociationBlock
            else:
                CrossBlockCls = CrossLayerSemanticAssociationBlock  # "heavy" (default)
            self.cross_semantic_blocks = nn.ModuleList([
                CrossBlockCls(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=0.,  # cross blocks share template's effective drop_path
                    act_layer=act_layer, norm_layer=norm_layer)
                for _ in range(cross_semantic_depth)
            ])
        else:
            # Ablation mode: 0-param placeholder
            self.cross_semantic_blocks = nn.ModuleList([
                NoParamCrossSemanticPlaceholder() for _ in range(cross_semantic_depth)
            ])

        # ================================================================
        #  Redundant Pruning: Energy-based adaptive (Phase 6) or Placeholder
        #  Prune-and-restore: keeps 256-token grid for CenterPredictor compat.
        #  Template tokens from deepest layer are used for similarity scoring.
        # ================================================================
        self.use_redundant_pruning = use_redundant_pruning
        if use_redundant_pruning:
            self.redundant_pruning = RedundantInformationPruning(
                template_size=8,      # 128 / 16
                search_size=16,       # 256 / 16
                dim=embed_dim,
                center_ratio=pruning_center_ratio,
                energy_ratio=pruning_energy_ratio,
                min_keep_ratio=pruning_min_keep_ratio,
                sim_aggregation=pruning_sim_aggregation,
                fill_value_mode=pruning_fill_value,
                soft_scale=pruning_soft_scale,
            )
        else:
            self.redundant_pruning = RedundantPruningPlaceholder()

        # ---- Cached template features (populated during forward) ----
        self.cached_template_features = []

        # ---- Final normalization ----
        self.norm = norm_layer(embed_dim)

        # ---- Weight initialization ----
        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        """Initialize weights, matching ViT convention."""
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            from timm.models.helpers import named_apply
            named_apply(partial(_init_vit_weights, head_bias=0., jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        """Load pretrained weights from a checkpoint file."""
        from lib.models.ostrack.vit import _load_weights
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def finetune_track(self, cfg, patch_start_index=1):
        """
        Prepare backbone for tracking: resize position embeddings for
        template/search sizes.
        """
        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER
        self.return_stage = cfg.MODEL.RETURN_STAGES
        self.add_sep_seg = cfg.MODEL.BACKBONE.SEP_SEG
        self.add_cls_token = cfg.MODEL.BACKBONE.ADD_CLS_TOKEN

        # Resize patch embedding if patch size differs from pretrained
        if new_patch_size != self.patch_size:
            print('Inconsistent Patch Size With The Pretrained Weights, Interpolate The Weight!')
            old_patch_embed = {}
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size,
                                          in_chans=3, embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']
        self.patch_size = new_patch_size

        # Resize position embeddings for search and template
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # Search region position embedding
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic', align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

        # Template region position embedding
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic', align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)

        # CLS token position embedding
        if self.add_cls_token and patch_start_index > 0:
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            self.cls_pos_embed = nn.Parameter(cls_pos_embed)

        # Separate segment embeddings
        if self.add_sep_seg:
            self.template_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.template_segment_pos_embed = trunc_normal_(self.template_segment_pos_embed, std=.02)
            self.search_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.search_segment_pos_embed = trunc_normal_(self.search_segment_pos_embed, std=.02)

        if self.return_inter:
            for i_layer in self.return_stage:
                if i_layer != 11:
                    _norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = _norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

    def forward_features(self, z, x,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False):
        """
        Forward pass through the shared-block SETrack backbone.

        Template branch: shared_self_blocks[0:9]
        Search front:    shared_self_blocks[0:6] (SAME weights as template front)
        Search cross:    cross_semantic_blocks[0:3] (0-param placeholders)

        Args:
            z: template image, [B, 3, H_t, W_t]
            x: search region image, [B, 3, H_s, W_s]
            ce_template_mask: IGNORED (no CE)
            ce_keep_rate: IGNORED (no CE)
            return_last_attn: IGNORED

        Returns:
            merged: [B, L_t+L_s, C]
            aux_dict: dict with 'attn', 'removed_indexes_s', 'cached_template_features'
        """
        B = x.shape[0]

        # ---- 1. Patch embedding ----
        z_tokens = self.patch_embed(z)  # [B, 64, 768]
        x_tokens = self.patch_embed(x)  # [B, 256, 768]

        # ---- 2. Add position embeddings ----
        z_tokens = z_tokens + self.pos_embed_z
        x_tokens = x_tokens + self.pos_embed_x

        if self.add_sep_seg:
            x_tokens = x_tokens + self.search_segment_pos_embed
            z_tokens = z_tokens + self.template_segment_pos_embed

        z_tokens = self.pos_drop(z_tokens)
        x_tokens = self.pos_drop(x_tokens)

        # ---- 3. Template branch: shared_self_blocks[0:9] ----
        self.cached_template_features = []
        for i in range(self.template_branch_depth):
            z_tokens = self.shared_self_blocks[i](z_tokens)
            if i in self.template_cache_indices:
                self.cached_template_features.append(z_tokens)

        # ---- 4. Search front: shared_self_blocks[0:6] (SAME weights as template) ----
        for i in range(self.search_self_depth):
            x_tokens = self.shared_self_blocks[i](x_tokens)

        # ---- 5. Cross-layer Semantic Association: 3× MHCA blocks ----
        # Each block does cross-attention: search(Q) x [search+current_t+deep_t](K,V)
        for i, blk in enumerate(self.cross_semantic_blocks):
            current_t = self.cached_template_features[i] if i < len(self.cached_template_features) else None
            deep_t = self.cached_template_features[-1] if self.cached_template_features else None
            x_tokens = blk(x_tokens,
                           current_template_tokens=current_t,
                           deep_template_tokens=deep_t)

        # ---- 6. Redundant pruning (energy-based adaptive, prune-and-restore) ----
        # Uses z_tokens (final template output) for similarity scoring.
        if self.use_redundant_pruning:
            x_tokens, removed_indexes_s, keep_indexes_s, pruning_info = \
                self.redundant_pruning(search_tokens=x_tokens,
                                       template_tokens=z_tokens,
                                       return_indices=True)
            # Add fill metadata to pruning_info for debugging
            pruning_info['fill_mode'] = self.redundant_pruning.fill_value_mode
            pruning_info['soft_scale'] = getattr(self.redundant_pruning, 'soft_scale', 0.0)
        else:
            removed_indexes_s = None
            keep_indexes_s = None
            pruning_info = {}

        # ---- 7. Final normalization ----
        z_tokens = self.norm(z_tokens)
        x_tokens = self.norm(x_tokens)

        # ---- 8. Concatenate for head compatibility ----
        merged = torch.cat([z_tokens, x_tokens], dim=1)  # [B, 320, 768]

        aux_dict = {
            "attn": None,
            "removed_indexes_s": [] if removed_indexes_s is None else removed_indexes_s,
            "keep_indexes_s": keep_indexes_s,
            "pruning_info": pruning_info,
            "cached_template_features": self.cached_template_features,
        }

        return merged, aux_dict

    def forward(self, z, x,
                ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):
        return self.forward_features(z, x,
                                     ce_template_mask=ce_template_mask,
                                     ce_keep_rate=ce_keep_rate,
                                     return_last_attn=return_last_attn)


# ---- Factory functions ----

import os


def _create_vision_transformer_setrack(pretrained=False, **kwargs):
    """Create a VisionTransformerSETrack instance."""
    model = VisionTransformerSETrack(**kwargs)

    if pretrained:
        if isinstance(pretrained, str) and len(pretrained) > 0:
            if 'npz' in pretrained:
                model.load_pretrained(pretrained, prefix='')
            elif pretrained.endswith('.pth'):
                if os.path.exists(pretrained):
                    checkpoint = torch.load(pretrained, map_location="cpu")
                    state_dict = checkpoint["model"]

                    # ---- Remap MAE blocks.X keys to SETrack shared_self_blocks.X ----
                    # MAE ViT-Base has 12 blocks (blocks.0..blocks.11).
                    # SETrack has 9 shared_self_blocks (indices 0..8, shared between branches).
                    # Also map: blocks.X.* → shared_self_blocks.X.* for X in 0..8.
                    remapped = {}
                    shared_depth = kwargs.get('template_branch_depth', 9)
                    blocks_mapped = 0
                    blocks_skipped = 0
                    for k, v in state_dict.items():
                        if k.startswith('blocks.'):
                            # Parse block index: blocks.X.xxx → X
                            rest = k[len('blocks.'):]
                            dot_pos = rest.find('.')
                            if dot_pos > 0:
                                blk_idx_str = rest[:dot_pos]
                                blk_suffix = rest[dot_pos:]
                                try:
                                    blk_idx = int(blk_idx_str)
                                except ValueError:
                                    remapped[k] = v
                                    continue
                                if blk_idx < shared_depth:
                                    new_key = f'shared_self_blocks.{blk_idx}{blk_suffix}'
                                    remapped[new_key] = v
                                    blocks_mapped += 1
                                else:
                                    blocks_skipped += 1
                            else:
                                remapped[k] = v
                        else:
                            remapped[k] = v

                    missing_keys, unexpected_keys = model.load_state_dict(remapped, strict=False)
                    print('Load pretrained model from: ' + pretrained)
                    print(f'SETrack MAE remap: mapped {blocks_mapped} blocks→shared_self_blocks, '
                          f'skipped {blocks_skipped} blocks beyond depth {shared_depth}')
                    if missing_keys:
                        print(f'  Missing keys (first 10): {missing_keys[:10]}')
                    if unexpected_keys:
                        print(f'  Unexpected keys (first 10): {unexpected_keys[:10]}')
                else:
                    print(f'SETrack: Pretrained file not found ({pretrained}), using random init.')
            else:
                print('Warning: unrecognized pretrained format, using random init.')

    return model


def vit_base_patch16_224_setrack(pretrained=False, **kwargs):
    """ ViT-Base SETrack backbone (Phase 6 — redundant information pruning).

    Creates a VisionTransformerSETrack with:
      - 9 shared SemanticSelfAssociationBlocks
      - 3 CrossLayerSemanticAssociationBlocks (MHCA)
      - 1 RedundantInformationPruning (energy-based adaptive)
      - USE_CROSS_SEMANTIC=True, USE_REDUNDANT_PRUNING=True by default.
    """
    # Defaults (kwargs from caller override these)
    model_kwargs = dict(
        patch_size=16, embed_dim=768, num_heads=12,
        template_branch_depth=9,
        search_self_depth=6,
        cross_semantic_depth=3,
        share_self_blocks=True,
        cross_placeholder_type="no_param",
    )
    model_kwargs.update(kwargs)
    model = _create_vision_transformer_setrack(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_setrack(pretrained=False, **kwargs):
    """ ViT-Large SETrack backbone (reserved for future use). """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, num_heads=16,
        template_branch_depth=9,
        search_self_depth=6,
        cross_semantic_depth=3,
        share_self_blocks=True,
        cross_placeholder_type="no_param",
        **kwargs)
    model = _create_vision_transformer_setrack(pretrained=pretrained, **model_kwargs)
    return model
