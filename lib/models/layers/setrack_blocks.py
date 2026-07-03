"""
SETrack-specific building blocks (Phase 6 — redundant information pruning).

This module provides:
  - SemanticSelfAssociationBlock: standard self-attention transformer block
    (no candidate elimination). SHARED between template and search branches.
  - NoParamCrossSemanticPlaceholder: 0-param placeholder (for ablation: USE_CROSS_SEMANTIC=False).
  - CrossLayerSemanticAssociationBlock: cross-attention module that lets search
    tokens attend to cached template features from multiple layers (Phase 5).
  - RedundantPruningPlaceholder: placeholder for redundant information pruning (Identity).
  - RedundantInformationPruning: energy-based adaptive pruning using cosine similarity
    between search tokens and template center region tokens (Phase 6).
"""
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath

from lib.models.layers.attn import Attention


class SemanticSelfAssociationBlock(nn.Module):
    """Standard transformer self-attention block.

    Plain ViT encoder block WITHOUT candidate elimination.
    SHARED between template-branch and search-branch self-attention layers.
    Architecture: LN -> Attention -> residual -> LN -> MLP -> residual
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, mask=mask)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class NoParamCrossSemanticPlaceholder(nn.Module):
    """0-parameter placeholder for cross-layer semantic association.

    Used when USE_CROSS_SEMANTIC=False (Phase 4 / ablation mode).
    Pure Identity passthrough — search tokens pass through unchanged.
    """

    def __init__(self):
        super().__init__()

    def forward(self, search_tokens, current_template_tokens=None, deep_template_tokens=None):
        return search_tokens


class CrossLayerSemanticAssociationBlock(nn.Module):
    """Cross-layer semantic association via Multi-Head Cross-Attention (Phase 5).

    Lets search region tokens attend to cached template features from
    both the current template layer and the deepest template layer,
    enabling cross-layer semantic alignment.

    Architecture:
      1. LN_q(search)  -> Q_proj -> Q
      2. LN_kv(cat[search, cur_t, deep_t]) -> K_proj -> K, V_proj -> V
      3. Cross-Attention: softmax(QK^T / sqrt(d)) * V
      4. Output projection + residual
      5. LN -> MLP -> residual

    Input shapes:
      search_tokens:          [B, N_s, C]   (e.g. [B, 256, 768])
      current_template_tokens: [B, N_t, C]   (e.g. [B, 64, 768])
      deep_template_tokens:    [B, N_t, C]   (e.g. [B, 64, 768])

    Output shape: [B, N_s, C] (same as search_tokens input)
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # ---- Query projection (from search tokens only) ----
        self.norm_q = norm_layer(dim)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # ---- Key/Value projection (from [search + current_t + deep_t]) ----
        self.norm_kv = norm_layer(dim)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # ---- Output projection ----
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.attn_drop = nn.Dropout(attn_drop)

        # ---- Drop path ----
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # ---- MLP ----
        self.norm_mlp = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, search_tokens, current_template_tokens=None, deep_template_tokens=None,
                return_attn=False):
        """
        Args:
            search_tokens: [B, N_s, C] search region tokens
            current_template_tokens: [B, N_t, C] template from current cross-layer
            deep_template_tokens: [B, N_t, C] template from deepest cached layer
            return_attn: if True, also return attention weights

        Returns:
            search_tokens: [B, N_s, C] (same shape as input)
            attn: (optional) [B, num_heads, N_s, N_s+2*N_t]
        """
        B, N_s, C = search_tokens.shape

        # ---- Build Key/Value tokens ----
        # Concatenate [search_tokens; current_template_tokens; deep_template_tokens]
        kv_parts = [search_tokens]
        if current_template_tokens is not None:
            kv_parts.append(current_template_tokens)
        if deep_template_tokens is not None:
            kv_parts.append(deep_template_tokens)
        kv_tokens = torch.cat(kv_parts, dim=1)  # [B, N_s + N_t + N_t, C]
        N_kv = kv_tokens.shape[1]

        # ---- Project Q, K, V ----
        q = self.q_proj(self.norm_q(search_tokens))   # [B, N_s, C]
        k = self.k_proj(self.norm_kv(kv_tokens))       # [B, N_kv, C]
        v = self.v_proj(self.norm_kv(kv_tokens))       # [B, N_kv, C]

        # ---- Reshape for multi-head attention ----
        q = q.reshape(B, N_s, self.num_heads, self.head_dim).permute(0, 2, 1, 3)   # [B, h, N_s, d]
        k = k.reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, N_kv, d]
        v = v.reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, N_kv, d]

        # ---- Scaled dot-product attention ----
        attn = (q @ k.transpose(-2, -1)) * self.scale    # [B, h, N_s, N_kv]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v                                       # [B, h, N_s, d]
        x = x.transpose(1, 2).reshape(B, N_s, C)          # [B, N_s, C]

        # ---- Output projection + residual ----
        x = self.proj(x)
        x = self.proj_drop(x)
        search_tokens = search_tokens + self.drop_path(x)

        # ---- MLP + residual ----
        search_tokens = search_tokens + self.drop_path(self.mlp(self.norm_mlp(search_tokens)))

        if return_attn:
            return search_tokens, attn
        return search_tokens


class LightCrossLayerSemanticAssociationBlock(nn.Module):
    """Lightweight cross-layer semantic association (Phase 14).

    Like CrossLayerSemanticAssociationBlock but WITHOUT the MLP stage.
    Only cross-attention + residual — no MLP, no norm_mlp.

    Architecture:
      1. LN_q(search)  -> Q_proj -> Q
      2. LN_kv(cat[search, cur_t, deep_t]) -> K_proj -> K, V_proj -> V
      3. Cross-Attention: softmax(QK^T / sqrt(d)) * V
      4. Output projection + residual
      (NO MLP stage)

    Parameter savings vs heavy: ~4.72M per block (MLP fc1+fc2 + norm_mlp).

    Input/Output shapes are identical to CrossLayerSemanticAssociationBlock.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # ---- Query projection (from search tokens only) ----
        self.norm_q = norm_layer(dim)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # ---- Key/Value projection (from [search + current_t + deep_t]) ----
        self.norm_kv = norm_layer(dim)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # ---- Output projection ----
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.attn_drop = nn.Dropout(attn_drop)

        # ---- Drop path ----
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # NOTE: NO MLP — this is the key difference from the heavy version.

    def forward(self, search_tokens, current_template_tokens=None, deep_template_tokens=None,
                return_attn=False):
        B, N_s, C = search_tokens.shape

        # ---- Build Key/Value tokens ----
        kv_parts = [search_tokens]
        if current_template_tokens is not None:
            kv_parts.append(current_template_tokens)
        if deep_template_tokens is not None:
            kv_parts.append(deep_template_tokens)
        kv_tokens = torch.cat(kv_parts, dim=1)  # [B, N_s + N_t + N_t, C]
        N_kv = kv_tokens.shape[1]

        # ---- Project Q, K, V ----
        q = self.q_proj(self.norm_q(search_tokens))
        k = self.k_proj(self.norm_kv(kv_tokens))
        v = self.v_proj(self.norm_kv(kv_tokens))

        # ---- Multi-head attention ----
        q = q.reshape(B, N_s, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N_s, C)

        # ---- Output projection + residual ----
        x = self.proj(x)
        x = self.proj_drop(x)
        search_tokens = search_tokens + self.drop_path(x)

        # NOTE: NO MLP stage — output goes directly to next module.

        if return_attn:
            return search_tokens, attn
        return search_tokens


class RedundantInformationPruning(nn.Module):
    """Redundant Information Pruning via energy-based adaptive token selection.

    Phase 6: Replaces the placeholder with actual token pruning logic.

    Algorithm:
      1. Extract template center region tokens (see get_template_center_tokens).
      2. Compute cosine similarity between every search token and center template tokens.
      3. Aggregate -> per-search-token importance score (higher = more relevant).
      4. Energy-based adaptive keep/remove decision (sorted cumulative energy threshold).
      5. Restore pruned tokens to full grid with zero-fill (compatible with CenterPredictor).

    This is NOT OSTrack's Candidate Elimination (CE). CE uses template→search attention
    weights to remove background. RIM uses search↔template-center similarity to remove
    redundant background tokens — a different pruning criterion and mechanism.
    """

    def __init__(self, template_size=8, search_size=16, dim=768,
                 center_ratio=0.5, energy_ratio=0.7, min_keep_ratio=0.5,
                 sim_aggregation="mean", fill_value_mode="zero", soft_scale=0.1, eps=1e-6):
        """
        Args:
            template_size: int, template grid size (128/16=8 for patch_size=16)
            search_size: int, search grid size (256/16=16)
            dim: int, feature dimension (unused in this parameter-free module,
                 reserved for future learnable scoring)
            center_ratio: float, ratio of template center region (0.5 → 4×4 from 8×8)
            energy_ratio: float, energy retention ratio for adaptive pruning
            min_keep_ratio: float, minimum fraction of tokens to keep
            sim_aggregation: str, "mean" or "max" aggregation over center template dim
            fill_value_mode: str, "zero" (fill with 0), "original" (keep original),
                             or "soft" (multiply removed tokens by soft_scale)
            soft_scale: float, scale factor for soft fill mode (default 0.1)
            eps: float, numerical stability
        """
        super().__init__()
        self.template_size = template_size
        self.search_size = search_size
        self.dim = dim
        self.center_ratio = center_ratio
        self.energy_ratio = energy_ratio
        self.min_keep_ratio = min_keep_ratio
        self.sim_aggregation = sim_aggregation
        self.fill_value_mode = fill_value_mode
        self.soft_scale = soft_scale
        self.eps = eps

        # Compute center indices once
        self.register_buffer('center_indices', self._compute_center_indices())

    def _compute_center_indices(self):
        """Compute flattened indices of the template center region.

        Example: H=W=8, center_ratio=0.5 → center region = [2:6, 2:6] → 4×4=16 tokens.
        """
        H, W = self.template_size, self.template_size
        c_h = int(H * self.center_ratio)
        c_w = int(W * self.center_ratio)
        start_h = (H - c_h) // 2
        start_w = (W - c_w) // 2
        rows = torch.arange(start_h, start_h + c_h)
        cols = torch.arange(start_w, start_w + c_w)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')
        indices = grid_y.flatten() * W + grid_x.flatten()
        return indices  # [N_c]

    def get_template_center_tokens(self, template_tokens):
        """Extract center region tokens from template feature grid.

        Args:
            template_tokens: [B, N_t, C] where N_t = H_t * W_t (e.g. [B, 64, 768])
        Returns:
            center_tokens: [B, N_c, C] (e.g. [B, 16, 768] for 4×4 center)
        """
        B, N_t, C = template_tokens.shape
        idx = self.center_indices.to(template_tokens.device)  # [N_c]
        idx = idx.unsqueeze(0).expand(B, -1)                  # [B, N_c]
        center_tokens = template_tokens.gather(dim=1, index=idx.unsqueeze(-1).expand(-1, -1, C))
        return center_tokens

    def forward(self, search_tokens, template_tokens, return_indices=False):
        """
        Args:
            search_tokens: [B, N_s, C] search region feature tokens
            template_tokens: [B, N_t, C] template feature tokens (from final layer)
            return_indices: if True, also return keep/remove indices and info

        Returns:
            restored_tokens: [B, N_s, C] tokens with pruned positions zero-filled
            removed_indexes_s: [B, N_removed] indices of removed tokens (per sample)
            keep_indexes_s: [B, N_keep] indices of kept tokens (per sample)
            pruning_info: dict with per-sample statistics
        """
        B, N_s, C = search_tokens.shape
        device = search_tokens.device

        # ---- 1. Extract template center tokens ----
        center_tokens = self.get_template_center_tokens(template_tokens)  # [B, N_c, C]

        # ---- 2. L2 normalize ----
        search_norm = torch.nn.functional.normalize(search_tokens, dim=-1)   # [B, N_s, C]
        center_norm = torch.nn.functional.normalize(center_tokens, dim=-1)   # [B, N_c, C]

        # ---- 3. Cosine similarity: [B, N_s, N_c] ----
        sim = torch.bmm(search_norm, center_norm.transpose(1, 2))  # [B, N_s, N_c]

        # ---- 4. Aggregate over center dimension → importance score [B, N_s] ----
        if self.sim_aggregation == "mean":
            scores = sim.mean(dim=-1)
        elif self.sim_aggregation == "max":
            scores = sim.max(dim=-1)[0]
        else:
            raise ValueError("Unknown sim_aggregation: %s" % self.sim_aggregation)

        # ---- 5. Energy-based adaptive pruning (per sample) ----
        min_keep = int(N_s * self.min_keep_ratio)
        keep_list, remove_list = [], []
        keep_nums = []

        for b in range(B):
            score = scores[b]  # [N_s]
            score_shift = score - score.min() + self.eps
            sorted_score, sorted_idx = torch.sort(score_shift, descending=True)
            total_energy = sorted_score.sum()
            cum_energy = torch.cumsum(sorted_score, dim=0)
            # Find first index where cumulative energy >= energy_ratio * total
            keep_mask = cum_energy >= self.energy_ratio * total_energy
            keep_idx_energy = torch.where(keep_mask)[0]
            if len(keep_idx_energy) > 0:
                keep_num_energy = keep_idx_energy[0].item() + 1
            else:
                keep_num_energy = N_s
            keep_num = max(keep_num_energy, min_keep)
            keep_nums.append(keep_num)
            keep_list.append(sorted_idx[:keep_num])
            remove_list.append(sorted_idx[keep_num:])

        # ---- 6. Build unified-size indices for batched scatter ----
        # Use max keep_num across batch so all samples have same restored shape
        max_keep = max(keep_nums)
        keep_indexes_s = torch.zeros(B, max_keep, dtype=torch.long, device=device)
        remove_indexes_s_list = []
        for b in range(B):
            keep_indexes_s[b, :keep_nums[b]] = keep_list[b]
            if len(remove_list[b]) > 0:
                remove_indexes_s_list.append(remove_list[b])

        # ---- 7. Restore to full grid based on fill strategy ----
        if self.fill_value_mode == "zero":
            # Zero-fill: pruned positions = 0 (original behavior)
            restored_tokens = torch.zeros_like(search_tokens)
            for b in range(B):
                k_idx = keep_list[b]
                restored_tokens[b, k_idx] = search_tokens[b, k_idx]
        elif self.fill_value_mode == "original":
            # Original-fill: keep all tokens, no modification
            # RIM only computes keep/remove stats, does NOT alter features
            restored_tokens = search_tokens.clone()
        elif self.fill_value_mode == "soft":
            # Soft-fill: removed tokens are attenuated by soft_scale
            # Keep tokens unchanged, removed tokens *= soft_scale
            restored_tokens = search_tokens.clone()
            for b in range(B):
                r_idx = remove_list[b]
                if len(r_idx) > 0:
                    restored_tokens[b, r_idx] = search_tokens[b, r_idx] * self.soft_scale
        else:
            raise ValueError("Unknown fill_value_mode: %s (use 'zero', 'original', or 'soft')"
                             % self.fill_value_mode)

        # ---- 8. Build pruning info ----
        pruning_info = {
            "keep_num": keep_nums,
            "removed_num": [N_s - k for k in keep_nums],
            "energy_ratio": self.energy_ratio,
            "center_token_num": center_tokens.shape[1],
            "scores": scores,
        }

        if return_indices:
            return restored_tokens, remove_indexes_s_list, keep_indexes_s, pruning_info
        return restored_tokens, remove_indexes_s_list


class RedundantPruningPlaceholder(nn.Module):
    """Placeholder for redundant information pruning (Identity, 0 params).

    Used when USE_REDUNDANT_PRUNING=False (Phase 5 / ablation mode).
    """

    def __init__(self):
        super().__init__()

    def forward(self, search_tokens, template_tokens=None, return_indices=False):
        if return_indices:
            return search_tokens, [], None, {}
        return search_tokens, None
