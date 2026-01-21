"""
PatchTST model architecture with Student-t outputs
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class PatchEmbedding(nn.Module):
    """Patch embedding for PatchTST"""
    
    def __init__(self, patch_len: int, d_model: int, num_features: int):
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.num_features = num_features
        
        # Linear projection for patches
        self.patch_proj = nn.Linear(patch_len * num_features, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, C] where L is sequence length, C is num_features
        Returns:
            [B, num_patches, d_model]
        """
        B, L, C = x.shape
        
        # Create patches
        num_patches = L // self.patch_len
        if num_patches * self.patch_len < L:
            # Pad if needed
            pad_len = (num_patches + 1) * self.patch_len - L
            x = F.pad(x, (0, 0, 0, pad_len))
            num_patches += 1
        
        # Reshape to patches: [B, num_patches, patch_len, C]
        x = x[:, :num_patches * self.patch_len, :]
        x = x.view(B, num_patches, self.patch_len, C)
        
        # Flatten patches and project
        x = x.reshape(B, num_patches, self.patch_len * C)
        x = self.patch_proj(x)  # [B, num_patches, d_model]
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model]
        Returns:
            [B, L, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feedforward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class PatchTSTModel(nn.Module):
    """PatchTST model for time series forecasting"""
    
    def __init__(
        self,
        num_features: int,
        d_model: int = 192,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        patch_len: int = 16,
        dropout: float = 0.1,
        num_assets: int = 4,
        horizon: int = 288,  # For LF: 288 steps = 24h @ 5m
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.patch_len = patch_len
        self.horizon = horizon
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(patch_len, d_model, num_features)
        
        # Asset embedding
        self.asset_embed = nn.Embedding(num_assets, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Output heads for Student-t distribution
        self.mu_head = nn.Linear(d_model, horizon)
        self.log_sigma_head = nn.Linear(d_model, horizon)
        self.nu_head = nn.Linear(d_model, 1)  # Degrees of freedom (can be per-asset)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, x: torch.Tensor, asset_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, L, C] input features
            asset_ids: [B] asset indices
        Returns:
            mu: [B, H] mean log returns
            log_sigma: [B, H] log standard deviation
            nu: [B] or [B, 1] degrees of freedom
        """
        B, L, C = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, d_model]
        
        # Add asset embedding (broadcast to all patches)
        asset_emb = self.asset_embed(asset_ids)  # [B, d_model]
        asset_emb = asset_emb.unsqueeze(1)  # [B, 1, d_model]
        x = x + asset_emb
        
        # Positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Use last patch for prediction
        x = x[:, -1, :]  # [B, d_model]
        
        # Output heads
        mu = self.mu_head(x)  # [B, H]
        log_sigma = self.log_sigma_head(x)  # [B, H]
        nu = self.nu_head(x)  # [B, 1]
        
        # Ensure nu > 2 (minimum for Student-t variance)
        nu = F.softplus(nu) + 2.1
        
        return mu, log_sigma, nu.squeeze(-1)  # [B, H], [B, H], [B]


def load_model_checkpoint(
    checkpoint_path: str,
    num_features: int,
    num_assets: int,
    horizon: int,
    d_model: int = 192,
    num_layers: int = 6,
    device: str = "cpu"
) -> PatchTSTModel:
    """Load model from checkpoint"""
    model = PatchTSTModel(
        num_features=num_features,
        d_model=d_model,
        num_layers=num_layers,
        num_assets=num_assets,
        horizon=horizon,
    )
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        logger.info(f"Loaded model from {checkpoint_path}")
    except Exception as e:
        logger.warning(f"Could not load checkpoint {checkpoint_path}: {e}")
        logger.info("Using randomly initialized model")
    
    return model.to(device)
