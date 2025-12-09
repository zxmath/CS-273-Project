"""
STTransformer (Encoder-Only) model implementation.

This is the original encoder-only architecture that predicts all future frames in parallel.
Supports optional autoencoder for feature compression.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to sys.path for Config import
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from configs.Config import Config


class ResidualBlock(nn.Module):
    """A standard residual block: FFN + shortcut connection with pre-normalization."""
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        return x + self.ffn(self.norm(x))


class ResidualMLPHead(nn.Module):
    """
    Residual MLP head for output projection.
    
    Projects context vector through residual blocks to produce trajectory predictions.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_res_blocks=2, dropout=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU()
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, hidden_dim * 2, dropout) for _ in range(n_res_blocks)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        x = self.output_norm(x)
        return self.output_layer(x)


class SequenceAutoencoder(nn.Module):
    """
    Autoencoder for compressing input sequences.
    
    Encodes input sequence to a lower-dimensional latent space,
    then decodes back. Useful for feature compression/denoising.
    
    Args:
        input_dim: Input feature dimension
        latent_dim: Latent space dimension (compressed)
        window_size: Sequence length
        n_layers: Number of encoder/decoder layers
    """
    def __init__(self, input_dim, latent_dim, window_size, n_layers=2, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder: input_dim -> latent_dim
        encoder_layers = []
        dims = [input_dim] + [latent_dim * (2 ** i) for i in range(n_layers)]
        for i in range(n_layers):
            encoder_layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        encoder_layers.append(nn.Linear(dims[-1], latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder: latent_dim -> input_dim
        decoder_layers = []
        dims = [latent_dim] + [latent_dim * (2 ** i) for i in range(n_layers)][::-1]
        for i in range(n_layers):
            decoder_layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        decoder_layers.append(nn.Linear(dims[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, window_size, input_dim)
        Returns:
            reconstructed: (batch_size, window_size, input_dim)
            latent: (batch_size, window_size, latent_dim)
        """
        B, S, D = x.shape
        # Encode each timestep independently
        x_flat = x.view(-1, D)  # (B*S, D)
        latent_flat = self.encoder(x_flat)  # (B*S, latent_dim)
        latent = latent_flat.view(B, S, self.latent_dim)
        
        # Decode
        recon_flat = self.decoder(latent_flat)  # (B*S, D)
        reconstructed = recon_flat.view(B, S, D)
        
        return reconstructed, latent


class STTransformer(nn.Module):
    """
    Spatio-Temporal Transformer for NFL player trajectory prediction.
    
    Architecture:
    1. (Optional) Autoencoder: Compresses input features
    2. Input projection: feature_dim -> hidden_dim
    3. Positional encoding: learnable position embeddings
    4. Transformer encoder: temporal attention over sequence
    5. Attention pooling: aggregate sequence to context vector
    6. Residual MLP head: context -> trajectory predictions
    
    Input: (batch_size, window_size, input_dim)
    Output: (batch_size, horizon, 2) - cumulative displacements (dx, dy)
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dim,
        horizon,
        window_size,
        n_heads,
        n_layers,
        dropout=0.1,
        use_autoencoder=False,
        autoencoder_latent_dim=None,
        autoencoder_n_layers=2
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.use_autoencoder = use_autoencoder
        
        # Optional: Autoencoder for feature compression
        if use_autoencoder:
            if autoencoder_latent_dim is None:
                autoencoder_latent_dim = input_dim // 2  # Compress by half
            self.autoencoder = SequenceAutoencoder(
                input_dim=input_dim,
                latent_dim=autoencoder_latent_dim,
                window_size=window_size,
                n_layers=autoencoder_n_layers,
                dropout=dropout
            )
            effective_input_dim = autoencoder_latent_dim
        else:
            self.autoencoder = None
            effective_input_dim = input_dim

        # 1. Spatial: Feature embedding
        self.input_projection = nn.Linear(effective_input_dim, hidden_dim)
        
        # 2. Temporal: Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, window_size, hidden_dim))
        self.embed_dropout = nn.Dropout(dropout)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # 4. Pooling (Attention Pooling mechanism)
        self.pool_ln = nn.LayerNorm(hidden_dim)
        self.pool_attn = nn.MultiheadAttention(hidden_dim, num_heads=n_heads, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # 5. Output Head (ResidualMLPHead)
        self.head = ResidualMLPHead(
            input_dim=hidden_dim,
            hidden_dim=Config.STTRANSFORMER_MLP_HIDDEN_DIM,
            output_dim=horizon * 2,
            n_res_blocks=Config.STTRANSFORMER_N_RES_BLOCKS,
            dropout=0.2
        )

    def forward(self, x):
        B, S, _ = x.shape
        
        # Optional: Autoencoder compression
        if self.use_autoencoder:
            _, x = self.autoencoder(x)  # Use latent representation
        
        x_embed = self.input_projection(x)
        x = x_embed + self.pos_embed[:, :S, :]
        x = self.embed_dropout(x)
        
        h = self.transformer_encoder(x)

        q = self.pool_query.expand(B, -1, -1)
        ctx, _ = self.pool_attn(q, self.pool_ln(h), self.pool_ln(h))
        ctx = ctx.squeeze(1)

        out = self.head(ctx)
        out = out.view(B, self.horizon, 2)
        
        # Cumulative sum to get cumulative displacements
        out = torch.cumsum(out, dim=1)
        
        return out

