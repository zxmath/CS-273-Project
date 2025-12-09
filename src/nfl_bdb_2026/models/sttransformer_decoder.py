"""
Full Spatio-Temporal Transformer with Encoder-Decoder Architecture.

This model uses:
1. Encoder: Processes input sequence (past frames)
2. Decoder: Autoregressively generates future frames
3. Cross-attention: Decoder attends to encoder outputs

Also includes optional autoencoder for feature compression.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Optional

# Add project root to sys.path for Config import
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from configs.Config import Config


class ResidualBlock(nn.Module):
    """Residual block with pre-normalization."""
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


class STTransformerDecoder(nn.Module):
    """
    Full Spatio-Temporal Transformer with Encoder-Decoder Architecture.
    
    Architecture:
    1. (Optional) Autoencoder: Compresses input features
    2. Encoder: Processes input sequence (window_size frames)
    3. Decoder: Autoregressively generates future frames (horizon frames)
    4. Cross-attention: Decoder attends to encoder outputs
    
    Input: (batch_size, window_size, input_dim)
    Output: (batch_size, horizon, 2) - cumulative displacements (dx, dy)
    
    Training: Teacher forcing with ground truth
    Inference: Autoregressive generation (use previous predictions)
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dim,
        horizon,
        window_size,
        n_heads,
        n_encoder_layers,
        n_decoder_layers,
        dropout=0.1,
        use_teacher_forcing=True,
        use_autoencoder=False,
        autoencoder_latent_dim=None,
        autoencoder_n_layers=2
    ):
        super().__init__()
        self.horizon = horizon
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.use_teacher_forcing = use_teacher_forcing
        self.use_autoencoder = use_autoencoder
        self._future_inputs = None  # Store for teacher forcing
        self._teacher_forcing_prob = None  # For scheduled sampling (set dynamically)
        
        # 0. Optional Autoencoder
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
        
        # 1. Input projection for encoder
        self.encoder_input_proj = nn.Linear(effective_input_dim, hidden_dim)
        self.encoder_pos_embed = nn.Parameter(torch.randn(1, window_size, hidden_dim))
        self.encoder_dropout = nn.Dropout(dropout)
        
        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        
        # 3. Decoder input projection and positional encoding
        self.decoder_input_dim = 2  # (dx, dy) for each future timestep
        self.decoder_start_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, horizon, hidden_dim))
        self.decoder_input_proj = nn.Linear(self.decoder_input_dim, hidden_dim)
        self.decoder_dropout = nn.Dropout(dropout)
        
        # 4. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)
        
        # 5. Output head: project decoder outputs to (dx, dy)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # (dx, dy)
        )
        
    def set_future_inputs(self, future_inputs):
        """
        Set future inputs for teacher forcing during training.
        Called by training loop before forward pass.
        """
        self._future_inputs = future_inputs
    
    def set_teacher_forcing_prob(self, prob):
        """
        Set teacher forcing probability for scheduled sampling.
        
        Args:
            prob: Probability of using teacher forcing (0.0 to 1.0)
                1.0 = always use teacher forcing (ground truth)
                0.0 = always use autoregressive (own predictions)
        """
        self._teacher_forcing_prob = prob
    
    def clear_future_inputs(self):
        """Clear future inputs to ensure autoregressive mode."""
        self._future_inputs = None
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input sequence (batch_size, window_size, input_dim)
        
        Returns:
            predictions: (batch_size, horizon, 2) - cumulative displacements
        """
        batch_size = x.shape[0]
        
        # Optional: Autoencoder compression
        if self.use_autoencoder:
            _, x = self.autoencoder(x)  # Use latent representation
        
        # ===================================================================
        # ENCODER: Process input sequence
        # ===================================================================
        x_embed = self.encoder_input_proj(x)  # (B, window_size, hidden_dim)
        x_embed = x_embed + self.encoder_pos_embed
        x_embed = self.encoder_dropout(x_embed)
        
        encoder_output = self.encoder(x_embed)  # (B, window_size, hidden_dim)
        
        # ===================================================================
        # DECODER: Generate future frames
        # ===================================================================
        # Check if we should use teacher forcing
        future_inputs = getattr(self, '_future_inputs', None)
        
        # Scheduled sampling: randomly decide whether to use teacher forcing
        if self.training and self.use_teacher_forcing and future_inputs is not None:
            if self._teacher_forcing_prob is not None:
                # Use scheduled sampling probability
                # Use torch random for reproducibility (seed-controlled)
                use_tf = torch.rand(1, device=x.device).item() < self._teacher_forcing_prob
            else:
                # Always use teacher forcing if available (default behavior)
                use_tf = True
        else:
            # No teacher forcing: use autoregressive
            use_tf = False
        
        if use_tf:
            # Teacher forcing: use ground truth cumulative displacements
            # Convert to incremental (dx, dy) by taking diff
            future_cumulative = future_inputs  # (B, horizon, 2) - already cumulative
            # Convert to incremental: first step is first value, rest are differences
            future_incremental = torch.zeros_like(future_cumulative)
            future_incremental[:, 0, :] = future_cumulative[:, 0, :]
            future_incremental[:, 1:, :] = future_cumulative[:, 1:, :] - future_cumulative[:, :-1, :]
            
            decoder_input_embed = self.decoder_input_proj(future_incremental)
            decoder_input_embed = decoder_input_embed + self.decoder_pos_embed
            decoder_input_embed = self.decoder_dropout(decoder_input_embed)
            
            # Create causal mask for decoder
            causal_mask = torch.triu(
                torch.ones(self.horizon, self.horizon, device=x.device), 
                diagonal=1
            ).bool()
            
            # Decode all timesteps in parallel (teacher forcing)
            decoder_output = self.decoder(
                tgt=decoder_input_embed,
                memory=encoder_output,
                tgt_mask=causal_mask
            )  # (B, horizon, hidden_dim)
            
            # Project to incremental (dx, dy) predictions
            predictions_inc = self.output_head(decoder_output)  # (B, horizon, 2)
            
            # Convert to cumulative displacements
            predictions = torch.cumsum(predictions_inc, dim=1)
        else:
            # Autoregressive or scheduled sampling with own predictions
            # Autoregressive: generate one timestep at a time
            predictions_inc = []
            decoder_input = self.decoder_start_token.expand(batch_size, -1, -1)
            
            for t in range(self.horizon):
                # Add positional encoding
                pos_emb = self.decoder_pos_embed[:, t:t+1, :]
                decoder_input_step = decoder_input[:, -1:, :] + pos_emb  # Only use last token
                decoder_input_step = self.decoder_dropout(decoder_input_step)
                
                # Decode one timestep
                decoder_output = self.decoder(
                    tgt=decoder_input_step,
                    memory=encoder_output
                )  # (B, 1, hidden_dim)
                
                # Predict incremental (dx, dy) for this timestep
                pred_step = self.output_head(decoder_output)  # (B, 1, 2)
                predictions_inc.append(pred_step)
                
                # Use predicted (dx, dy) as input for next step
                pred_embed = self.decoder_input_proj(pred_step)  # (B, 1, hidden_dim)
                decoder_input = torch.cat([decoder_input, pred_embed], dim=1)
            
            # Stack and convert to cumulative
            predictions_inc = torch.cat(predictions_inc, dim=1)  # (B, horizon, 2)
            predictions = torch.cumsum(predictions_inc, dim=1)
        
        return predictions


# Wrapper compatible with training interface
class STTransformerDecoderWrapper(nn.Module):
    """
    Wrapper to match the interface of STTransformer for training.
    
    This wrapper handles teacher forcing by storing targets during training.
    """
    def __init__(self, input_dim, hidden_dim, horizon, window_size, n_heads, n_layers, **kwargs):
        super().__init__()
        # Split layers between encoder and decoder
        n_encoder_layers = max(1, n_layers // 2)
        n_decoder_layers = max(1, n_layers - n_encoder_layers)
        
        # Register as proper submodule
        self.model = STTransformerDecoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            window_size=window_size,
            n_heads=n_heads,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            dropout=kwargs.get('dropout', 0.1),
            use_teacher_forcing=kwargs.get('use_teacher_forcing', True),
            use_autoencoder=kwargs.get('use_autoencoder', False),
            autoencoder_latent_dim=kwargs.get('autoencoder_latent_dim', None),
            autoencoder_n_layers=kwargs.get('autoencoder_n_layers', 2)
        )
        
    def forward(self, x):
        """Forward pass - uses stored future_inputs if available (teacher forcing)."""
        return self.model(x)
    
    def set_future_inputs(self, future_inputs):
        """Set future inputs for teacher forcing."""
        self.model.set_future_inputs(future_inputs)
    
    def set_teacher_forcing_prob(self, prob):
        """Set teacher forcing probability for scheduled sampling."""
        self.model.set_teacher_forcing_prob(prob)
    
    def clear_future_inputs(self):
        """Clear future inputs to ensure autoregressive mode."""
        self.model.clear_future_inputs()
    
    def eval(self):
        """Set to evaluation mode."""
        super().eval()  # Call parent's eval() first
        self.model.eval()
        return self
    
    def train(self, mode=True):
        """Set to training mode."""
        super().train(mode)  # Call parent's train() first
        self.model.train(mode)
        return self
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Get state dict."""
        return self.model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict."""
        return self.model.load_state_dict(state_dict, strict=strict)

