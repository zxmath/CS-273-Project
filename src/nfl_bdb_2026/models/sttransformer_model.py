"""
Registration and factory for STTransformer (encoder-only) model.
"""

from .sttransformer import STTransformer
from .registry import register_model


@register_model('sttransformer')
def create_sttransformer(config, **kwargs):
    """Factory for STTransformer model."""
    return STTransformer(
        input_dim=kwargs.get('input_dim'),
        hidden_dim=config.HIDDEN_DIM,
        horizon=config.MAX_FUTURE_HORIZON,
        window_size=config.WINDOW_SIZE,
        n_heads=config.STTRANSFORMER_N_HEADS,
        n_layers=config.STTRANSFORMER_N_LAYERS,
        dropout=kwargs.get('dropout', 0.1),
        use_autoencoder=config.STTRANSFORMER_USE_AE if hasattr(config, 'STTRANSFORMER_USE_AE') else False,
        autoencoder_latent_dim=config.STTRANSFORMER_AE_LATENT_DIM if hasattr(config, 'STTRANSFORMER_AE_LATENT_DIM') else None,
        autoencoder_n_layers=config.STTRANSFORMER_AE_N_LAYERS if hasattr(config, 'STTRANSFORMER_AE_N_LAYERS') else 2,
    )

