"""
Registration and factory for STTransformerDecoder model.
"""

from .sttransformer_decoder import STTransformerDecoderWrapper
from .registry import register_model


@register_model('sttransformer_decoder')
def create_sttransformer_decoder(config, **kwargs):
    """Factory for STTransformerDecoder model."""
    return STTransformerDecoderWrapper(
        input_dim=kwargs.get('input_dim'),
        hidden_dim=config.HIDDEN_DIM,
        horizon=config.MAX_FUTURE_HORIZON,
        window_size=config.WINDOW_SIZE,
        n_heads=config.STTRANSFORMER_DECODER_N_HEADS if hasattr(config, 'STTRANSFORMER_DECODER_N_HEADS') else config.STTRANSFORMER_N_HEADS,
        n_layers=config.STTRANSFORMER_DECODER_N_LAYERS if hasattr(config, 'STTRANSFORMER_DECODER_N_LAYERS') else config.STTRANSFORMER_N_LAYERS,
        dropout=kwargs.get('dropout', 0.1),
        use_teacher_forcing=config.STTRANSFORMER_DECODER_USE_TF if hasattr(config, 'STTRANSFORMER_DECODER_USE_TF') else True,
        use_autoencoder=config.STTRANSFORMER_DECODER_USE_AE if hasattr(config, 'STTRANSFORMER_DECODER_USE_AE') else False,
        autoencoder_latent_dim=config.STTRANSFORMER_DECODER_AE_LATENT_DIM if hasattr(config, 'STTRANSFORMER_DECODER_AE_LATENT_DIM') else None,
        autoencoder_n_layers=config.STTRANSFORMER_DECODER_AE_N_LAYERS if hasattr(config, 'STTRANSFORMER_DECODER_AE_N_LAYERS') else 2,
    )

