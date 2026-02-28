"""
C3-LDM Models Package

Census-Consistent, Conditional Latent Diffusion for Population Mapping
"""

from .baseline import BaselineDasymetric
from .learned_baseline import LinearRegressionBaseline, CNNBaseline, ScaledDasymetric
from .vae import ResidualVAE
from .time_embedding import TimeEmbedding, TimestepBlock, TimestepEmbedSequential
from .conditional_encoder import DualBranchConditionalEncoder
from .product_embedding import ProductEmbedding
from .unet_simple import SimpleUNet
from .census_layer import CensusConsistencyLayer, CensusConsistencyLayerVectorized
from .sampler import DDPMSampler, DDIMSampler, C3LDMSampler

__all__ = [
    'BaselineDasymetric',
    'LinearRegressionBaseline',
    'CNNBaseline',
    'ScaledDasymetric',
    'ResidualVAE',
    'TimeEmbedding',
    'TimestepBlock',
    'TimestepEmbedSequential',
    'DualBranchConditionalEncoder',
    'ProductEmbedding',
    'SimpleUNet',
    'CensusConsistencyLayer',
    'CensusConsistencyLayerVectorized',
    'DDPMSampler',
    'DDIMSampler',
    'C3LDMSampler',
]
