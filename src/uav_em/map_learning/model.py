"""可选PyTorch空间瓶颈编解码器；未知区不以电磁真值监督。"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as functional

from .data import CHANNELS


ARCHITECTURE = 'spatial_anchor_local_decode_v1'
PATCH_ARCHITECTURE = 'spatial_patch_decode_v2'
ARCHITECTURES = (ARCHITECTURE, PATCH_ARCHITECTURE)


@dataclass(frozen=True)
class EncoderSettings:
    """patch_size为分块格数；latent_tokens为每帧瓶颈数，不代表无损压缩。

    hidden_dim为向量维数，heads为注意力头数，layers为空间编码层数。
    """

    patch_size: int = 4
    hidden_dim: int = 96
    latent_tokens: int = 32
    heads: int = 4
    layers: int = 2
    architecture: str = ARCHITECTURE

    def validate(self):
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 1
               for name in ('patch_size', 'hidden_dim', 'latent_tokens', 'heads', 'layers')):
            raise ValueError('encoder settings must be positive integers')
        if self.hidden_dim % self.heads:
            raise ValueError('hidden_dim must be divisible by heads')
        if self.architecture not in ARCHITECTURES:
            raise ValueError('unsupported map architecture')


def coordinates(height, width, reference):
    rows, columns = torch.meshgrid(torch.linspace(-1, 1, height, device=reference.device, dtype=reference.dtype),
                                  torch.linspace(-1, 1, width, device=reference.device, dtype=reference.dtype), indexing='ij')
    return torch.stack((columns, rows), dim=-1).reshape(1, height * width, 2)


class MapAutoencoder(nn.Module):
    def __init__(self, settings=EncoderSettings()):
        super().__init__()
        settings.validate()
        self.settings = settings
        hidden = settings.hidden_dim
        latent_rows = max(divisor for divisor in range(1, math.isqrt(settings.latent_tokens) + 1)
                          if settings.latent_tokens % divisor == 0)
        self.latent_grid = (latent_rows, settings.latent_tokens // latent_rows)
        if settings.architecture == PATCH_ARCHITECTURE:
            self.stem = nn.Sequential(nn.Conv2d(len(CHANNELS), hidden, 3, padding=1), nn.GELU())
            self.patch = nn.Conv2d(hidden, hidden, settings.patch_size, stride=settings.patch_size)
        else:
            self.stem = nn.Identity()
            self.patch = nn.Conv2d(len(CHANNELS), hidden, settings.patch_size, stride=settings.patch_size)
        self.encoder_position = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        layer = nn.TransformerEncoderLayer(hidden, settings.heads, hidden * 3, dropout=0, batch_first=True)
        self.spatial = nn.TransformerEncoder(layer, settings.layers)
        self.queries = nn.Parameter(torch.randn(1, settings.latent_tokens, hidden) * 0.02)
        self.compress = nn.MultiheadAttention(hidden, settings.heads, dropout=0, batch_first=True)
        self.latent_norm = nn.LayerNorm(hidden)
        self.decoder_position = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.expand = nn.MultiheadAttention(hidden, settings.heads, dropout=0, batch_first=True)
        if settings.architecture == PATCH_ARCHITECTURE:
            self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())
            self.patch_output = nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
                nn.Conv2d(hidden, hidden * settings.patch_size ** 2, 1), nn.PixelShuffle(settings.patch_size),
                nn.GELU(), nn.Conv2d(hidden, len(CHANNELS), 3, padding=1))
        else:
            self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, len(CHANNELS)))

    def encode(self, maps):
        if maps.ndim != 4 or maps.shape[1] != len(CHANNELS):
            raise ValueError('maps must have shape [batch, channels, rows, columns]')
        height, width = maps.shape[-2:]
        patch = self.settings.patch_size
        patches = self.patch(functional.pad(self.stem(maps), (0, -width % patch, 0, -height % patch)))
        values = patches.flatten(2).transpose(1, 2)
        values = self.spatial(values + self.encoder_position(coordinates(*patches.shape[-2:], values)))
        spatial_grid = values.transpose(1, 2).reshape(maps.shape[0], self.settings.hidden_dim, *patches.shape[-2:])
        anchors = functional.adaptive_avg_pool2d(spatial_grid, self.latent_grid).flatten(2).transpose(1, 2)
        query = self.queries + anchors
        compressed, _ = self.compress(query, values, values, need_weights=False)
        return self.latent_norm(query + compressed)

    def decode(self, tokens, height, width):
        if tokens.ndim != 3 or tokens.shape[1:] != (self.settings.latent_tokens, self.settings.hidden_dim):
            raise ValueError('tokens do not match the configured bottleneck')
        if any(type(value) is not int or value < 1 for value in (height, width)):
            raise ValueError('decode size must be positive integers')
        if self.settings.architecture == PATCH_ARCHITECTURE:
            patch = self.settings.patch_size
            rows, columns = math.ceil(height / patch), math.ceil(width / patch)
            grid = tokens.transpose(1, 2).reshape(tokens.shape[0], self.settings.hidden_dim, *self.latent_grid)
            local = functional.interpolate(grid, size=(rows, columns), mode='bilinear', align_corners=False)
            query = local.flatten(2).transpose(1, 2) + self.decoder_position(coordinates(rows, columns, tokens))
            expanded, _ = self.expand(query, tokens, tokens, need_weights=False)
            features = self.output(query + expanded).transpose(1, 2).reshape(
                tokens.shape[0], self.settings.hidden_dim, rows, columns)
            return self.patch_output(features)[:, :, :height, :width]
        grid = tokens.transpose(1, 2).reshape(tokens.shape[0], self.settings.hidden_dim, *self.latent_grid)
        local = functional.interpolate(grid, size=(height, width), mode='bilinear', align_corners=False)
        local = local.flatten(2).transpose(1, 2)
        query = local + self.decoder_position(coordinates(height, width, tokens))
        expanded, _ = self.expand(query, tokens, tokens, need_weights=False)
        return self.output(query + expanded).transpose(1, 2).reshape(tokens.shape[0], len(CHANNELS), height, width)

    def forward(self, maps):
        return self.decode(self.encode(maps), *maps.shape[-2:])


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)


def balanced_binary(logits, target):
    error = functional.binary_cross_entropy_with_logits(logits, target, reduction='none')
    positive, negative = target > 0.5, target <= 0.5
    groups = positive.any().to(error.dtype) + negative.any().to(error.dtype)
    return (masked_mean(error, positive) + masked_mean(error, negative)) / groups.clamp_min(1)


def shape_binary(logits, target):
    probability = logits.sigmoid().flatten(1)
    labels = target.flatten(1)
    dice = 1 - (2 * (probability * labels).sum(1) + 1) / (probability.sum(1) + labels.sum(1) + 1)
    mask = target.unsqueeze(1)
    maximum = functional.max_pool2d(mask, 3, stride=1, padding=1)
    minimum = -functional.max_pool2d(-mask, 3, stride=1, padding=1)
    weight = 1 + 2 * (maximum != minimum).squeeze(1).to(logits.dtype)
    error = functional.binary_cross_entropy_with_logits(logits, target, reduction='none')
    return (error * weight).sum() / weight.sum() + dice.mean()


def reconstruction_loss(predicted, target, objective='balanced_v1'):
    """功率／年龄只监督有支持格；强干扰格另加权，未知mask与障碍单独监督。"""
    if objective not in ('balanced_v1', 'boundary_v2'):
        raise ValueError('unsupported reconstruction objective')
    binary = shape_binary if objective == 'boundary_v2' else balanced_binary
    support = target[:, 1] > 0.5
    high = support & (target[:, 0] > 40 / 90)
    power_error = functional.smooth_l1_loss(predicted[:, 0], target[:, 0], reduction='none')
    losses = {'power': masked_mean(power_error, support) + masked_mean(power_error, high),
              'support': binary(predicted[:, 1], target[:, 1]),
              'uncertainty': functional.smooth_l1_loss(predicted[:, 2], target[:, 2]),
              'age': masked_mean(functional.smooth_l1_loss(predicted[:, 3], target[:, 3], reduction='none'), support),
              'blocked': binary(predicted[:, 4], target[:, 4]),
              'height': functional.smooth_l1_loss(predicted[:, 5], target[:, 5])}
    total = 5 * losses['power'] + losses['support'] + losses['uncertainty'] + losses['age'] + losses['blocked'] + losses['height']
    return total, losses


def metric_sums(predicted, target):
    support = target[:, 1] > 0.5
    high = support & (target[:, 0] > 40 / 90)
    predicted_support = predicted[:, 1] > 0
    blocked, predicted_blocked = target[:, 4] > 0.5, predicted[:, 4] > 0
    return {'power_error_db': float(((predicted[:, 0] - target[:, 0]).abs() * support).sum().item() * 90),
            'power_count': int(support.sum().item()),
            'high_error_db': float(((predicted[:, 0] - target[:, 0]).abs() * high).sum().item() * 90),
            'high_count': int(high.sum().item()),
            'support_intersection': int((support & predicted_support).sum().item()),
            'support_union': int((support | predicted_support).sum().item()),
            'support_false_positive': int((~support & predicted_support).sum().item()),
            'support_false_negative': int((support & ~predicted_support).sum().item()),
            'unknown_count': int((~support).sum().item()),
            'blocked_intersection': int((blocked & predicted_blocked).sum().item()),
            'blocked_union': int((blocked | predicted_blocked).sum().item()),
            'uncertainty_error_db': float((predicted[:, 2] - target[:, 2]).abs().sum().item() * 30),
            'height_error_m': float((predicted[:, 5] - target[:, 5]).abs().sum().item() * 80),
            'cell_count': target[:, 1].numel(),
            'age_error_seconds': float(((predicted[:, 3] - target[:, 3]).abs() * support).sum().item() * 35)}
