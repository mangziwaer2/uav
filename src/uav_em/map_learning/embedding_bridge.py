"""连续地图嵌入接口；不加载LLM，随机初始化不代表已经完成语义对齐。"""

import math

import torch
from torch import nn


class MapEmbeddingProjector(nn.Module):
    """将[B,T,K,C]地图向量和实际时间映射到[B,T*K,D]；D由目标LLM指定。"""

    def __init__(self, llm_dim, encoder_dim=96, latent_tokens=32, hidden_dim=256, time_scale_seconds=35.0):
        super().__init__()
        dimensions = (llm_dim, encoder_dim, latent_tokens, hidden_dim)
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError('embedding dimensions must be positive integers')
        if not math.isfinite(time_scale_seconds) or time_scale_seconds <= 0:
            raise ValueError('time scale must be positive and finite')
        self.config = {'llm_dim': llm_dim, 'encoder_dim': encoder_dim, 'latent_tokens': latent_tokens,
                       'hidden_dim': hidden_dim, 'time_scale_seconds': time_scale_seconds}
        self.project = nn.Sequential(nn.LayerNorm(encoder_dim), nn.Linear(encoder_dim, hidden_dim),
                                     nn.GELU(), nn.Linear(hidden_dim, llm_dim))
        self.time = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, llm_dim))

    def forward(self, tokens, valid, offset_seconds):
        if tokens.ndim != 4 or tuple(tokens.shape[-2:]) != (self.config['latent_tokens'], self.config['encoder_dim']):
            raise ValueError('expected [batch,frames,latent_tokens,encoder_dim]')
        if (valid.shape != tokens.shape[:2] or valid.dtype != torch.bool or offset_seconds.shape != valid.shape
                or valid.device != tokens.device or offset_seconds.device != tokens.device):
            raise ValueError('time and boolean validity must match token frames and device')
        if not tokens.is_floating_point() or not offset_seconds.is_floating_point() or not tokens.shape[1]:
            raise ValueError('tokens and offsets must be floating point with nonempty frames')
        if not valid[:, -1].all() or not torch.all(offset_seconds[:, -1].abs() <= 1e-6):
            raise ValueError('last frame must be valid at current time zero')
        if not torch.isfinite(offset_seconds[valid]).all() or torch.any(offset_seconds[valid] > 1e-6):
            raise ValueError('valid offsets must be finite and nonpositive')
        for row in range(len(tokens)):
            times = offset_seconds[row, valid[row]]
            if torch.any(times[1:] <= times[:-1]):
                raise ValueError('valid frames must be strictly chronological without duplicates')
        if not torch.isfinite(tokens[valid]).all():
            raise ValueError('valid map tokens must be finite')
        clean = tokens.masked_fill(~valid[:, :, None, None], 0)
        offsets = offset_seconds.masked_fill(~valid, 0).to(tokens.dtype)
        time_embedding = self.time((offsets / self.config['time_scale_seconds']).unsqueeze(-1))
        embedded = self.project(clean) + time_embedding.unsqueeze(2)
        mask = valid.unsqueeze(-1).expand(*valid.shape, self.config['latent_tokens'])
        embedded = embedded.masked_fill(~mask.unsqueeze(-1), 0)
        return embedded.flatten(1, 2), mask.flatten(1, 2)


def insert_map_embeddings(text_embeddings, text_valid, map_embeddings, map_valid, insert_at, labels=None):
    """在文字词嵌入之后插入地图；返回标准张量字段，不执行语言模型。"""
    if (text_embeddings.ndim != 3 or map_embeddings.ndim != 3
            or text_embeddings.shape[0] != map_embeddings.shape[0]
            or text_embeddings.shape[-1] != map_embeddings.shape[-1]):
        raise ValueError('text and maps must share batch and embedding dimensions')
    if type(insert_at) is not int or not 0 <= insert_at <= text_embeddings.shape[1]:
        raise ValueError('insert_at must be a valid text sequence boundary')
    for embeddings, mask in ((text_embeddings, text_valid), (map_embeddings, map_valid)):
        if mask.shape != embeddings.shape[:2] or mask.dtype != torch.bool or mask.device != embeddings.device:
            raise ValueError('attention masks must be boolean and match embedding sequences')
    if text_embeddings.dtype != map_embeddings.dtype or text_embeddings.device != map_embeddings.device:
        raise ValueError('cast maps explicitly to the target text embedding dtype and device')
    embeddings = torch.cat((text_embeddings[:, :insert_at], map_embeddings, text_embeddings[:, insert_at:]), dim=1)
    valid = torch.cat((text_valid[:, :insert_at], map_valid, text_valid[:, insert_at:]), dim=1)
    result = {'inputs_embeds': embeddings.masked_fill(~valid.unsqueeze(-1), 0), 'attention_mask': valid.long(),
              'position_ids': (valid.long().cumsum(-1) - 1).clamp_min(0)}
    if labels is not None:
        if labels.shape != text_valid.shape or labels.dtype != torch.long or labels.device != embeddings.device:
            raise ValueError('labels must be int64 text positions on the embedding device')
        ignored = labels.new_full(map_valid.shape, -100)
        combined = torch.cat((labels[:, :insert_at], ignored, labels[:, insert_at:]), dim=1)
        result['labels'] = combined.masked_fill(~valid, -100)
    return result
