"""本地LLM连续嵌入接线检查，不训练投影器、不评价地图理解。"""

import json
from pathlib import Path

import torch

from .alignment_data import AlignmentExamples, MAP_MARKER
from .data import file_hash, json_write
from .embedding_bridge import MapEmbeddingProjector, insert_map_embeddings
from .training import resolve_device


def prompt_token_parts(tokenizer, question):
    if question.count(MAP_MARKER) != 1:
        raise ValueError('question requires exactly one map placeholder')
    rendered = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': '只根据提供的机载地图信息回答，未知不是安全，估计变化不代表源移动。'},
         {'role': 'user', 'content': question}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if rendered.count(MAP_MARKER) != 1:
        raise ValueError('chat template changed map placeholder')
    prefix, suffix = rendered.split(MAP_MARKER)
    before = tokenizer.encode(prefix, add_special_tokens=False)
    after = tokenizer.encode(suffix, add_special_tokens=False)
    if not before or not after:
        raise ValueError('template must put maps inside nonempty chat prefix and suffix')
    return before, after, rendered


def prepare_llm_inputs(model, tokenizer, projector, example, max_positions):
    before, after, rendered = prompt_token_parts(tokenizer, example['question'])
    embedding = model.get_input_embeddings()
    device = embedding.weight.device
    projector_device = next(projector.parameters()).device
    tokens = example['tokens'].unsqueeze(0).to(device=projector_device, dtype=next(projector.parameters()).dtype)
    valid = example['valid'].unsqueeze(0).to(projector_device)
    offsets = example['offset_seconds'].unsqueeze(0).to(projector_device)
    positions = len(before) + len(after) + tokens.shape[1] * tokens.shape[2]
    if type(max_positions) is not int or max_positions < positions:
        raise ValueError(f'input needs {positions} positions, exceeds explicit budget; no truncation permitted')
    maps, map_valid = projector(tokens, valid, offsets)
    text_ids = torch.tensor([before + after], device=device, dtype=torch.long)
    text = embedding(text_ids)
    inputs = insert_map_embeddings(text, torch.ones_like(text_ids, dtype=torch.bool),
                                   maps.to(device=device, dtype=text.dtype), map_valid.to(device), len(before))
    return inputs, {'rendered_template_with_map_placeholder': rendered,
                    'prefix_text_ids': before, 'suffix_text_ids': after,
                    'text_positions': len(before) + len(after), 'map_positions': maps.shape[1],
                    'valid_map_positions': int(map_valid.sum().item()), 'total_positions': positions,
                    'note': 'placeholder replaced by continuous vectors, not tokenized as numeric text'}


def smoke_local_llm(alignment_dataset, model_path, output, sample_index=0, device='cpu', seed=7, max_positions=2048):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be nonnegative integer')
    data = AlignmentExamples(alignment_dataset, 'validation')
    if type(sample_index) is not int or not 0 <= sample_index < len(data):
        raise ValueError('invalid sample_index')
    model_path, destination = Path(model_path).resolve(), Path(output)
    if not model_path.is_dir():
        raise ValueError('need existing local model directory')
    example = data.model_input(sample_index)
    destination.mkdir(parents=True, exist_ok=False)
    device = resolve_device(device)
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    json_write(destination / 'run_config.json', {'alignment_dataset': str(Path(alignment_dataset).resolve()),
               'alignment_manifest_sha256': file_hash(Path(alignment_dataset) / 'manifest.json'),
               'model_path': str(model_path), 'sample_index': sample_index, 'device': str(device), 'dtype': str(dtype),
               'seed': seed, 'max_positions': max_positions, 'local_files_only': True,
               'random_projector': True, 'optimizer_steps': 0})
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=dtype, attn_implementation='eager').to(device)
    model.eval().requires_grad_(False)
    encoder = data.manifest['encoder_settings']
    projector = MapEmbeddingProjector(model.get_input_embeddings().weight.shape[1], encoder['hidden_dim'],
                                      encoder['latent_tokens']).to(device).eval()
    with torch.inference_mode():
        inputs, prompt = prepare_llm_inputs(model, tokenizer, projector, example,
                                           min(max_positions, model.config.max_position_embeddings))
        result = model(**inputs, use_cache=False, return_dict=True)
        logits = result.logits[:, -1].float()
        if not torch.isfinite(logits).all():
            raise RuntimeError('nonfinite local LLM last-position logits')
        next_id = int(logits.argmax(-1).item())
    json_write(destination / 'prompt.json', prompt)
    relevant = sorted(path for path in model_path.iterdir()
                      if path.is_file() and (path.suffix == '.safetensors' or path.name in
                         ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'generation_config.json',
                          'model.safetensors.index.json')))
    report = {'status': 'forward_interface_ok_not_semantic_alignment', 'sample_id': data.rows[sample_index]['id'],
              'encoder_checkpoint_sha256': data.manifest['checkpoint_sha256'],
              'model_files_sha256': {path.name: file_hash(path) for path in relevant},
              'projector_config': projector.config, 'projector_trained': False,
              'inputs_embeds_shape': list(inputs['inputs_embeds'].shape),
              'logits_shape': list(result.logits.shape), 'finite_last_logits': True,
              'random_projector_next_id': next_id, 'random_projector_next_text': tokenizer.decode([next_id]),
              'optimizer_steps': 0, 'source_answer_passed': False, 'full_map_passed': False,
              'torch_version': str(torch.__version__), 'transformers_version': transformers.__version__,
              'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
              'text_positions': prompt['text_positions'], 'map_positions': prompt['map_positions'],
              'valid_map_positions': prompt['valid_map_positions'],
              'note': 'random embeddings only test tensor plumbing; next token is not a map answer or policy evidence'}
    json_write(destination / 'report.json', report)
    with (destination / 'README.md').open('x', encoding='utf-8') as stream:
        stream.write('# 本地LLM连续嵌入接线检查\n\n只有一次无梯度forward，没有训练／优化器／完整生成或能力评价。'
                     '\n投影器随机初始化，输出片段不是有效地图回答。输入未包含标准答案、原始全图或隐藏源。'
                     '\n先看report.json和prompt.json；后者用占位符标出实际连续向量插入位置，并记录前后文字ID。'
                     '\nrun_config.json及模型文件／编码器哈希用于追溯；模型权重本身未改动。\n')
    return report
