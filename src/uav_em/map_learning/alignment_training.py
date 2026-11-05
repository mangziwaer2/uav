"""冻结LLM的地图投影／时间层训练；独立传入机载测向，不训练源定位或RL。"""

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from .data import file_hash, json_write
from .embedding_bridge import MapEmbeddingProjector
from .llm_bridge import prepare_llm_inputs, prompt_token_parts
from .sensor_evidence import SensorAlignmentExamples
from .training import resolve_device


@dataclass(frozen=True)
class AlignmentTrainingSettings:
    """位置预算含答案及地图padding；microbatch固定1，累积后更新。"""

    epochs: int = 1
    gradient_accumulation: int = 8
    learning_rate: float = 0.0002
    max_positions: int = 8192
    history_seconds: float = 60.0
    checkpoint_every: int = 25
    validation_examples: int = 16
    max_steps: int | None = None
    seed: int = 7

    def validate(self):
        for name in ('epochs', 'gradient_accumulation', 'max_positions', 'checkpoint_every', 'validation_examples'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be nonnegative integer')
        if self.max_steps is not None and (type(self.max_steps) is not int or self.max_steps < 1):
            raise ValueError('max_steps must be None or positive integer')
        for name in ('learning_rate', 'history_seconds'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')


def answer_token_ids(tokenizer, answer):
    if type(tokenizer.eos_token_id) is not int or tokenizer.eos_token_id < 0:
        raise ValueError('tokenizer needs explicit scalar eos_token_id')
    serialized = json.dumps(answer, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return tokenizer.encode(serialized, add_special_tokens=False) + [tokenizer.eos_token_id]


def supervised_inputs(model, tokenizer, projector, example, answer, max_positions):
    target = answer_token_ids(tokenizer, answer)
    inputs, metadata = prepare_llm_inputs(model, tokenizer, projector, example, max_positions - len(target))
    embedding = model.get_input_embeddings()
    target_ids = torch.tensor([target], device=embedding.weight.device, dtype=torch.long)
    prompt_length = inputs['inputs_embeds'].shape[1]
    inputs['inputs_embeds'] = torch.cat((inputs['inputs_embeds'], embedding(target_ids)), dim=1)
    inputs['attention_mask'] = torch.cat((inputs['attention_mask'], torch.ones_like(target_ids)), dim=1)
    inputs['position_ids'] = (inputs['attention_mask'].cumsum(-1) - 1).clamp_min(0)
    inputs['labels'] = torch.cat((target_ids.new_full((1, prompt_length), -100), target_ids), dim=1)
    metadata.update({'prompt_positions': prompt_length, 'answer_positions': len(target),
                     'total_positions': prompt_length + len(target),
                     'supervision': 'causal_teacher_forcing; prompt_and_map_labels_ignored; eos_included'})
    return inputs, metadata


def model_fingerprint(model_path):
    path = Path(model_path)
    if not path.is_dir():
        raise ValueError('need existing local model directory')
    selected = sorted(item for item in path.iterdir() if item.is_file() and
                      (item.suffix == '.safetensors' or item.name in
                       ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
                        'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt',
                        'tokenizer.model', 'model.safetensors.index.json')))
    if not any(item.suffix == '.safetensors' for item in selected):
        raise ValueError('local model must contain safetensors weights')
    return {item.name: file_hash(item) for item in selected}


def audit_token_budget(alignment_dataset, evidence_dataset, tokenizer, settings, model_max_positions):
    settings.validate()
    if settings.max_positions > model_max_positions:
        raise ValueError('explicit max_positions exceeds model context limit')
    result = {'schema_version': 1, 'max_positions': settings.max_positions,
              'history_seconds': settings.history_seconds, 'includes_complete_answers_and_eos': True,
              'truncated_examples': 0, 'splits': {}}
    exceeded = 0
    for split in ('train', 'validation'):
        data = SensorAlignmentExamples(alignment_dataset, evidence_dataset, split, settings.history_seconds)
        dimensions = data.manifest['encoder_settings']
        lengths, longest, kinds = [], None, {}
        for index, row in enumerate(data.rows):
            before, after, _ = prompt_token_parts(tokenizer, data.question(index))
            maps = len(row['inputs']['frame_indices']) * dimensions['latent_tokens']
            answer = len(answer_token_ids(tokenizer, data.answer(index)))
            length = len(before) + len(after) + maps + answer
            lengths.append(length)
            kinds[row['kind']] = kinds.get(row['kind'], 0) + 1
            if longest is None or length > longest['positions']:
                longest = {'sample_id': row['id'], 'index': index, 'positions': length,
                           'prompt_positions': len(before) + len(after) + maps, 'answer_positions': answer}
            exceeded += int(length > settings.max_positions)
        result['splits'][split] = {'examples': len(lengths), 'kinds': kinds, 'longest': longest,
                                   'position_percentiles': dict(zip(('min', 'p50', 'p95', 'max'),
                                                                   np.percentile(lengths, [0, 50, 95, 100]).tolist()))}
        print(f'长度审核 {split}：{len(lengths)}题，最长{longest["positions"]}位置。', flush=True)
    result['over_budget_examples'] = exceeded
    result['passed'] = exceeded == 0
    return result


def configure_frozen_llm(model):
    model.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('LLM must be fully frozen')


def training_order(length, epochs, seed):
    return [index for epoch in range(epochs)
            for index in torch.randperm(length, generator=torch.Generator().manual_seed(seed + epoch)).tolist()]


def save_training_checkpoint(path, projector, optimizer, scaler, step, cursor, identity, settings):
    temporary = path.with_suffix('.tmp')
    torch.save({'schema_version': 1, 'projector_config': projector.config,
                'projector': projector.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                'optimizer_steps': step, 'sample_cursor': cursor, 'identity': identity, 'settings': asdict(settings),
                'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                'rng_python': random.getstate()}, temporary)
    temporary.replace(path)


def restore_training_checkpoint(path, projector, optimizer, scaler, identity, settings, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('schema_version') != 1 or checkpoint['identity'] != identity:
        raise ValueError('resume model, encoder, evidence or data identity mismatch')
    previous, current = checkpoint['settings'].copy(), asdict(settings)
    for key in ('max_steps', 'checkpoint_every', 'validation_examples'):
        previous.pop(key)
        current.pop(key)
    if previous != current or checkpoint['projector_config'] != projector.config:
        raise ValueError('resume training settings mismatch')
    projector.load_state_dict(checkpoint['projector'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value) and key != 'step':
                state[key] = value.to(device)
    if checkpoint['scaler']:
        scaler.load_state_dict(checkpoint['scaler'])
    torch.set_rng_state(checkpoint['rng_cpu'])
    random.setstate(checkpoint['rng_python'])
    if device.type == 'cuda' and checkpoint['rng_cuda']:
        torch.cuda.set_rng_state(checkpoint['rng_cuda'][0], device)
    return checkpoint['optimizer_steps'], checkpoint['sample_cursor']


def precision_context(device, dtype):
    return torch.autocast(device_type='cuda', dtype=dtype) if device.type == 'cuda' else nullcontext()


def is_cuda_oom(error):
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    accelerator_error = getattr(torch, 'AcceleratorError', RuntimeError)
    message = str(error).lower()
    return isinstance(error, accelerator_error) and 'cuda' in message and 'out of memory' in message


def update_projector(model, tokenizer, projector, data, indices, optimizer, scaler, settings, device, dtype, progress=None):
    optimizer.zero_grad(set_to_none=True)
    losses, positions = [], []
    for index in indices:
        if progress is not None:
            progress.update({'sample_id': data.rows[index]['id'], 'phase': 'prepare_supervision'})
        with precision_context(device, dtype):
            inputs, metadata = supervised_inputs(model, tokenizer, projector, data.model_input(index),
                                                 data.answer(index), settings.max_positions)
            if progress is not None:
                progress.update({'phase': 'forward', 'positions': metadata['total_positions'],
                                 'prompt_positions': metadata['prompt_positions'],
                                 'answer_positions': metadata['answer_positions']})
            output = model(**inputs, use_cache=False, return_dict=True)
            loss = output.loss
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite training loss')
        if progress is not None:
            progress['phase'] = 'backward'
        scaler.scale(loss / len(indices)).backward()
        losses.append(float(loss.detach()))
        positions.append(metadata['total_positions'])
        del output, inputs, loss
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0, error_if_nonfinite=True)
    if not math.isfinite(float(norm)) or float(norm) == 0:
        raise RuntimeError('projector has no finite nonzero gradient')
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError('frozen LLM unexpectedly has gradients')
    if progress is not None:
        progress['phase'] = 'optimizer_step'
    scaler.step(optimizer)
    scaler.update()
    return {'loss': sum(losses) / len(losses), 'gradient_norm': float(norm), 'max_positions': max(positions)}


def validation_loss(model, tokenizer, projector, data, settings, device, dtype):
    model.eval()
    projector.eval()
    indices = np.linspace(0, len(data) - 1, min(len(data), settings.validation_examples), dtype=int).tolist()
    losses = []
    with torch.no_grad():
        for index in indices:
            with precision_context(device, dtype):
                inputs, _ = supervised_inputs(model, tokenizer, projector, data.model_input(index),
                                             data.answer(index), settings.max_positions)
                loss = model(**inputs, use_cache=False, return_dict=True).loss
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite validation loss')
            losses.append(float(loss))
    model.train()
    projector.train()
    return {'teacher_forced_loss': sum(losses) / len(losses), 'examples': len(losses),
            'indices': indices, 'free_generation_evaluated': False}


def train_map_alignment(alignment_dataset, evidence_dataset, model_path, output,
                        settings=AlignmentTrainingSettings(), device='cuda', resume=None, budget_only=False):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import transformers

    settings.validate()
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    device = resolve_device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA explicitly requested but unavailable; no CPU fallback')
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == 'cuda' else torch.float32
    model_path = Path(model_path).resolve()
    identity = {'model_files_sha256': model_fingerprint(model_path),
                'alignment_manifest_sha256': file_hash(Path(alignment_dataset) / 'manifest.json'),
                'sensor_manifest_sha256': file_hash(Path(evidence_dataset) / 'manifest.json')}
    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    json_write(destination / 'run_config.json',
               {'settings': asdict(settings), 'identity': identity, 'alignment_dataset': str(Path(alignment_dataset).resolve()),
                'evidence_dataset': str(Path(evidence_dataset).resolve()), 'model_path': str(model_path),
                'device': str(device), 'dtype': str(dtype), 'microbatch_size': 1, 'gradient_checkpointing': True,
                'attention_implementation': 'sdpa', 'resume': str(resume) if resume else None, 'budget_only': budget_only,
                'torch': str(torch.__version__), 'transformers': transformers.__version__,
                'cuda_version': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                'source_hashes': {name: file_hash(Path(__file__).parent / name) for name in
                                  ('alignment_training.py', 'sensor_evidence.py', 'alignment_data.py',
                                   'llm_bridge.py', 'embedding_bridge.py')}})
    budget = audit_token_budget(alignment_dataset, evidence_dataset, tokenizer, settings, config.max_position_embeddings)
    json_write(destination / 'token_budget.json', budget)
    if not budget['passed']:
        json_write(destination / 'report.json', {'status': 'token_budget_exceeded', 'optimizer_steps': 0,
                                                'over_budget_examples': budget['over_budget_examples']})
        return {'status': 'token_budget_exceeded', 'optimizer_steps': 0}
    if budget_only:
        json_write(destination / 'report.json', {'status': 'budget_only_passed', 'optimizer_steps': 0})
        return {'status': 'budget_only_passed', 'optimizer_steps': 0}
    data = SensorAlignmentExamples(alignment_dataset, evidence_dataset, 'train', settings.history_seconds)
    validation = SensorAlignmentExamples(alignment_dataset, evidence_dataset, 'validation', settings.history_seconds)
    order = training_order(len(data), settings.epochs, settings.seed)
    first_index = order[0]
    json_write(destination / 'first_training_example.json',
               {'sample_id': data.rows[first_index]['id'], 'question': data.question(first_index),
                'supervision_only_answer': data.answer(first_index),
                'note': 'answer is a separate causal teacher-forcing target, not part of user question'})
    step, cursor, saved_step = 0, 0, 0
    progress = {}
    phase, projector, optimizer = 'load_llm', None, None
    started = time.monotonic()
    status, failure = 'completed', None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                                                    torch_dtype=dtype, attn_implementation='sdpa').to(device)
        configure_frozen_llm(model)
        encoder = data.manifest['encoder_settings']
        projector = MapEmbeddingProjector(model.get_input_embeddings().weight.shape[1], encoder['hidden_dim'],
                                          encoder['latent_tokens']).to(device)
        optimizer = torch.optim.AdamW(projector.parameters(), lr=settings.learning_rate)
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and dtype == torch.float16)
        if resume:
            step, cursor = restore_training_checkpoint(resume, projector, optimizer, scaler,
                                                       identity, settings, device)
        saved_step = step
        save_training_checkpoint(destination / 'last.pt', projector, optimizer, scaler, step, cursor, identity, settings)
        json_write(destination / 'trainable_parameters.json',
                   {'llm_trainable': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                    'projector_and_time_trainable': sum(parameter.numel() for parameter in projector.parameters()),
                    'map_encoder_loaded_for_training': False, 'map_tokens_frozen': True})
        with (destination / 'history.jsonl').open('x', encoding='utf-8') as history:
            while cursor < len(order) and (settings.max_steps is None or step < settings.max_steps):
                group = order[cursor:cursor + settings.gradient_accumulation]
                phase = 'forward_backward_optimizer'
                metrics = update_projector(model, tokenizer, projector, data, group, optimizer, scaler,
                                            settings, device, dtype, progress)
                step += 1
                cursor += len(group)
                metrics.update({'optimizer_step': step, 'samples_seen': cursor,
                                'sample_ids': [data.rows[index]['id'] for index in group]})
                history.write(json.dumps(metrics, allow_nan=False) + '\n')
                history.flush()
                print(f'投影训练 step={step} samples={cursor} loss={metrics["loss"]:.4f}', flush=True)
                if step % settings.checkpoint_every == 0:
                    phase = 'save_checkpoint'
                    save_training_checkpoint(destination / 'last.pt', projector, optimizer, scaler,
                                             step, cursor, identity, settings)
                    saved_step = step
            phase = 'save_checkpoint'
            save_training_checkpoint(destination / 'last.pt', projector, optimizer, scaler,
                                     step, cursor, identity, settings)
            saved_step = step
            phase = 'validation_teacher_forcing'
            result = validation_loss(model, tokenizer, projector, validation, settings, device, dtype)
            json_write(destination / 'validation.json', result)
            if cursor < len(order):
                status = 'step_limit_reached'
    except Exception as error:
        if not is_cuda_oom(error):
            raise
        status = 'cuda_oom'
        failure = {'phase': phase, 'message': str(error), 'traceback': traceback.format_exc(),
                   'active_microbatch': progress.copy(),
                   'completed_optimizer_steps': step, 'saved_optimizer_steps': saved_step,
                   'resume_checkpoint': str(destination / 'last.pt') if (destination / 'last.pt').exists() else None,
                   'note': 'stop without retry, truncation or CPU fallback; resume only from saved checkpoint'}
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        json_write(destination / 'oom.json', failure)
        print('CUDA显存不足，已停止；保留长度审核、OOM现场及已保存checkpoint。', flush=True)
    report = {'status': status, 'optimizer_steps': step, 'saved_optimizer_steps': saved_step,
              'samples_seen': cursor, 'dataset_examples': len(data), 'elapsed_seconds': time.monotonic() - started,
              'llm_weights_frozen': True, 'encoder_retrained': False, 'rl_started': False,
              'full_training_schedule_completed': cursor == len(order),
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
              'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
              'failure_phase': failure['phase'] if failure else None,
              'semantic_map_understanding_validated': False}
    json_write(destination / 'report.json', report)
    return report
