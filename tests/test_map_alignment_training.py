from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')

from test_map_sensor_evidence import evidence_data
from uav_em.map_learning.alignment_data import MAP_MARKER
from uav_em.map_learning.alignment_training import (AlignmentTrainingSettings, answer_token_ids,
                                                    audit_token_budget, configure_frozen_llm, is_cuda_oom,
                                                    supervised_inputs, train_map_alignment, update_projector)
from uav_em.map_learning.alignment_training import restore_training_checkpoint, save_training_checkpoint
from uav_em.map_learning.data import file_hash
from uav_em.map_learning.embedding_bridge import MapEmbeddingProjector
from uav_em.map_learning.sensor_evidence import SensorAlignmentExamples, build_sensor_evidence


class ShortTokenizer:
    eos_token_id = 127

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False}
        return 'SYSTEM:' + messages[0]['content'] + '\nUSER:' + messages[1]['content'] + '\nASSISTANT:'

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        assert MAP_MARKER not in text
        return [ord(text[index]) % 120 for index in range(0, len(text), 8)]


def tiny_qwen():
    config = transformers.Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                                      head_dim=8, max_position_embeddings=4096, attention_dropout=0)
    return transformers.Qwen3ForCausalLM(config)


def test_answer_supervision_masks_prompt_maps_and_padding_but_includes_eos():
    model = tiny_qwen().requires_grad_(False)
    projector = MapEmbeddingProjector(32, 16, 4, hidden_dim=16)
    example = {'tokens': torch.randn(3, 4, 16), 'valid': torch.tensor([False, True, True]),
               'offset_seconds': torch.tensor([0., -2., 0.]), 'question': 'history\n' + MAP_MARKER + '\nROI'}
    answer = {'power': [[-80.123, None]], 'age': [1, None]}
    inputs, metadata = supervised_inputs(model, ShortTokenizer(), projector, example, answer, 4096)
    start = metadata['prompt_positions']
    assert inputs['inputs_embeds'].requires_grad
    assert torch.all(inputs['labels'][0, :start] == -100)
    assert inputs['labels'][0, start:].tolist() == answer_token_ids(ShortTokenizer(), answer)
    assert inputs['labels'][0, -1] == 127
    assert torch.all(inputs['labels'][inputs['attention_mask'] == 0] == -100)
    assert inputs['position_ids'][0, -1] == inputs['attention_mask'].sum() - 1
    output = model(**inputs, use_cache=False)
    expected = torch.nn.functional.cross_entropy(output.logits[:, :-1].reshape(-1, 128),
                                                  inputs['labels'][:, 1:].reshape(-1))
    assert torch.allclose(output.loss, expected)
    with pytest.raises(ValueError, match='no truncation'):
        supervised_inputs(model, ShortTokenizer(), projector, example, answer, metadata['total_positions'] - 1)


def test_real_checkpointed_backward_only_updates_projector_and_time():
    model = tiny_qwen()
    configure_frozen_llm(model)
    assert model.is_gradient_checkpointing
    initial_llm = {key: value.clone() for key, value in model.state_dict().items()}
    projector = MapEmbeddingProjector(32, 16, 4, hidden_dim=16)
    initial = {key: value.clone() for key, value in projector.state_dict().items()}

    class Examples:
        def model_input(self, index):
            return {'tokens': torch.randn(2, 4, 16), 'valid': torch.tensor([True, True]),
                    'offset_seconds': torch.tensor([-2., 0.]), 'question': 'map\n' + MAP_MARKER + '\nROI'}

        def answer(self, index):
            return {'power': -80.123, 'supported': 1}

    settings = AlignmentTrainingSettings(gradient_accumulation=2, max_positions=4096)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    metrics = update_projector(model, ShortTokenizer(), projector, Examples(), [0, 1], optimizer, scaler,
                                settings, torch.device('cpu'), torch.float32)
    assert metrics['gradient_norm'] > 0
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in initial_llm.items())
    assert not torch.equal(initial['project.3.weight'], projector.state_dict()['project.3.weight'])
    assert not torch.equal(initial['time.2.weight'], projector.state_dict()['time.2.weight'])


@pytest.fixture
def training_paths(evidence_data, tmp_path, monkeypatch):
    source, alignment = evidence_data
    evidence = tmp_path / 'evidence'
    build_sensor_evidence(alignment, source, evidence)
    model_path = tmp_path / 'model'
    tiny_qwen().save_pretrained(model_path)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: ShortTokenizer())
    return alignment, evidence, model_path


def test_budget_counts_both_splits_and_refuses_long_answers(training_paths):
    alignment, evidence, _ = training_paths
    settings = AlignmentTrainingSettings(max_positions=4096)
    report = audit_token_budget(alignment, evidence, ShortTokenizer(), settings, 4096)
    assert report['passed']
    assert report['splits']['train']['examples'] == 1
    assert report['splits']['validation']['examples'] == 1
    assert report['includes_complete_answers_and_eos']
    failed = audit_token_budget(alignment, evidence, ShortTokenizer(), replace(settings, max_positions=1), 4096)
    assert not failed['passed']
    assert failed['over_budget_examples'] == 2
    with pytest.raises(ValueError, match='context limit'):
        audit_token_budget(alignment, evidence, ShortTokenizer(), settings, 4)


def test_complete_training_and_resume_preserve_frozen_model_and_identity(training_paths, tmp_path):
    alignment, evidence, model_path = training_paths
    weights = model_path / 'model.safetensors'
    original_hash = file_hash(weights)
    settings = AlignmentTrainingSettings(epochs=2, gradient_accumulation=1, max_positions=4096,
                                         checkpoint_every=1, validation_examples=1, max_steps=1)
    first = tmp_path / 'first'
    result = train_map_alignment(alignment, evidence, model_path, first, settings, 'cpu')
    assert result['status'] == 'step_limit_reached'
    assert result['optimizer_steps'] == 1
    assert not result['full_training_schedule_completed']
    continued = tmp_path / 'continued'
    result = train_map_alignment(alignment, evidence, model_path, continued, replace(settings, max_steps=2),
                                  'cpu', resume=first / 'last.pt')
    assert result['status'] == 'completed'
    assert result['optimizer_steps'] == 2
    assert result['full_training_schedule_completed']
    full = tmp_path / 'full'
    train_map_alignment(alignment, evidence, model_path, full, replace(settings, max_steps=2), 'cpu')
    resumed = torch.load(continued / 'last.pt', weights_only=True)
    direct = torch.load(full / 'last.pt', weights_only=True)
    assert all(torch.equal(value, direct['projector'][key]) for key, value in resumed['projector'].items())
    assert file_hash(weights) == original_hash
    assert json.loads((continued / 'trainable_parameters.json').read_text())['llm_trainable'] == 0
    with pytest.raises(ValueError, match='settings mismatch'):
        train_map_alignment(alignment, evidence, model_path, tmp_path / 'wrong_resume',
                              replace(settings, learning_rate=0.01), 'cpu', resume=first / 'last.pt')


def test_oom_stops_without_retry_and_keeps_initial_resume_checkpoint(training_paths, tmp_path, monkeypatch):
    alignment, evidence, model_path = training_paths
    calls = []

    def oom(*args):
        calls.append(True)
        raise torch.cuda.OutOfMemoryError('synthetic unit test OOM, not hardware evidence')

    monkeypatch.setattr('uav_em.map_learning.alignment_training.update_projector', oom)
    output = tmp_path / 'oom'
    result = train_map_alignment(alignment, evidence, model_path, output,
                                  AlignmentTrainingSettings(max_positions=4096), 'cpu')
    assert result['status'] == 'cuda_oom'
    assert result['optimizer_steps'] == 0
    assert len(calls) == 1
    assert (output / 'last.pt').is_file()
    assert json.loads((output / 'oom.json').read_text())['resume_checkpoint']
    assert not (output / 'validation.json').exists()


def test_training_help_is_lazy_and_independent_of_working_directory():
    script = Path(__file__).resolve().parents[1] / 'tools/train_map_alignment.py'
    result = subprocess.run([sys.executable, str(script), '--help'], cwd=script.parent, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert b'--evidence-dataset' in result.stdout
    assert b'--resume' in result.stdout


def test_cuda_oom_classification_handles_accelerator_error_without_hiding_other_errors():
    assert is_cuda_oom(torch.cuda.OutOfMemoryError('out of memory'))
    if hasattr(torch, 'AcceleratorError'):
        assert is_cuda_oom(torch.AcceleratorError('CUDA error: out of memory'))
        assert not is_cuda_oom(torch.AcceleratorError('CUDA error: illegal memory access'))
    assert not is_cuda_oom(ValueError('out of memory'))
    assert not is_cuda_oom(RuntimeError('ordinary failure'))


def test_accelerator_oom_also_records_failure_without_retry(training_paths, tmp_path, monkeypatch):
    if not hasattr(torch, 'AcceleratorError'):
        pytest.skip('AcceleratorError is unavailable in this torch version')
    alignment, evidence, model_path = training_paths

    def oom(*args):
        raise torch.AcceleratorError('CUDA error: out of memory; synthetic test')

    monkeypatch.setattr('uav_em.map_learning.alignment_training.update_projector', oom)
    output = tmp_path / 'accelerator_oom'
    result = train_map_alignment(alignment, evidence, model_path, output,
                                  AlignmentTrainingSettings(max_positions=4096), 'cpu')
    assert result['status'] == 'cuda_oom'
    assert result['saved_optimizer_steps'] == 0
    assert 'AcceleratorError' in json.loads((output / 'oom.json').read_text())['traceback']


def test_resume_empty_scaler_state_keeps_new_scaler_initialization(tmp_path):
    projector = MapEmbeddingProjector(32, 16, 4, hidden_dim=16)
    optimizer = torch.optim.AdamW(projector.parameters())
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    settings = AlignmentTrainingSettings()
    path = tmp_path / 'last.pt'
    save_training_checkpoint(path, projector, optimizer, scaler, 0, 0, {'model': 'fixed'}, settings)

    class NewScaler:
        def load_state_dict(self, state):
            raise AssertionError('empty BF16 scaler state must not overwrite fresh FP16 scaler')

    assert restore_training_checkpoint(path, projector, optimizer, NewScaler(), {'model': 'fixed'},
                                       settings, torch.device('cpu')) == (0, 0)
