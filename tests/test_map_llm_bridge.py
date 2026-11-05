from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip('torch')

from uav_em.map_learning.alignment_data import MAP_MARKER
from uav_em.map_learning.embedding_bridge import MapEmbeddingProjector
from uav_em.map_learning.llm_bridge import prepare_llm_inputs, prompt_token_parts


class TemplateTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False}
        return 'SYSTEM:' + messages[0]['content'] + '\nUSER:' + messages[1]['content'] + '\nASSISTANT:'

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        assert MAP_MARKER not in text
        return [ord(character) % 120 for character in text]


def example_input():
    return {'tokens': torch.randn(4, 4, 16), 'valid': torch.tensor([False, True, True, True]),
            'offset_seconds': torch.tensor([0., -12., -6., 0.]),
            'question': 'history\n' + MAP_MARKER + '\ncompare ROI'}


def test_template_replaces_marker_without_text_tokenizing_map_vectors():
    before, after, rendered = prompt_token_parts(TemplateTokenizer(), example_input()['question'])
    assert before and after
    assert rendered.count(MAP_MARKER) == 1


@pytest.mark.parametrize('question', ['no maps here', MAP_MARKER + MAP_MARKER])
def test_missing_or_repeated_map_marker_rejected(question):
    with pytest.raises(ValueError, match='exactly one'):
        prompt_token_parts(TemplateTokenizer(), question)


def test_real_tiny_qwen_forward_and_gradient_to_projector_without_llm_update():
    transformers = pytest.importorskip('transformers')
    config = transformers.Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                                      head_dim=8, max_position_embeddings=1024, attention_dropout=0)
    model = transformers.Qwen3ForCausalLM(config).eval().requires_grad_(False)
    projector = MapEmbeddingProjector(32, encoder_dim=16, latent_tokens=4, hidden_dim=16)
    inputs, metadata = prepare_llm_inputs(model, TemplateTokenizer(), projector, example_input(), 1024)
    assert inputs['inputs_embeds'].shape[1] == metadata['text_positions'] + 16
    assert metadata['map_positions'] == 16
    assert metadata['valid_map_positions'] == 12
    initial = model.get_input_embeddings().weight.detach().clone()
    output = model(**inputs, use_cache=False)
    assert torch.isfinite(output.logits).all()
    output.logits[:, -1].square().mean().backward()
    assert projector.project[-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())
    assert torch.equal(initial, model.get_input_embeddings().weight)
    with pytest.raises(ValueError, match='no truncation'):
        prepare_llm_inputs(model, TemplateTokenizer(), projector, example_input(), 1)


@pytest.mark.parametrize('entry', ['build_map_alignment_dataset.py', 'check_map_llm_bridge.py', 'audit_map_alignment_dataset.py'])
def test_alignment_tools_help_is_lazy_and_works_outside_root(entry):
    script = Path(__file__).resolve().parents[1] / 'tools' / entry
    result = subprocess.run([sys.executable, str(script), '--help'], cwd=script.parent,
                            capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert b'--dataset' in result.stdout
