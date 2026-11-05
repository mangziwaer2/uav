import pytest

torch = pytest.importorskip('torch')

from uav_em.map_learning.embedding_bridge import MapEmbeddingProjector, insert_map_embeddings


def bridge_example():
    projector = MapEmbeddingProjector(llm_dim=24, hidden_dim=16)
    tokens = torch.randn(2, 4, 32, 96)
    valid = torch.tensor([[False, True, True, True], [True, True, True, True]])
    offsets = torch.tensor([[0., -12., -6., 0.], [-18., -12., -6., 0.]])
    return projector, tokens, valid, offsets


def test_four_frames_produce_128_embedding_positions_with_padding_mask():
    projector, tokens, valid, offsets = bridge_example()
    tokens[0, 0] = float('nan')
    embedded, mask = projector(tokens, valid, offsets)
    assert embedded.shape == (2, 128, 24)
    assert mask.sum(1).tolist() == [96, 128]
    assert torch.isfinite(embedded).all()
    assert torch.all(embedded[0, :32] == 0)
    embedded.sum().backward()
    assert projector.project[-1].weight.grad.abs().sum() > 0
    assert projector.time[-1].weight.grad.abs().sum() > 0


def test_actual_time_changes_embeddings_but_cannot_change_other_frames():
    projector, tokens, valid, offsets = bridge_example()
    first, _ = projector(tokens, valid, offsets)
    offsets[:, 1] = -13
    second, _ = projector(tokens, valid, offsets)
    assert not torch.equal(first[:, 32:64], second[:, 32:64])
    assert torch.equal(first[:, 64:], second[:, 64:])


@pytest.mark.parametrize('failure', ['future', 'duplicate', 'unordered', 'missing_current', 'nan'])
def test_invalid_causal_frame_metadata_rejected(failure):
    projector, tokens, valid, offsets = bridge_example()
    if failure == 'future':
        offsets[:, 1] = 2
    elif failure == 'duplicate':
        offsets[:, 1] = offsets[:, 2]
    elif failure == 'unordered':
        offsets[:, 1] = -3
    elif failure == 'missing_current':
        valid[:, -1] = False
    else:
        offsets[:, 1] = float('nan')
    with pytest.raises(ValueError):
        projector(tokens, valid, offsets)


def test_inserted_maps_receive_no_text_labels_and_padding_is_ignored():
    projector, tokens, valid, offsets = bridge_example()
    maps, map_valid = projector(tokens, valid, offsets)
    text = torch.randn(2, 5, 24)
    text_valid = torch.tensor([[True, True, True, True, False], [True, True, True, True, True]])
    labels = torch.tensor([[-100, -100, -100, 10, -100], [-100, -100, -100, 11, 12]])
    result = insert_map_embeddings(text, text_valid, maps, map_valid, 2, labels)
    assert result['inputs_embeds'].shape == (2, 133, 24)
    assert torch.equal(result['inputs_embeds'][:, :2], text[:, :2])
    assert torch.all(result['labels'][:, 2:130] == -100)
    assert result['labels'][0, 131] == 10
    assert result['labels'][0, -1] == -100
    assert result['attention_mask'].sum(1).tolist() == [100, 133]
    assert result['position_ids'][0, 131] == 99
    result['inputs_embeds'].sum().backward()
    assert projector.project[-1].weight.grad is not None


def test_bridge_requires_explicit_matching_llm_dimension_and_dtype():
    projector, tokens, valid, offsets = bridge_example()
    maps, mask = projector(tokens, valid, offsets)
    with pytest.raises(ValueError, match='dimensions'):
        insert_map_embeddings(torch.randn(2, 2, 25), torch.ones(2, 2, dtype=torch.bool), maps, mask, 1)
    with pytest.raises(ValueError, match='cast'):
        insert_map_embeddings(torch.randn(2, 2, 24, dtype=torch.float64), torch.ones(2, 2, dtype=torch.bool), maps, mask, 1)
