"""既有两份结构程序的分块归因；不生成新表达式或修改接受器。"""
from copy import deepcopy
from itertools import product
import math

from .structural_policy import validate_policy

BLOCKS = {
    'entry': ('entry_score', 'entry_offset'),
    'exit': ('exit_score', 'max_arms'),
    'schedule': ('priority', 'sample_quota', 'edge_quota', 'stop', 'operation'),
}
COMBINATIONS = tuple(''.join(bits) for bits in product('DR', repeat=3))


def compose_program(draft, revision, combination):
    """D=原第二初稿，R=原第二修订；三位依次为入口/接回/调度。"""
    if combination not in COMBINATIONS:
        raise ValueError('exactly three D/R block selectors required')
    parents = {'D': validate_policy(draft), 'R': validate_policy(revision)}
    result = deepcopy(parents['D'])
    for bit, fields in zip(combination, BLOCKS.values()):
        for key in fields:
            result[key] = deepcopy(parents[bit][key])
    result['name'] = 'researcher_block_recombination_' + combination
    return validate_policy(result)


def paired_block_contrasts(rows, metric):
    """完整2³设计内的逐对R−D差；不将两个状态当独立重复或显著性样本。"""
    mapping = {}
    for row in rows:
        code, value = row['combination'], row[metric]
        if code not in COMBINATIONS or code in mapping:
            raise ValueError('unique complete factorial rows required')
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError('finite numeric contrast required')
        mapping[code] = value
    if set(mapping) != set(COMBINATIONS):
        raise ValueError('all eight cells required; no missing-cell average')
    output = {}
    for axis, block in enumerate(BLOCKS):
        pairs = []
        for code in COMBINATIONS:
            if code[axis] != 'D':
                continue
            other = code[:axis] + 'R' + code[axis+1:]
            pairs.append(dict(draft_blocks=code, revision_blocks=other,
                              delta_R_minus_D=mapping[other] - mapping[code]))
        output[block] = dict(pairs=pairs, mean_delta_R_minus_D=sum(p['delta_R_minus_D'] for p in pairs)/4,
                             minimum_delta=min(p['delta_R_minus_D'] for p in pairs),
                             maximum_delta=max(p['delta_R_minus_D'] for p in pairs))
    return output
