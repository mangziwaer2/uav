"""固定生成过程的截断诊断；不生成新算法，不改变动作/目标/分配器。"""
from copy import deepcopy

from .decision_objective import select_candidate
from .offline_process import validate_program


PREFIX_BOUNDARIES = {'local_prefix': 2, 'via_prefix': 4, 'global_prefix': 5, 'full_process': 6}


def diagnostic_prefix(program, mode):
    """保留全部节点，避免剩余搜索节点数改变feedback额度；只截断跳转。"""
    original = validate_program(program)
    if mode not in PREFIX_BOUNDARIES or len(original['nodes']) != 6:
        raise ValueError('six-node parent and declared diagnostic mode required')
    boundary = PREFIX_BOUNDARIES[mode]
    result = deepcopy(original)
    for node in result['nodes']:
        node['next'] = {key: 'stop' if type(target) is int and target >= boundary else target
                        for key, target in node['next'].items()}
    return validate_program(result)


def check_identical_executed_prefix(full_report, prefix_report):
    """忽略最后被截断的next；数值提议、状态和实际工作必须逐位一致。"""
    if full_report['common_pool_resources'] != prefix_report['common_pool_resources']:
        raise ValueError('initial pool work differs')
    full = full_report['process_trace']
    prefix = prefix_report['process_trace']
    if len(prefix) > len(full):
        raise ValueError('diagnostic trace is longer than parent')
    fields = ('node', 'state', 'proposal', 'construction', 'feedback', 'outcome', 'work_delta', 'error')
    for parent, intervention in zip(full, prefix):
        if any(parent[field] != intervention[field] for field in fields):
            raise ValueError('prefix changed proposals, feedback or resource allocation')
    return {'identical_prefix_nodes': len(prefix), 'unused_budget_redistributed': False,
            'same_state_control_only': True}


def selection_comparison(records):
    """同一冻结池的两个预定选择规则；不读取真值，不做搜索。"""
    risk = select_candidate(records, 'prefix_v2')
    old_j = select_candidate(records, 'pool_j')
    if risk is None or old_j is None:
        raise ValueError('at least one feasible candidate required')
    def summary(record):
        return {'id': record['id'], 'metrics': deepcopy(record['metrics']),
                'prefix': deepcopy(record['prefix']), 'path': deepcopy(record['path'])}
    initial_count = next((i for i, r in enumerate(records) if r['id'].startswith('process_candidate_')), len(records))
    return {'same_frozen_pool': True, 'same_candidate_count': len(records),
            'initial_candidate_count': initial_count,
            'feasible_generated_candidate_count': sum(r['feasible'] for r in records[initial_count:]),
            'selected_ids_differ': risk['id'] != old_j['id'],
            'risk_choice': summary(risk), 'old_j_choice': summary(old_j),
            'search_objective_changed': False, 'new_measurements_or_truth_used': False,
            'not_a_search_objective_ablation': True}
