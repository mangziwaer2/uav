from copy import deepcopy
import json

import pytest

from uav_em import constraint_replay as replay


def read(path):return json.loads(path.read_text())


def test_restored_seed_is_pre_optimization_and_keeps_original_causal_inputs():
    record=read(replay.M5/'runs/allocation_N/00.json')
    old=next(f for r in record['rounds'] for f in r['fits'] if 'geometry_refinement' in f)
    before=deepcopy(old);seed=replay.restore_seed(old);trace=old['geometry_refinement']
    assert old==before and seed['model']==trace['seed_model'] and 'geometry_refinement' not in seed
    assert seed['fit_objective']==trace['previous_objective']
    assert seed['model']!=old['model']
    assert seed['window']==old['window'] and seed['job']==old['job'] and seed['gp_soft_constraints']==old['gp_soft_constraints']
    assert seed['fit_objective']==pytest.approx(seed['model']['fit_mse']+.25*seed['gp_shape_mse'])


def test_only_five_original_targets_recomputed_and_keep_is_unchanged(monkeypatch):
    calls=[]
    def refine(seed,public):
        calls.append(deepcopy(seed));result=deepcopy(seed)
        result['model']['peaks_dbm'][0]-=1.
        result['geometry_refinement']=dict(accepted=True)
        return result
    monkeypatch.setattr(replay,'constrained_refine_fit',refine)
    for method in replay.METHODS:
        for i in range(3):
            packet=read(replay.M5/'inputs'/f'{i:02d}.json');old=read(replay.M5/'runs'/method/f'{i:02d}.json');before=deepcopy(old)
            fits,models,committed,updates=replay.replay_fits(packet,old)
            assert old==before and committed==old['pool_committed']
            assert len(updates)==int(old['pool_committed'])
            if not committed:assert models==packet['models']
            else:
                assert sum(m['weight'] for m in models)==pytest.approx(1.)
                assert [m['weight'] for m in models]==[m['weight'] for m in old['source_models']]
                for m in models[len(packet['prior_models']):]:assert m['scored_points']==0
            originals=[f for r in old['rounds'] for f in r['fits']]
            for a,b in zip(originals,fits):
                if 'geometry_refinement' not in a:assert a==b
    assert len(calls)==5
