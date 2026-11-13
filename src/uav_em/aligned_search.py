"""研究者编写的C一致RRT*搜索原语；不改变观测、GP或共同接受器。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import math
import time
import numpy as np

from .balanced_objective import (BalancedOptimizer, PowerTimeSettings, _BalancedSession,
    choose, normalized_expected_power, paired_seed, power_time_score)
from .decision_objective import ObjectiveSettings
from .planning import PlanResult, path_length
from .repair_diagnostics import search_proposal
from .repair_process import canonical_path
from .repair_search import FixedObjective, repair_rrt_star
from .route_repair import anchors, parse_proposal

VERSION = 'aligned_search_primitive_v1'
OBJECTIVES = ('old_J', 'power_time_C')


class PowerTimeSearchObjective:
    """精确逐格积分 C_edge=∫(1+λ E[P/P0]) ds/v；缓存与预算为决策内共享。"""
    def __init__(self, snapshot, speed, budget, settings=PowerTimeSettings()):
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError('positive finite speed required')
        self.snapshot, self.geometry, self.budget = snapshot, snapshot.geometry, budget
        self.speed, self.settings, self.cache = speed, settings, {}
        self.integrator = FixedObjective(snapshot, settings.scale_dbm, budget)
        started = time.perf_counter()
        mask = snapshot.supported
        prior = float(normalized_expected_power(settings.prior_mean_dbm, settings.prior_std_db, settings.scale_dbm))
        self.q = np.full(mask.shape, prior)
        self.q[mask] = normalized_expected_power(snapshot.estimate_dbm[mask], snapshot.uncertainty_db[mask], settings.scale_dbm)
        self.grid_nodes_built, self.field_nodes = self.q.size, 0
        self.grid_build_seconds = time.perf_counter()-started

    def edge_cost(self, start, end):
        key = tuple(sorted((tuple(map(float, start)), tuple(map(float, end)))))
        if key in self.cache:
            self.budget.cache_hits += 1
            return self.cache[key]
        self.budget.take_edge()
        value = math.inf
        if self.geometry.edge_free(start, end):
            weights, nodes = self.integrator.quadrature(start, end)
            self.field_nodes += len(nodes)
            value = float(weights @ (1 + self.settings.weight*self.q[nodes[:,1], nodes[:,0]]) / self.speed)
        self.cache[key] = value
        return value


class PrimitiveSession(_BalancedSession):
    def __init__(self, *args, search_objective, profile, **kwargs):
        if search_objective not in OBJECTIVES:
            raise ValueError('unknown search objective')
        super().__init__(*args, **kwargs)
        self.search_objective, self.profile = search_objective, profile
        # Both arms use a separate empty search cache; acceptor's old-J cache is separate.
        self.search_problem = (PowerTimeSearchObjective(self.snapshot, self.speed, self.work, self.score_settings)
            if search_objective == 'power_time_C' else FixedObjective(self.snapshot, self.problem.reference, self.work))

    def search(self, a, b, quota, sampling, purpose):
        key = self.profile+'::'+canonical_path(self.best)+'::'+purpose
        seed = paired_seed(self.scene_seed, self.snapshot.time_seconds, VERSION, key)
        reserve = self.work.reserved_edges
        self.work.reserved_edges += min(128, max(0, (self.work.edge_limit-self.work.edges-reserve)//2))
        before = self.work.summary()
        try:
            result, stats = repair_rrt_star(self.search_problem, a, b, quota, seed, sampling)
        finally:
            self.work.reserved_edges = reserve
        self.searches.append(dict(purpose=purpose, sampling=deepcopy(sampling), sampling_counts=stats,
            budget_before=before, budget_after=self.work.summary(), **asdict(result), actual_seed=seed,
            seed_stage=VERSION, seed_purpose=key, search_objective=self.search_objective,
            search_cost_unit='seconds' if self.search_objective=='power_time_C' else 'old_J_research_m',
            search_cache_separate_from_acceptor=True))
        return result.path


def run_primitive(snapshot, context, pool_report, maximum, scene_seed, objective, profile='local_pad1',
                  settings=PowerTimeSettings(), limits=ObjectiveSettings()):
    started = time.perf_counter()
    chosen = choose(pool_report['candidates'])
    session = PrimitiveSession(snapshot, context, anchors(chosen['path'],80.), maximum, limits,
        scene_seed, pool_report, settings, 'balanced', search_objective=objective, profile=profile)
    initial = deepcopy(session.work.summary())
    ctx = session.request(1,1)
    proposal = None
    if session.work.summary()['samples_left'] and session.work.summary()['edges_left'] > len(session.best)-1:
        proposal = parse_proposal(search_proposal(ctx,profile,min(700,session.work.summary()['samples_left'])),ctx)
        feedback = session.apply(proposal)
    else:
        feedback = dict(accepted=False, reason='insufficient_shared_resources')
    session.trace.append(dict(context=ctx, proposal=proposal, feedback=feedback, error=None))
    session.work.reserved_edges = 0
    for a,b in zip(session.best,session.best[1:]):
        session.work.take_edge(); session.work.verification_edges += 1
        if not snapshot.geometry.edge_free(a,b):
            raise ValueError('final geometry invalid')
    selected = choose(session.records)
    assert canonical_path(selected['path']) == canonical_path(session.best)
    assert selected['metrics']['length_m'] <= maximum+1e-7
    elapsed = time.perf_counter()-started
    return dict(contract_version=VERSION, search_objective=objective, profile=profile, path=session.best,
        candidates=session.records, trace=session.trace, searches=session.searches, resources=session.work.summary(),
        starting_resources_already_paid=deepcopy(pool_report['resources']), initial_resources=initial,
        selected_candidate_id=selected['id'], selected_balance=selected['balance'], selected_prefix=selected['prefix'],
        best_metrics=selected['metrics'], final_geometry_verified=True, policy_errors=0, actual_advisor_calls=0,
        new_measurements=0, model_calls=0, score_field_nodes=session.score_nodes,
        grid_nodes_built=getattr(session.search_problem,'grid_nodes_built',0),
        search_field_nodes=getattr(session.search_problem,'field_nodes',None),
        grid_build_seconds=getattr(session.search_problem,'grid_build_seconds',0),
        numeric_compute_seconds=elapsed, primitive_accepted_proposals=session.accepted_proposals)


class AlignedOptimizer:
    policy = None
    last_call = {}
    def __init__(self, scene_seed, objective, settings=PowerTimeSettings(), limits=ObjectiveSettings()):
        if objective not in OBJECTIVES:
            raise ValueError('unknown search objective')
        self.base = BalancedOptimizer(scene_seed,settings,limits=limits)
        self.scene_seed,self.objective,self.settings,self.limits=scene_seed,objective,settings,limits

    def optimize(self,snapshot,context,route,max_length_m,seed,call_limit):
        started=time.perf_counter()
        result,report=self.base.optimize(snapshot,context,route,max_length_m,seed,0)
        if not result.path:return result,report
        maximum=min(max_length_m,report['mission_distance_cap_m']-report['executed_distance_m'])
        extra=run_primitive(snapshot,context,report,maximum,self.scene_seed,self.objective,settings=self.settings,limits=self.limits)
        searches=report['searches']+extra['searches']
        trace=report['trace']+extra['trace']
        score_nodes=report['score_field_nodes']+extra['score_field_nodes']
        common=deepcopy(report['resources'])
        report.update(extra, searches=searches, trace=trace, score_field_nodes=score_nodes,
            primitive_initial_resources=common, selection_rule='balanced',
            accepted_proposals=int(not route or canonical_path(route)!=canonical_path(extra['path'])),
            numeric_compute_seconds=time.perf_counter()-started)
        result.path=extra['path'];result.length_m=path_length(result.path)
        result.cost=extra['selected_balance']['score_seconds']
        result.iterations=extra['resources']['samples'];result.elapsed_seconds=report['numeric_compute_seconds']
        return result,report
