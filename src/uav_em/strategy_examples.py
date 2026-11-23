"""Artificial finite corridor-policy examples with the real 1.25 D0 limit.

Every field below is an invented, declared model parameter. This module neither
reads scientific results nor imports a simulator, sensor or hidden scenario.
Its examples test the common numerical policy tools, never an LLM advantage.
"""
from copy import deepcopy
import hashlib
import json
import math


def _base(center, *, scale, velocity, peak=-67., rate=.05):
    return dict(center_m=list(center), scale_m=scale, peak_dbm=peak,
        velocity_mps=list(velocity), rate_db_per_second=rate,
        reference_time_seconds=0.)


def _world(identifier, center, *, scale, velocity, weight=.5):
    return dict(id=identifier, weight=weight, background_dbm=-100.,
                bases=[_base(center, scale=scale, velocity=velocity)])


def _body(start, *, history_dbm, worlds, obstacles):
    goal = [180., 0.]
    return dict(version='corridor-task-v1', now_seconds=0.,
        start_m=list(start), goal_m=goal, speed_mps=15.,
        sensor_interval_seconds=1., measurement_noise_db=1.,
        bounds_m=[[-10., -80.], [200., 80.]], obstacles_m=obstacles,
        spent_length_m=0., initial_distance_m=math.dist(start, goal),
        remaining_energy_wh=17., reserve_energy_wh=15., flight_power_w=350.,
        history=[dict(time_seconds=0., position_m=list(start),
                      interference_dbm=history_dbm)],
        worlds=worlds, depth=3, heading_degrees=[-20., 0., 20.],
        corner_offset_m=.25)


def build_examples():
    """Return fresh task *bodies*, before the common paid preparation.

    A already has informative readings on an ordinary forward route. In B two
    short entrance walls block both oblique headings during the first three
    normal intervals; mirror worlds then have identical likelihoods at every
    reachable measurement. Its central building preserves two later corridors.
    The single-world control removes uncertainty about the declared field.

    D0 is the initial straight-line start-to-goal distance, including in the
    obstructed cases. No example enlarges it using an obstacle detour.
    """
    informative = _body([0., 10.], history_dbm=-95.74175747197118,
        worlds=[_world('north_dynamic', [85., 30.], scale=24., velocity=[1., 0.]),
                _world('south_dynamic', [85., -30.], scale=24., velocity=[1., 0.])],
        obstacles=[[65., -12., 100., 12.]])
    no_information = _body([0., 0.], history_dbm=-99.99999547717543,
        worlds=[_world('north_dynamic', [115., 25.], scale=18., velocity=[.5, 0.]),
                _world('south_dynamic', [115., -25.], scale=18., velocity=[.5, 0.])],
        obstacles=[[0., 1., 50., 8.], [0., -8., 50., -1.],
                   [65., -12., 100., 12.]])
    single_world = deepcopy(informative)
    single_world['worlds'] = [deepcopy(informative['worlds'][0])]
    single_world['worlds'][0]['weight'] = 1.
    return dict(informative=informative, no_information=no_information,
                single_world=single_world)


def build_assessment_models():
    """Return evaluator-only models; these never enter a prepared task.

    The noise control keeps both declared fields and changes the future scalar
    noise from 1 to 3 dB. The field control replaces the model pool with a narrow
    moving hotspot outside both declared worlds. Neither case guarantees a
    conditional-policy gain or loss; both expose dependence on model validity.
    """
    informative = build_examples()['informative']
    outside = dict(id='outside_narrow_hotspot', weight=1., background_dbm=-100.,
        bases=[_base([85., 0.], scale=12., velocity=[-1., 0.], peak=-63., rate=.2)])
    return dict(noise_mismatch=dict(worlds=deepcopy(informative['worlds']),
                                    measurement_noise_db=3.),
                field_mismatch=dict(worlds=[outside], measurement_noise_db=1.))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def run_examples():
    """Freeze complete synthetic policies before alternative-model assessment."""
    from .strategy_graph import StrategyBudget, assess_strategy, prepare_task
    from .strategy_search import search_dynamic, search_open_loop

    cases = {}
    for name, body in build_examples().items():
        prepared = prepare_task(body, StrategyBudget())
        fixed = search_open_loop(prepared, StrategyBudget())
        contingent = search_dynamic(prepared, StrategyBudget())
        policies = {key:deepcopy(row['best']['policy']) for key, row in
                    (('open_loop', fixed), ('conditional', contingent)) if row['best'] is not None}
        frozen = dict(policies=policies, digest=_digest(policies))
        cases[name] = dict(input=body, prepared=prepared, open_loop=fixed, conditional=contingent,
            frozen_before_assessment=frozen,
            expected_gain_seconds=(fixed['best']['expected_score_seconds']-contingent['best']['expected_score_seconds']
                if fixed['best'] and contingent['best'] else None),
            billed_work_by_method={key:{counter:prepared['work'][counter]+row['work'][counter]
                for counter in prepared['work']} for key, row in (('open_loop', fixed), ('conditional', contingent))},
            scope='artificial model expectation; each method is billed the same full common preparation plus its own search')
    # All planning results are frozen before even constructing the evaluator-only
    # alternatives. No alternate model is passed to either numerical search.
    models = build_assessment_models()
    case = cases['informative']
    original = deepcopy(case['prepared'])
    assessments = {}
    for name, model in models.items():
        rows = {method:assess_strategy(case['prepared'], policy, model['worlds'],
                    model['measurement_noise_db'], StrategyBudget())
                for method, policy in case['frozen_before_assessment']['policies'].items()}
        assessments[name] = dict(artificial_model=model, results=rows,
            conditional_gain_seconds=rows['open_loop']['expected_score_seconds']-rows['conditional']['expected_score_seconds'],
            scope='offline alternative artificial expectation, after policy freeze; not unknown-field truth or validation data')
    assert case['prepared'] == original
    for row in cases.values():
        assert _digest(row['frozen_before_assessment']['policies']) == row['frozen_before_assessment']['digest']
    limited = search_dynamic(case['prepared'], StrategyBudget(action_evaluations=2))
    return dict(version='corridor-strategy-examples-v1', cases=cases, assessments=assessments,
        budget_retention=limited, new_flights=0, new_sensor_points=0, new_true_field_queries=0,
        new_gp_queries=0, new_experimental_llm_calls=0, new_training_runs=0,
        interpretation='Generated finite corridors and shared exact belief DP only; no evidence of LLM mission gain.')
