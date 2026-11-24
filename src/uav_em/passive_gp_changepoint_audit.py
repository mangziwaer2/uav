"""One frozen, read-only audit of two public P7 route menus."""

from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp

from .config import AircraftConfig, BeliefConfig, MapConfig
from .environment import Geometry
from .passive_gp_changepoint import RegimeForecast, fit_regimes, price_legal_routes
from .passive_map import FixedPriorPosterior, _common_time, candidates
from .passive_pricing_integral import _trajectory, compare_aligned_prices


ROOT = Path(__file__).resolve().parents[2]
P7 = ROOT / "outputs/passive_route_actions_P7_20261124"
OUT = ROOT / "outputs/passive_gp_changepoint_P7_audit_20261124"
CONTRACT = ROOT / "docs/passive_gp_changepoint_P7_audit_contract.md"
PARAMS = ROOT / "outputs/passive_closed_loop_P4_20261124/public_inputs.json"
PRIMARY, SECONDARY = (P7 / f"public_route_{idx}.json" for idx in (7, 8))
CHECKPOINT, REVIEW = (P7 / name for name in ("public_checkpoint.json", "review.json"))
SPECS = ((12, 12), (24, 9))
HAZARD = 1 / 24
EPS = 1e-8
LIMITS = dict(fits=458, queries=750_000, integrals=444)


def _read(path):
    return json.loads(path.read_bytes())


def _put(name, value):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _hash(path):
    return sha256(path.read_bytes()).hexdigest()


def _sample(log, stamp):
    source = log["points"][stamp]
    if source["time_seconds"] != stamp:
        raise ValueError("expected original one-second measurement cadence")
    return {key: source[key] for key in ("time_seconds", "position_m", "interference_dbm")}


def _public_inputs():
    params, log, packet, review = map(_read, (PARAMS, PRIMARY, CHECKPOINT, REVIEW))
    parameters = {key: params[key] for key in ("map", "aircraft", "belief")}
    if packet["buildings"] != parameters["map"]["buildings"] or parameters["belief"]["gp_time_scale_seconds"] != 20:
        raise ValueError("public geometry or nominal GP parameters changed")
    cases = []
    for now, count in SPECS:
        frame = log["frames"][now]
        decision = next(row for row in log["decisions"] if row["t"] == now)
        routes = [row["route"] for row in decision["candidates"]]
        points = [_sample(log, stamp) for stamp in range(now + 1)]
        if frame["t"] != now or frame["prefix_size"] != now + 1 or len(routes) != count:
            raise ValueError("original checkpoint shape changed")
        if now == 24:
            if (packet["point_rows"] != [[p["time_seconds"], *p["position_m"], p["interference_dbm"]] for p in points]
                    or [row["route_xy_m"] for row in review["routes"]] != routes
                    or review["incumbent_idx"] != decision["reference"]
                    or packet["position_xy_m"] != frame["xy"] or packet["flown_m"] != frame["flown_m"]):
                raise ValueError("24-second public checkpoint does not match the log")
            original_prices = [row["risk_proxy_tq"] for row in review["routes"]]
        else:
            original_prices = [row["costs"]["risk"] for row in decision["candidates"]]
        cases.append(dict(now=now, xy=frame["xy"], flown_m=frame["flown_m"], points=points, routes=routes,
                          incumbent=decision["reference"], cap_m=log["cap_m"], original_prices=original_prices))
    return dict(parameters=parameters, cases=cases)


def _best(costs, travel, indices):
    minimum = min(costs[index] for index in indices)
    return min((index for index in indices if costs[index] <= minimum + EPS),
               key=lambda index: (travel[index], index))


def _condition(weights, densities):
    joint = np.asarray(weights) + densities
    return np.exp(joint - logsumexp(joint, axis=-1, keepdims=True))


def _quadrature(forecast, costs, eligible, travel, fixed, node_count):
    nodes, quadrature_weights = hermgauss(node_count)
    means, std = np.asarray(forecast.means_dbm), np.asarray(forecast.observation_std_db)
    readings = (means[:, None] + math.sqrt(2) * std[:, None] * nodes).ravel()
    density = -0.5 * ((readings[:, None] - means) / std) ** 2 - np.log(std)
    posterior = _condition(forecast.log_weights, density)
    expected = posterior @ costs
    actions = np.array([_best(row, travel, eligible) for row in expected])
    weights = np.outer(np.exp(forecast.log_weights), quadrature_weights / math.sqrt(math.pi)).ravel()
    gain = expected[:, fixed] - expected[np.arange(len(actions)), actions]
    return dict(nodes=node_count, expected_proxy_gain=float(weights @ gain),
                change_probability=float(weights @ (actions != fixed)))


def _prepare(case, parameters):
    aircraft = AircraftConfig(**parameters["aircraft"])
    settings = BeliefConfig(**parameters["belief"])
    geometry = Geometry(MapConfig(**parameters["map"]), aircraft.clearance_m, aircraft.altitude_m,
                        aircraft.vertical_clearance_m)
    routes, now, incumbent = (case[key] for key in ("routes", "now", "incumbent"))
    speed = aircraft.speed_mps
    goal = list(geometry.config.goal_m)
    cap = 1.25 * math.dist(geometry.config.start_m, goal)
    energy = aircraft.initial_energy_wh - now * aircraft.flight_power_w / 3600
    if (case["cap_m"] != cap or case["points"][-1]["position_m"] != case["xy"]
            or candidates(case["xy"], goal, routes[incumbent], geometry, aircraft,
                          now, case["flown_m"], energy, cap) != routes):
        raise ValueError("original whole-mission legality does not reconstruct")
    nominal = FixedPriorPosterior(case["points"], geometry, now, settings)
    baseline = compare_aligned_prices(routes, nominal, now, speed, step_s=0.25)
    nominal_prices = [row["costs"]["risk"] for row in baseline["rows"]]
    error = max(abs(left - right) for left, right in zip(nominal_prices, case["original_prices"]))
    if error > 1e-7:
        raise ValueError("strong N does not reconstruct the P7 prices")
    belief = fit_regimes(case["points"], geometry, now, settings, change_probability=HAZARD, max_samples=25)
    costs = price_legal_routes(routes, belief, speed)
    travel = [row["time_s"] for row in baseline["rows"]]
    lead = [_common_time(routes[incumbent], route, speed) - 1 for route in routes]
    eligible = [idx for idx, seconds in enumerate(lead) if seconds > EPS]
    knots, vertices = _trajectory(routes[incumbent], speed)
    xy = tuple(np.interp(1.0, knots, axis) for axis in vertices.T)
    forecast = belief.freeze_next(xy, now + 1)
    mean, std, _ = nominal.predict([xy], [now + 1])
    nominal_forecast = RegimeForecast(now, now + 1, xy, (0.0,), (float(mean[0]),),
                                      (math.hypot(std[0], settings.measured_error_db),))
    indices = range(len(routes))
    expected = costs["expected_proxy_tq"]
    world_costs = np.asarray(costs["proxy_by_run_length"])
    best_all = _best(expected, travel, indices)
    fixed = _best(expected, travel, eligible)
    return dict(t_s=now, routes=routes, incumbent=incumbent, travel_s=travel, lead_after_sample_s=lead,
                eligible=eligible, nominal_prices=nominal_prices, nominal_best=_best(nominal_prices, travel, indices),
                no_reset_best=_best(world_costs[-1], travel, indices), mixture_best=best_all, waiting_best=fixed,
                waiting_cost=expected[fixed] - expected[best_all], price_reconstruction_error=error,
                mixture=asdict(forecast), nominal=asdict(nominal_forecast), costs=costs,
                optimum_by_run_length=[_best(row, travel, eligible) for row in world_costs],
                quadrature=[_quadrature(forecast, world_costs, eligible, travel, fixed, size) for size in (32, 64)],
                resources=dict(fits=belief.fit_count + 1,
                               queries=baseline["queries"] + costs["queries"] + belief.fit_count + 1,
                               integrals=len(routes) * (len(belief.posteriors) + 1)))


def _score(frozen, sample):
    def restore(data):
        return RegimeForecast(**{key: tuple(value) if isinstance(value, list) else value
                                 for key, value in data.items()})

    forecast = restore(frozen["mixture"])
    nominal = restore(frozen["nominal"])
    if (sample["time_seconds"] != forecast.sample_s or
            math.dist(sample["position_m"], forecast.xy_m) > EPS):
        raise ValueError("logged sample did not follow the frozen shared prefix")
    matched = dict(sample, position_m=forecast.xy_m)
    mixture_log = forecast.log_predictive(matched)
    nominal_log = nominal.log_predictive(matched)
    posterior = _condition(forecast.log_weights, forecast._log_densities(sample["interference_dbm"]))
    costs = frozen["costs"]["proxy_by_run_length"]
    weighted = posterior @ costs
    chosen = _best(weighted, frozen["travel_s"], frozen["eligible"])
    return dict(sample=sample, log_density=dict(nominal=nominal_log, mixture=mixture_log),
                posterior_by_frozen_run_length=posterior.tolist(), conditional_action=chosen,
                weighted_costs=[float(cost) for cost in weighted])


def run():
    OUT.mkdir(exist_ok=False)
    with (OUT / "contract.md").open("xb") as stream:
        stream.write(CONTRACT.read_bytes())
    module = Path(__file__).resolve()
    implementation = (module, *(module.with_name(name) for name in (
        "passive_gp_changepoint.py", "passive_pricing_integral.py", "passive_pricing.py", "passive_map.py",
        "point_gp.py", "config.py", "environment.py", "planning.py")))
    paths = (*implementation, ROOT / "requirements.txt", PARAMS, PRIMARY, SECONDARY, CHECKPOINT, REVIEW, CONTRACT)
    hashes = {str(path.relative_to(ROOT)): _hash(path) for path in paths}
    _put("freeze.json", dict(sha256=hashes, change_probability=HAZARD, limits=LIMITS))
    public = _public_inputs()
    _put("public_inputs.json", public)
    frozen = [_prepare(case, public["parameters"]) for case in public["cases"]]
    resources = {key: sum(case["resources"][key] for case in frozen) for key in LIMITS}
    if any(resources[key] > maximum for key, maximum in LIMITS.items()):
        raise ValueError("audit exhausted its frozen resource budget")
    _put("forecasts.json", frozen)
    forecast_sha = _hash(OUT / "forecasts.json")
    if any(_hash(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError("audit source or public inputs changed before feedback")
    logs = [_read(path) for path in (PRIMARY, SECONDARY)]
    samples = [[_sample(log, now + 1) for log in logs] for now, _ in SPECS]
    if any(left != right for left, right in samples):
        raise ValueError("P7 logs did not share the licensed ordinary observations")
    scores = [_score(case, pair[0]) for case, pair in zip(_read(OUT / "forecasts.json"), samples)]
    _put("results.json", dict(forecasts_sha256=forecast_sha, resources=resources, scores=scores,
                              new_flights=0, llm_calls=0))


if __name__ == "__main__":
    run()
