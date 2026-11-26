"""Frozen EP1 fitting, public selection requests, and sequential predictive scores."""

from copy import deepcopy
from dataclasses import asdict
from itertools import combinations
import json
import math
from types import SimpleNamespace

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from .equivalent_field import kernel, power_and_jacobian
from .event_inference import Budget, FAMILIES, digest, fit_job, point
from .point_gp import space_time_kernel
from .probability import gaussian_logpdf, logsumexp, normalized


INSTRUCTIONS = ("根据给定的历史观测和预测诊断，选择两个不同的候选来预测未来普通读数。"
                "只返回 JSON，其中 request_id 原样返回，ids 是两个候选代号，reason 是一句基于已给证据的理由。")
PUBLIC = dict(map=dict(bounds_m=[[0., 0.], [400., 400.]]), goal_m=[390., 200.])
MEANINGS = dict(static="固定空间分布；中心不动，强度不随时间变化。",
                translate="空间分布以固定速度平移；分量峰值强度不变。",
                strength_change="空间中心不动，分量强度以固定 dB/s 速率变化。")
RESOURCE_KEYS = ("solver_calls", "residual_calls", "jacobian_calls", "dictionary_kernel_terms")


def wire_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode()


def history_prefix(history, count):
    if len(history) != count:
        raise ValueError("exact frozen prefix length required")
    result = [point(sample) for sample in history]
    if any(sample["time_seconds"] != index for index, sample in enumerate(result)):
        raise ValueError("EP1 requires one ordered point per integer second")
    if any(not all(0 <= axis <= 400 for axis in sample["position_m"]) for sample in result):
        raise ValueError("point outside EP1 public bounds")
    return result


def fit_structures(history):
    if len(history) not in (8, 12):
        raise ValueError("EP1 fitting requires eight or twelve points")
    history = history_prefix(history, len(history))
    bank = SimpleNamespace(history=history, models=[], log_weights=[])
    jobs = [dict(family=family, region="task_disk", max_bases=2) for family in FAMILIES]
    budget = Budget()
    budget.reserve(jobs)
    fits = {}
    for job in jobs:
        before = asdict(budget)
        family = job["family"]
        fit = fit_job(bank, deepcopy(PUBLIC), job, budget, f"EP1-{len(history)}-{family}")
        resources = {key: getattr(budget, key) - before[key] for key in RESOURCE_KEYS}
        if (resources["solver_calls"] > 2 or resources["residual_calls"] > 80
                or resources["jacobian_calls"] > 80
                or resources["dictionary_kernel_terms"] > len(history) * dict(
                    static=39, translate=156, strength_change=78)[family]):
            raise ValueError("EP1 per-structure resource violation")
        if fit["window"] != history or fit["model"]["scored_points"] != 0:
            raise ValueError("fitting changed the frozen prefix or inherited evidence")
        fits[family] = dict(fit=fit, resources=resources, history_sha256=digest(history),
                            model_sha256=digest(fit["model"]))
    return fits


def model_mean(model, position, time_s):
    value = power_and_jacobian(kernel(model["bases"], [position], [time_s],
                                    model["reference_time_seconds"]), model["peaks_dbm"])[0][0]
    if not math.isfinite(float(value)):
        raise ValueError("nonfinite EP1 prediction")
    return float(value)


def distribution(means, stds, log_weights):
    means, stds = np.asarray(means, float), np.asarray(stds, float)
    log_weights = np.asarray(log_weights, float)
    if (not len(means) or means.shape != stds.shape or means.shape != log_weights.shape
            or not np.all(np.isfinite(means)) or not np.all(np.isfinite(stds)) or np.any(stds <= 0)
            or not math.isclose(logsumexp(log_weights), 0., abs_tol=1e-10)):
        raise ValueError("finite normalized predictive distribution required")
    weights = np.exp(log_weights)
    center = float(weights @ means)
    lower, upper = float(np.min(means - 12 * stds)), float(np.max(means + 12 * stds))

    def quantile(probability):
        return float(brentq(lambda value: float(weights @ ndtr((value - means) / stds)) - probability,
                            lower, upper, xtol=1e-10))

    return dict(means_dbm=means.tolist(), stds_db=stds.tolist(), log_weights=log_weights.tolist(),
                mean_dbm=center, lower90_dbm=quantile(.05), upper90_dbm=quantile(.95))


def score_issued(issued, observation):
    observation = point(observation)
    if (issued["position_m"] != observation["position_m"]
            or issued["time_seconds"] != observation["time_seconds"]):
        raise ValueError("observation must match the previously frozen query")
    value = observation["interference_dbm"]
    likelihood = [gaussian_logpdf(value, mean, std)
                  for mean, std in zip(issued["means_dbm"], issued["stds_db"])]
    nll = -logsumexp(np.asarray(issued["log_weights"]) + likelihood)
    error = value - issued["mean_dbm"]
    return dict(time_seconds=observation["time_seconds"], position_m=observation["position_m"],
                observed_dbm=value, mean_dbm=issued["mean_dbm"], nll=float(nll),
                squared_error=float(error ** 2), absolute_error=float(abs(error)),
                lower90_dbm=issued["lower90_dbm"], upper90_dbm=issued["upper90_dbm"],
                covered90=bool(issued["lower90_dbm"] <= value <= issued["upper90_dbm"]),
                interval_width_db=issued["upper90_dbm"] - issued["lower90_dbm"],
                all_mismatch=all(abs(value - mean) > 8 for mean in issued["means_dbm"]),
                world_nll=[-float(value) for value in likelihood])


class FrozenMixture:
    def __init__(self, models, history, *, update_weights):
        self.models = deepcopy(models)
        self.history = deepcopy(history)
        self.log_weights = np.full(len(models), -math.log(len(models)))
        self.fingerprint = digest(self.models)
        self.birth_count = len(history)
        self.update_weights = update_weights
        self.pending = None

    def issue(self, position, time_s):
        if self.pending is not None or digest(self.models) != self.fingerprint:
            raise ValueError("pending prediction or mutated frozen models")
        if time_s <= self.history[-1]["time_seconds"]:
            raise ValueError("only a new ordinary point can be predicted")
        means = [model_mean(model, position, time_s) for model in self.models]
        self.pending = dict(distribution(means, [model["std_db"] for model in self.models], self.log_weights),
                            position_m=list(position), time_seconds=float(time_s),
                            history_sha256=digest(self.history), models_sha256=self.fingerprint,
                            scored_points=len(self.history) - self.birth_count)
        return deepcopy(self.pending)

    def observe(self, observation):
        if (self.pending is None or digest(self.models) != self.fingerprint
                or self.pending["history_sha256"] != digest(self.history)):
            raise ValueError("no matching frozen prediction")
        row = score_issued(self.pending, observation)
        if self.update_weights:
            self.log_weights = normalized(self.log_weights - np.asarray(row["world_nll"]))
        self.history.append(point(observation))
        self.pending = None
        return row


class RollingGP:
    def __init__(self, history):
        self.history = deepcopy(history)
        self.pending = None
        self.decompositions = 0
        self.settings = SimpleNamespace(gp_length_scale_m=80., gp_time_scale_seconds=16., unknown_error_db=8.)

    def issue(self, position, time_s):
        if self.pending is not None or time_s <= self.history[-1]["time_seconds"]:
            raise ValueError("pending prediction or nonfuture GP query")
        history = self.history[-12:]
        xy = np.asarray([sample["position_m"] for sample in history])
        times = np.asarray([sample["time_seconds"] for sample in history])
        labels = np.asarray([sample["interference_dbm"] for sample in history])
        matrix = space_time_kernel(xy, times, xy, times, self.settings)
        matrix += (4. + 1e-8) * np.eye(len(history))
        cholesky = np.linalg.cholesky(matrix)
        self.decompositions += 1
        cross = space_time_kernel([position], [time_s], xy, times, self.settings)[0]
        alpha = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, labels - labels.mean()))
        projected = np.linalg.solve(cholesky, cross)
        variance = 64. + 4. - float(projected @ projected)
        if variance <= 0:
            raise ValueError("nonpositive GP predictive variance")
        self.pending = dict(distribution([float(labels.mean() + cross @ alpha)], [math.sqrt(variance)], [0.]),
                            position_m=list(position), time_seconds=float(time_s),
                            history_sha256=digest(self.history))
        return deepcopy(self.pending)

    def observe(self, observation):
        if self.pending is None or self.pending["history_sha256"] != digest(self.history):
            raise ValueError("no matching GP prediction")
        row = score_issued(self.pending, observation)
        self.history.append(point(observation))
        self.pending = None
        return row


def pilot_feedback(fits, prefix, observations, emit):
    prefix = history_prefix(prefix, 8)
    if len(observations) != 4 or set(fits) != set(FAMILIES):
        raise ValueError("three pilot fits and four feedback points required")
    pilot = FrozenMixture([fits[family]["fit"]["model"] for family in FAMILIES], prefix, update_weights=False)
    records = []
    for sample in observations:
        issued = pilot.issue(sample["position_m"], sample["time_seconds"])
        emit("issued", deepcopy(issued))
        row = pilot.observe(sample)
        record = dict(issued=issued, observation=point(sample), score=row)
        emit("scored", deepcopy(record))
        records.append(record)
    return records


def make_request(history, pilot, aliases, *, semantic):
    history = history_prefix(history, 12)
    if set(aliases) != {"A", "B", "C"} or set(aliases.values()) != set(FAMILIES) or len(pilot) != 4:
        raise ValueError("three distinct anonymous structures and four feedback rows required")
    candidates = []
    for alias, family in sorted(aliases.items()):
        index = FAMILIES.index(family)
        rows = []
        for record in pilot:
            observation = record["observation"]
            if observation != history[int(observation["time_seconds"])]:
                raise ValueError("feedback must match public history")
            issued = record["issued"]
            mean, std = issued["means_dbm"][index], issued["stds_db"][index]
            rows.append(dict(time_seconds=observation["time_seconds"], position_m=observation["position_m"],
                             mean_dbm=mean, std_db=std, observed_dbm=observation["interference_dbm"],
                             residual_db=observation["interference_dbm"] - mean))
        candidate = dict(id=alias, pilot_predictions=rows)
        if semantic:
            candidate["meaning"] = MEANINGS[family]
        candidates.append(candidate)
    scores = []
    for first, second in combinations(candidates, 2):
        losses = []
        for left, right in zip(first["pilot_predictions"], second["pilot_predictions"]):
            losses.append(-logsumexp([gaussian_logpdf(row["observed_dbm"], row["mean_dbm"], row["std_db"])
                                       - math.log(2) for row in (left, right)]))
        scores.append(dict(ids=[first["id"], second["id"]], mean_nll=float(np.mean(losses))))
    request = dict(instructions=INSTRUCTIONS, history=history,
                   public=dict(bounds_m=PUBLIC["map"]["bounds_m"], buildings=[],
                               units=dict(position="m", time="s", power="dBm", std="dB")),
                   candidates=candidates, pair_scores=scores, choose=2,
                   budget=dict(pilot_jobs=3, final_jobs=2, total_solver_calls=10,
                               total_residual_calls=400, total_jacobian_calls=400,
                               total_dictionary_kernel_terms=4992))
    request["request_id"] = digest(request)
    return request


def validate_request(request):
    allowed = {"instructions", "history", "public", "candidates", "pair_scores", "choose", "budget", "request_id"}
    if not isinstance(request, dict) or set(request) != allowed:
        raise ValueError("EP1 request public allowlist violation")
    if request["request_id"] != digest({key: value for key, value in request.items() if key != "request_id"}):
        raise ValueError("EP1 request hash mismatch")
    history_prefix(request["history"], 12)
    expected_public = dict(bounds_m=PUBLIC["map"]["bounds_m"], buildings=[],
                           units=dict(position="m", time="s", power="dBm", std="dB"))
    if (request["public"] != expected_public or request["instructions"] != INSTRUCTIONS
            or request["choose"] != 2 or len(request["candidates"]) != 3
            or {candidate["id"] for candidate in request["candidates"]} != {"A", "B", "C"}
            or request["budget"] != dict(pilot_jobs=3, final_jobs=2, total_solver_calls=10,
                                         total_residual_calls=400, total_jacobian_calls=400,
                                         total_dictionary_kernel_terms=4992)):
        raise ValueError("EP1 public values differ from the frozen design")
    for candidate in request["candidates"]:
        if set(candidate) not in ({"id", "pilot_predictions"}, {"id", "pilot_predictions", "meaning"}):
            raise ValueError("candidate public allowlist violation")
        if "meaning" in candidate and candidate["meaning"] not in MEANINGS.values():
            raise ValueError("unknown EP1 structure meaning")
        rows = candidate["pilot_predictions"]
        if len(rows) != 4:
            raise ValueError("four pilot feedback rows required")
        for index, row in enumerate(rows, 8):
            if (set(row) != {"time_seconds", "position_m", "mean_dbm", "std_db", "observed_dbm", "residual_db"}
                    or row["time_seconds"] != index
                    or row["position_m"] != request["history"][index]["position_m"]
                    or row["observed_dbm"] != request["history"][index]["interference_dbm"]
                    or row["std_db"] != 4.):
                raise ValueError("pilot feedback public allowlist violation")
    if (len(request["pair_scores"]) != 3 or any(set(row) != {"ids", "mean_nll"}
                                               for row in request["pair_scores"])):
        raise ValueError("EP1 pair scores public allowlist violation")
    return request


def numerical_selection(request):
    validate_request(request)
    best = min(request["pair_scores"], key=lambda item: (item["mean_nll"], item["ids"]))
    return list(best["ids"])


def parse_reply(request, raw):
    validate_request(request)

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate response field")
            result[key] = value
        return result

    response = json.loads(raw, object_pairs_hook=unique_keys,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    if not isinstance(response, dict) or set(response) != {"request_id", "ids", "reason"}:
        raise ValueError("expected one EP1 response object")
    if response["request_id"] != request["request_id"]:
        raise ValueError("wrong EP1 response binding")
    ids = response["ids"]
    if (not isinstance(ids, list) or len(ids) != 2 or any(type(alias) is not str for alias in ids)
            or len(set(ids)) != 2 or not set(ids) <= {"A", "B", "C"}
            or not isinstance(response["reason"], str) or not response["reason"].strip()):
        raise ValueError("two distinct candidate IDs and a reason required")
    return dict(response, ids=sorted(ids))


def charged_resources(pilot_fits, final_fits, families, *, pilot=True):
    result = {key: 0 for key in RESOURCE_KEYS}
    jobs = list(pilot_fits.values()) if pilot else []
    jobs += [final_fits[family] for family in families]
    for fit in jobs:
        for key in RESOURCE_KEYS:
            result[key] += fit["resources"][key]
    result["fit_jobs"] = len(jobs)
    return result
