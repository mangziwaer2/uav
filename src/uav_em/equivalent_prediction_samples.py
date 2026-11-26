"""Private EP1 synthetic tables; this module never enters model requests."""

from hashlib import sha256
import math

import numpy as np

from .event_inference import FAMILIES


def sample_seed(layer, rep, tag, *, namespace="EP1-v1"):
    if namespace not in ("EP1-v1", "MC1-v1"):
        raise ValueError("registered independent sample namespace required")
    return int.from_bytes(sha256(f"{namespace}/{layer}/{rep}/{tag}".encode("ascii")).digest()[:8], "big")


def sample_order(*, namespace="EP1-v1", repetitions=6):
    if (namespace, repetitions) not in (("EP1-v1", 6), ("MC1-v1", 3)):
        raise ValueError("registered sample design required")
    return sorted(((layer, rep) for layer in range(4) for rep in range(repetitions)),
                  key=lambda item: (sample_seed(*item, "order", namespace=namespace), *item))


def synthetic_table(layer, rep, *, namespace="EP1-v1"):
    repetitions = 6 if namespace == "EP1-v1" else 3
    if type(layer) is not int or type(rep) is not int or not 0 <= layer < 4 or not 0 <= rep < repetitions:
        raise ValueError("EP1 requires a registered layer and repetition")
    streams = {tag: np.random.Generator(np.random.PCG64(sample_seed(layer, rep, tag, namespace=namespace)))
               for tag in ("samples", "signal", "noise", "alias")}
    samples = streams["samples"]
    east0, north0 = samples.uniform(70., 130.), samples.uniform(140., 220.)
    turn = int(samples.choice([6, 9, 12]))
    lateral = int(samples.choice([-6, 0, 6]))
    times = np.arange(18, dtype=float)
    east = east0 + 10 * times
    north = north0 + lateral * np.maximum(0, times - turn)
    signal = streams["signal"]
    level = float(signal.uniform(-90., -78.))
    angle, spatial_phase, temporal_phase = signal.uniform(0, 2 * math.pi, size=3)
    sign = int(signal.choice([-1, 1]))
    jump_time = int(signal.integers(12, 18))
    spatial = 7 * np.sin(2 * math.pi * (east * math.cos(angle) + north * math.sin(angle)) / 240 + spatial_phase)
    temporal = 7 * np.sin(2 * math.pi * times / 32 + temporal_phase)
    clean = (level + spatial, level + temporal, level + .7 * (spatial + temporal),
             level + 9 * sign * (times >= jump_time))[layer]
    observed = clean + streams["noise"].normal(0., 2., size=18)
    aliases = {alias: FAMILIES[int(index)]
               for alias, index in zip(("A", "B", "C"), streams["alias"].permutation(3))}
    history = [dict(time_seconds=float(stamp), position_m=[float(xpos), float(ypos)],
                    interference_dbm=float(value)) for stamp, xpos, ypos, value in zip(times, east, north, observed)]
    return dict(layer=layer, repetition=rep, aliases=aliases, measurements=history, clean_dbm=clean.tolist(),
                parameters=dict(east0=float(east0), north0=float(north0), turn=turn, lateral=lateral,
                                level=level, angle=float(angle), spatial_phase=float(spatial_phase),
                                temporal_phase=float(temporal_phase), sign=sign, jump_time=jump_time))
