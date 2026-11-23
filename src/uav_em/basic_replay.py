"""Bounded B0 demonstration. Scientific runs are single-use, never implicit."""
import argparse
import hashlib
import itertools
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import ProjectConfig
from .environment import ElectromagneticWorld
from .motion import advance_route
from .point_gp import FrozenGP


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/basic_replay_B0_20261123'
ARMS = ('A', 'B', 'NA', 'NB')
TIMES = (0., 14., 28.)
LEVELS = (180, 240, 300, 360, 420)
CONFIG = ProjectConfig(seed=42317)
AIR = CONFIG.aircraft
GOAL = list(CONFIG.map.goal_m)
CAP = 1.25 * math.dist(CONFIG.map.start_m, GOAL)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)


def length(route):
    return sum(math.dist(first, second) for first, second in zip(route, route[1:]))


def schedule(route, now):
    points, clocks, durations = [], [], []
    for first, second in zip(route, route[1:]):
        origin = np.asarray(first, float)
        distance = math.dist(first, second)
        if distance < 1e-10:
            continue
        velocity = (np.asarray(second) - origin) * AIR.speed_mps / distance
        left = distance / AIR.speed_mps
        while left > 1e-9:
            at_review = any(abs(now - event) < 1e-8 for event in TIMES)
            limit = min([event-now for event in TIMES if event > now+1e-8] or [left])
            step = .5 if at_review else min(1., left, limit)
            move = np.zeros(2) if at_review else velocity * step
            points.append(origin + move / 2); clocks.append(now+step/2); durations.append(step)
            if not at_review:
                left -= step
            now += step; origin = origin + move
    return np.asarray(points), np.asarray(clocks), np.asarray(durations)


def price(request, route):
    xy, times, dt = schedule(route, request['time_s'])
    power = np.ones(len(dt))
    if 'gp' in request:
        grid = np.asarray(request['gp'], float).reshape(-1, 5)
        if len(grid):
            distance = np.linalg.norm(xy[:, None] - grid[None, :, :2], axis=2)
            age = abs(times[:, None]-grid[None, :, 2])
            metric = np.where((distance <= 45) & (age <= 6), (distance/45)**2+(age/6)**2, np.inf)
            which = metric.argmin(axis=1); supported = np.isfinite(metric.min(axis=1))
            power[supported] = 10**((grid[which[supported], 3]+80)/10)
    else:
        raw = np.asarray(request['points_xy_t_dbm'])
        distance = np.linalg.norm(xy[:, None]-raw[None, :, :2], axis=2)
        age = times[:, None]-raw[None, :, 2]
        weights = np.where((distance <= 80) & (age <= 20), np.exp(-(distance/40)**2-age/10), 0)
        total = weights.sum(axis=1); mask = total > 0
        power[mask] = (weights[mask] @ 10**((raw[:, 3]+80)/10))/total[mask]
    return float(dt @ (1+power))


def check(route, req, geometry):
    if not route or route[0] != req['position'] or route[-1] != GOAL:
        return 'endpoints'
    if not all(geometry.edge_free(first, second) for first, second in zip(route, route[1:])):
        return 'geometry'
    dist = length(route)
    if req['flown_m']+dist > CAP+1e-8:
        return 'length'
    future_hover = sum(event >= req['time_s'] for event in TIMES)*.5
    energy = (dist/AIR.speed_mps*AIR.flight_power_w + future_hover*AIR.hover_power_w)/3600
    return 'energy' if req['energy_wh']-energy < AIR.reserve_energy_wh else None


def route_for(req, levels):
    if (not isinstance(levels, list) or len(levels) != len(req['remaining_gate_x'])
            or any(type(value) not in (int, float) or value not in LEVELS for value in levels)):
        raise ValueError('waypoint_domain')
    return [req['position'], *[list(pair) for pair in zip(req['remaining_gate_x'], levels)], GOAL]


class Flight:
    def __init__(self, arm):
        self.arm = arm
        self.world = ElectromagneticWorld(CONFIG)
        self.xy = list(CONFIG.map.start_m); self.time = self.flown = self.exposure = 0.
        self.energy = AIR.initial_energy_wh; self.route = []; self.obs = []; self.events = []; self.frames = []
        self.sample()

    def sample(self):
        self.obs.append(asdict(self.world.observe(self.xy, self.time)))
        self.frames.append(dict(t=self.time, xy=self.xy, route=self.route, flown=self.flown, energy=self.energy))

    def request(self, index):
        req = dict(id=f'{self.arm}-{index}', time_s=self.time, position=self.xy, goal=GOAL,
                   bounds_m=[0,0,800,600], obstacles_xyxy_m=CONFIG.map.obstacles_m,
                   clearance_m=AIR.clearance_m, speed_mps=AIR.speed_mps, flown_m=self.flown,
                   max_total_m=CAP, energy_wh=self.energy, reserve_wh=AIR.reserve_energy_wh,
                   flight_power_w=AIR.flight_power_w, hover_power_w=AIR.hover_power_w,
                   review_seconds=TIMES, hover_per_review_s=.5, previous_route=self.route,
                   remaining_gate_x=[gate for gate in (240,400,620) if gate > self.xy[0]+1e-8], allowed_y=LEVELS,
                   points_xy_t_dbm=[[*row['position_m'], row['time_seconds'], row['interference_dbm']] for row in self.obs],
                   feedback=[{key:ev[key] for key in ('status','reason','executed')} for ev in self.events],
                   task='Civil survey simulation. Minimize T + integral of linear received power / -80 dBm reference over time; larger dBm is worse. Unknown is not safe. Gate y list specifies a full route to goal. Keep only if previous_route exists. Inflated obstacle boundary cannot be touched.',
                   proxy='Unsupported normalized power=1. Raw: local exponential distance/age kernel within 80m,20s. GP if present: nearest supported grid in scaled distance and time within 45m,6s. No precomputed route scores.',
                   reply='One JSON object: {"id":"request id","action":"route" or "keep","ys":[one allowed y per remaining gate, or empty for keep],"reason":"brief"}')
        if self.arm in ('B','NB'):
            model = FrozenGP(self.obs,self.world.geometry,self.time,CONFIG.belief,-100.)
            xy = np.asarray([[xx,yy] for xx in range(0,801,60) for yy in range(0,601,60)])
            req['gp'] = []; req['gp_columns'] = 'x,y,forecast_time_s,mean_dbm,std_db; supported only, not measurements'
            for ahead in (0,12,24):
                times = np.full(len(xy), self.time+ahead)
                mean, std, support = model.predict(xy,times)
                req['gp'] += [[*xy[pos].tolist(),float(times[pos]),round(mean[pos],3),round(std[pos],3)] for pos in np.flatnonzero(support)]
        return req

    def decide(self, req, receipt):
        options = [route_for(req,list(levels)) for levels in itertools.product(LEVELS,repeat=len(req['remaining_gate_x']))]
        if self.route and self.route not in options:
            options.append(self.route)
        legal = [route for route in options if not check(route,req,self.world.geometry)]
        proposed = None; reason = ''; status = 'numeric'; rationale = ''
        if self.arm.startswith('N'):
            proposed = min(legal,key=lambda route:(price(req,route),length(route),route), default=None)
        else:
            try:
                raw = receipt['raw']
                if len(raw.encode()) > 800 or receipt['late']:
                    raise ValueError('reply budget/deadline')
                answer = json.loads(raw)
                if (set(answer) != {'id','action','ys','reason'} or answer['id'] != req['id'] or
                        not isinstance(answer['reason'],str) or not answer['reason'].strip()):
                    raise ValueError('schema')
                rationale = answer['reason']; action = answer['action']
                if action == 'keep' and self.route and answer['ys'] == []:
                    proposed = self.route
                elif action == 'route':
                    proposed = route_for(req, answer['ys'])
                else:
                    raise ValueError('action')
                reason = check(proposed,req,self.world.geometry) or ''
                status = 'accepted' if not reason else 'fallback'
            except (KeyError,TypeError,ValueError) as exc:
                status='fallback'; reason=str(exc)
        selected = proposed if status != 'fallback' else (self.route or min(legal,key=length,default=None))
        if not selected or check(selected,req,self.world.geometry):
            raise RuntimeError('no legal remaining route; stop')
        self.events.append(dict(t=self.time,req=req['id'],old=self.route,proposed=proposed,executed=selected,
                                status=status, reason=reason,rationale=rationale,proxy=price(req,selected),
                                feasible_in_domain=len(legal),gp=req.get('gp',[])))
        self.route=selected
        self.exposure += .5*10**((self.world.truth(self.xy,self.time+.25)+80)/10)
        self.time+=.5; self.energy-=.5*AIR.hover_power_w/3600

    def fly(self, until):
        while len(self.route)>1 and self.time < until-1e-8:
            step=min(1.,until-self.time,length(self.route)/AIR.speed_mps)
            remain, moved=advance_route(self.route,step*AIR.speed_mps)
            for first,second in zip(moved,moved[1:]):
                dist=math.dist(first,second); dt=dist/AIR.speed_mps
                assert self.world.geometry.edge_free(first,second)
                self.exposure+=dt*10**((self.world.truth((np.asarray(first)+second)/2,self.time+dt/2)+80)/10)
                self.time+=dt; self.flown+=dist; self.energy-=dt*AIR.flight_power_w/3600
            self.xy=remain[0]; self.route=remain; self.sample()
        assert self.flown <= CAP+1e-6 and self.energy >= AIR.reserve_energy_wh


def run():
    OUT.mkdir(exist_ok=False)
    contract=(ROOT/'docs/basic_replay_contract.md').read_bytes()
    (OUT/'contract.md').write_bytes(contract)
    write(OUT/'freeze.json',dict(contract_sha256=hashlib.sha256(contract).hexdigest(),
          implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seed=42317,arms=ARMS,times=TIMES))
    flights={arm:Flight(arm) for arm in ARMS}
    for index, now in enumerate(TIMES):
        pending={}
        for arm, flight in flights.items():
            assert abs(flight.time-now)<1e-7
            req=flight.request(index); pending[arm]=req
            write(OUT/'public'/arm/f'{index}.json',req)
        issued=time.time(); write(OUT/f'issued_{index}.json',issued)
        for arm, flight in flights.items():
            receipt=None
            if arm in ('A','B'):
                file=OUT/'receipts'/f'{arm}-{index}.json'
                while not file.exists() and time.time() < issued+300:
                    time.sleep(.2)
                receipt=json.loads(file.read_text()) if file.exists() else dict(raw='',late=True)
            flight.decide(pending[arm],receipt)
            flight.fly(TIMES[index+1] if index+1<len(TIMES) else 90.)
            write(OUT/'checkpoints'/f'{arm}-{index}.json',dict(event=flight.events[-1],time=flight.time,flown=flight.flown))
    for arm, flight in flights.items():
        write(OUT/f'{arm}.json',dict(events=flight.events,frames=flight.frames,observations=flight.obs,
             metrics=dict(arrived=math.dist(flight.xy,GOAL)<1e-6,T=flight.time,Q=flight.exposure,
                          total=flight.time+flight.exposure,length=flight.flown,energy_wh=AIR.initial_energy_wh-flight.energy)))


def receive(arm,index,agent,raw):
    issue=json.loads((OUT/f'issued_{index}.json').read_text()); wall=time.time()
    file=OUT/'receipts'/f'{arm}-{index}.json'; tmp=file.with_suffix('.pending')
    write(tmp,dict(agent=agent,raw=raw,received_unix=wall,elapsed_s=wall-issue,late=wall>issue+300))
    if file.exists():
        raise FileExistsError(file)
    tmp.rename(file)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['run','receive']); parser.add_argument('args',nargs='*')
    args=parser.parse_args()
    if args.command=='run': run()
    else: receive(*args.args)
