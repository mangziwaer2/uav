"""Build one immutable, exhaustive evidence table before either controller selects."""

from dataclasses import dataclass
import hashlib
import json

from .passive_local_repair import LocalRepair, ResolvedRepair
from .passive_repair_pricing import LockedPrice, RepairComparison


ACTION_COLUMNS = ("action_id", "time_s", "priced_s", "unknown_s", "supported_not_compared_s", "outside_s",
                  "expected_q", "tail_q", "status", "sources")
SOURCE_COLUMNS = ("base_id", "edits", "split_s", "post_common_s", "missing_required_s", "delta_t_s",
                  "delta_window_expected_q", "delta_window_tail_q", "delta_post_expected_q",
                  "delta_post_tail_q", "status")


@dataclass(frozen=True)
class ActionSource:
    base_id: int
    edits: tuple[LocalRepair, ...]
    comparison: RepairComparison


@dataclass(frozen=True)
class RepairAction:
    action_id: int
    resolved: ResolvedRepair
    price: LockedPrice
    sources: tuple[ActionSource, ...]


@dataclass(frozen=True)
class FrozenActionTable:
    original_count: int
    waypoints: tuple
    locked_intervals: tuple
    rows: tuple[RepairAction, ...]
    time_only: bool
    splice_attempts: int

    def public(self):
        waypoints = {point: index for index, point in enumerate(self.waypoints)}

        def source_row(source):
            item = source.comparison
            edits = tuple((edit.left, edit.right, None if edit.waypoint_xy_m is None else waypoints[edit.waypoint_xy_m])
                          for edit in source.edits)
            return (source.base_id, edits, item.split_s, item.post_common_s, item.missing_required_s, item.delta_t_s,
                    item.delta_expected_q, item.delta_tail_q, item.delta_post_expected_q, item.delta_post_tail_q,
                    item.price.status)

        def action_row(action):
            price = action.price
            return (action.action_id, price.time_s, price.priced_s, price.unknown_s, price.supported_not_compared_s,
                    price.outside_s, price.expected_q, price.tail_q, price.status,
                    tuple(source_row(source) for source in action.sources))

        return dict(routes=tuple((row.action_id, row.resolved.route) for row in self.rows[:self.original_count]),
                    repair_waypoints_xy_m=self.waypoints, locked_intervals_s=self.locked_intervals,
                    action_table=dict(version=1, columns=ACTION_COLUMNS, source_columns=SOURCE_COLUMNS,
                                      time_only=self.time_only, rows=tuple(map(action_row, self.rows))))


def action_table_digest(public):
    return hashlib.sha256(json.dumps(public, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def freeze_actions(pricing, neighborhood, incumbent_id):
    originals = neighborhood.originals
    if ((pricing.now, pricing.speed) != (neighborhood.now, neighborhood.speed)
            or pricing.routes != tuple(row.route for row in originals) or not 0 <= incumbent_id < len(originals)):
        raise ValueError("pricing and legal action domain must use the same checkpoint")
    rows = []
    for index, resolved in enumerate(originals):
        sources = (ActionSource(incumbent_id, (), pricing.compare_original(index, incumbent_id)),
                   *(ActionSource(base, edits, pricing.compare_original(index, base))
                     for base, edits in neighborhood.original_aliases[index].items()))
        rows.append(RepairAction(index, resolved, pricing.menu[index], sources))
    for resolved, aliases in neighborhood.expanded:
        comparisons = pricing.compare_references(resolved.route, aliases)
        sources = [ActionSource(index, aliases[index], comparison) for index, comparison in zip(aliases, comparisons)]
        primary = next((index for index, source in enumerate(sources) if source.comparison.price.score is not None), 0)
        sources.insert(0, sources.pop(primary))
        rows.append(RepairAction(len(rows), resolved, sources[0].comparison.price, tuple(sources)))
    return FrozenActionTable(len(originals), neighborhood.waypoints, pricing.locked_intervals, tuple(rows),
                             pricing.menu[0].status == "time_only", neighborhood.tried)


def choose_numerical(table):
    def rank(action):
        time_s = action.price.time_s
        score = time_s if table.time_only else action.price.score
        return round(score, 8), time_s, action.action_id

    return min((row for row in table.rows if table.time_only or row.price.score is not None), key=rank)
