"""Pluggable serving schedulers (pure functions, no GPU).

Each scheduler decides how pending robot requests are grouped into
inference batches. Middleware under test implements `group_requests`
with one of these policies:

- sequential: one robot per batch (current Pi_05 behavior:
  model.py get_action_batch loops infer per env; model_server holds
  a global _model_lock). Baseline.
- round_robin: fill batches up to max_batch in arrival order.
- earliest_deadline: robots with fewest actions left go first
  (EDF analog from Armory arXiv:2608.00337).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PendingRequest:
    robot_id: str
    actions_left: int = 0  # actions remaining in current chunk (0 = starving)
    arrived_ms: float = 0.0


def sequential(pending: list[PendingRequest]) -> list[list[PendingRequest]]:
    """One batch per robot (status quo)."""
    return [[r] for r in pending]


def round_robin(pending: list[PendingRequest],
                max_batch: int = 4) -> list[list[PendingRequest]]:
    """Fill batches up to max_batch in arrival order."""
    batches: list[list[PendingRequest]] = []
    cur: list[PendingRequest] = []
    for r in pending:
        cur.append(r)
        if len(cur) >= max_batch:
            batches.append(cur)
            cur = []
    if cur:
        batches.append(cur)
    return batches


def earliest_deadline(pending: list[PendingRequest],
                      max_batch: int = 4) -> list[list[PendingRequest]]:
    """Starving robots first, then fewest actions left (Armory EDF)."""
    ordered = sorted(pending, key=lambda r: (r.actions_left, r.arrived_ms))
    return round_robin(ordered, max_batch)


SCHEDULERS = {
    "sequential": sequential,
    "round_robin": round_robin,
    "earliest_deadline": earliest_deadline,
}
