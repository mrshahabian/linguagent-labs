"""The simulator your scheduler drives.

You do not need to read this to solve the levels - but reading it after you
have solved them is one of the best ways to understand *why* the rules are
what they are. Every rule in ``docs/`` is enforced by a check in here.

The step loop is:

    1. Requests whose arrival step is now join ``waiting``.
    2. A frozen ``EngineView`` snapshot is taken.
    3. Your ``plan_step(view)`` returns a ``StepPlan``.
    4. The plan is *validated*. An illegal plan raises ``SchedulingError``
       rather than silently doing something surprising.
    5. The plan is applied: preempt, then decode, then prefill.
    6. Requests that reached ``max_tokens`` finish and release their memory.

Timing rules that fall out of that order, and that trip people up:

    * Prefill produces the request's *first* output token. A request with
      ``max_tokens = n`` therefore needs one prefill plus ``n - 1`` decodes.
    * The view is a snapshot. A request preempted this step is still in
      ``view.admitted``; a request admitted this step is still in
      ``view.waiting``.
    * Memory is held for the whole step. A request that finishes this step
      still occupies (and grows) its slots during it.
    * A preempted victim does no work this step, so its slots are free for
      this step's prefills immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .api import EngineView, KvEngineView, PrefillOp, Request, StepPlan


class SchedulingError(Exception):
    """Raised when a StepPlan breaks the engine's rules."""


@dataclass(frozen=True)
class Arrival:
    """A request to feed into the engine, and the step it shows up on."""

    id: str
    prompt_len: int
    max_tokens: int
    priority: int = 0
    at: int = 0


@dataclass
class _Live:
    """The engine's mutable copy of a request."""

    id: str
    prompt_len: int
    max_tokens: int
    priority: int
    arrival_at: int
    arrival_index: int
    prefill_remaining: int
    kv: int = 0
    generated: int = 0

    def snapshot(self) -> Request:
        return Request(
            id=self.id,
            prompt_len=self.prompt_len,
            max_tokens=self.max_tokens,
            priority=self.priority,
            prefill_remaining=self.prefill_remaining,
            kv=self.kv,
        )


@dataclass(frozen=True)
class StepRecord:
    """What actually happened on one step - one row of a worked-example table."""

    step: int
    waiting: tuple[str, ...]
    admitted: tuple[str, ...]
    kv_held: int
    preempt: tuple[str, ...]
    prefill: tuple[str, ...]  # rendered as "ID:tokens"
    decode: tuple[str, ...]
    finished: tuple[str, ...]


class Engine:
    """Runs a scheduler against a set of arrivals.

    max_work:     units of GPU work available per step. A prefill of n tokens
                  costs n; a decode costs 1.
    kv_capacity:  size of the KV-slot pool, or None for the levels that do not
                  model memory (1 and 2). When set, the view handed to your
                  scheduler is a ``KvEngineView``.
    chunked:      whether a prefill may cover only part of a request's
                  remaining context (Level 2 only).
    """

    def __init__(
        self,
        arrivals,
        max_work: int,
        kv_capacity: int | None = None,
        chunked: bool = False,
    ):
        self.max_work = max_work
        self.kv_capacity = kv_capacity
        self.chunked = chunked
        self.step_index = 0

        self._pending: list[_Live] = []
        for index, arrival in enumerate(arrivals):
            if arrival.max_tokens < 1:
                raise ValueError(f"{arrival.id}: max_tokens must be >= 1")
            self._pending.append(
                _Live(
                    id=arrival.id,
                    prompt_len=arrival.prompt_len,
                    max_tokens=arrival.max_tokens,
                    priority=arrival.priority,
                    arrival_at=arrival.at,
                    arrival_index=index,
                    prefill_remaining=arrival.prompt_len,
                )
            )
        self._by_id = {live.id: live for live in self._pending}
        if len(self._by_id) != len(self._pending):
            raise ValueError("duplicate request ids")

        self.waiting: list[_Live] = []
        self.admitted: list[_Live] = []
        self.finished: list[str] = []

    # ------------------------------------------------------------------ views

    def view(self) -> EngineView:
        waiting = tuple(live.snapshot() for live in self.waiting)
        admitted = tuple(live.snapshot() for live in self.admitted)
        if self.kv_capacity is None:
            return EngineView(max_work=self.max_work, waiting=waiting, admitted=admitted)
        return KvEngineView(
            max_work=self.max_work,
            waiting=waiting,
            admitted=admitted,
            kv_capacity=self.kv_capacity,
        )

    @property
    def kv_held(self) -> int:
        return sum(live.kv for live in self.admitted)

    def done(self) -> bool:
        return not (self._pending or self.waiting or self.admitted)

    # ------------------------------------------------------------------ driving

    def step(self, plan_step) -> StepRecord:
        """Advance one step: arrive, snapshot, plan, validate, apply."""
        self._arrive()

        waiting_before = tuple(live.id for live in self.waiting)
        admitted_before = tuple(live.id for live in self.admitted)
        kv_before = self.kv_held

        plan = plan_step(self.view())
        if not isinstance(plan, StepPlan):
            raise SchedulingError(
                f"step {self.step_index}: plan_step must return a StepPlan, "
                f"got {type(plan).__name__}"
            )
        self._validate(plan)
        finished = self._apply(plan)

        record = StepRecord(
            step=self.step_index,
            waiting=waiting_before,
            admitted=admitted_before,
            kv_held=kv_before,
            preempt=tuple(plan.preempt),
            prefill=tuple(f"{op.id}:{op.tokens}" for op in plan.prefill),
            decode=tuple(plan.decode),
            finished=finished,
        )
        self.step_index += 1
        return record

    def run(self, plan_step, max_steps: int = 100) -> list[StepRecord]:
        """Step until every request has finished, or max_steps is reached."""
        records = []
        while not self.done() and len(records) < max_steps:
            records.append(self.step(plan_step))
        if not self.done():
            raise SchedulingError(
                f"still running after {max_steps} steps - is the schedule making progress?"
            )
        return records

    # ------------------------------------------------------------------ internals

    def _arrive(self) -> None:
        arriving = [live for live in self._pending if live.arrival_at <= self.step_index]
        for live in arriving:
            self._pending.remove(live)
            self._enqueue(live)

    def _enqueue(self, live: _Live) -> None:
        """Insert into waiting at the position its arrival order demands.

        A preempted request rejoins where it originally arrived, not at the
        back of the queue - that is what keeps eviction from starving it.
        """
        position = len(self.waiting)
        for index, other in enumerate(self.waiting):
            if other.arrival_index > live.arrival_index:
                position = index
                break
        self.waiting.insert(position, live)

    def _validate(self, plan: StepPlan) -> None:
        where = f"step {self.step_index}"
        waiting_ids = {live.id for live in self.waiting}
        admitted_ids = {live.id for live in self.admitted}

        self._check_unique(plan.preempt, "preempt", where)
        self._check_unique(plan.decode, "decode", where)
        self._check_unique([op.id for op in plan.prefill], "prefill", where)

        for rid in plan.preempt:
            if rid not in admitted_ids:
                raise SchedulingError(
                    f"{where}: cannot preempt {rid!r} - only admitted requests can be "
                    f"preempted (admitted: {sorted(admitted_ids)})"
                )

        preempted = set(plan.preempt)
        for rid in plan.decode:
            if rid not in admitted_ids:
                raise SchedulingError(
                    f"{where}: cannot decode {rid!r} - it is not admitted "
                    f"(admitted: {sorted(admitted_ids)})"
                )
            if rid in preempted:
                raise SchedulingError(
                    f"{where}: {rid!r} is in both preempt and decode - a victim does no "
                    f"work on the step it is evicted"
                )
            if self._by_id[rid].prefill_remaining != 0:
                raise SchedulingError(
                    f"{where}: cannot decode {rid!r} - it still has "
                    f"{self._by_id[rid].prefill_remaining} prefill tokens outstanding"
                )

        for op in plan.prefill:
            if op.id not in self._by_id:
                raise SchedulingError(f"{where}: unknown request {op.id!r} in prefill")
            live = self._by_id[op.id]
            if op.id in preempted:
                raise SchedulingError(
                    f"{where}: {op.id!r} is in both preempt and prefill"
                )
            if op.id not in waiting_ids and op.id not in admitted_ids:
                raise SchedulingError(
                    f"{where}: cannot prefill {op.id!r} - it is neither waiting nor admitted"
                )
            if op.id in admitted_ids and live.prefill_remaining == 0:
                raise SchedulingError(
                    f"{where}: cannot prefill {op.id!r} - it is already fully prefilled "
                    f"(it should decode instead)"
                )
            if op.tokens < 1:
                raise SchedulingError(
                    f"{where}: prefill of {op.id!r} must be at least 1 token, got {op.tokens}"
                )
            if op.tokens > live.prefill_remaining:
                raise SchedulingError(
                    f"{where}: prefill of {op.id!r} is {op.tokens} tokens but only "
                    f"{live.prefill_remaining} remain"
                )
            if not self.chunked and op.tokens != live.prefill_remaining:
                raise SchedulingError(
                    f"{where}: prefill of {op.id!r} is {op.tokens} tokens but chunking is "
                    f"off - a prefill must cover the full remaining "
                    f"{live.prefill_remaining} tokens"
                )

        work = len(plan.decode) + sum(op.tokens for op in plan.prefill)
        if work > self.max_work:
            raise SchedulingError(
                f"{where}: plan uses {work} units of work but max_work is {self.max_work} "
                f"({len(plan.decode)} decode + "
                f"{sum(op.tokens for op in plan.prefill)} prefill)"
            )

        if self.kv_capacity is not None:
            projected = {live.id: live.kv for live in self.admitted}
            for rid in plan.preempt:
                projected.pop(rid, None)
            for rid in plan.decode:
                projected[rid] += 1
            for op in plan.prefill:
                projected[op.id] = projected.get(op.id, 0) + op.tokens
            total = sum(projected.values())
            if total > self.kv_capacity:
                raise SchedulingError(
                    f"{where}: plan needs {total} KV slots but kv_capacity is "
                    f"{self.kv_capacity} - remember a decode grows its request by one "
                    f"slot, and a request that finishes this step still holds its slots"
                )

        idle = not plan.decode and not plan.prefill
        if idle and (self.waiting or self.admitted):
            raise SchedulingError(
                f"{where}: no decode or prefill scheduled, but requests are waiting or "
                f"admitted - the schedule cannot make progress"
            )

    @staticmethod
    def _check_unique(ids, field_name: str, where: str) -> None:
        seen = set()
        for rid in ids:
            if rid in seen:
                raise SchedulingError(f"{where}: {rid!r} appears twice in {field_name}")
            seen.add(rid)

    def _apply(self, plan: StepPlan) -> tuple[str, ...]:
        # 1. Preemption first: frees memory immediately, and the victim does no
        #    work this step.
        for rid in plan.preempt:
            live = self._by_id[rid]
            live.prefill_remaining = live.prompt_len + live.generated
            live.kv = 0
            self.admitted.remove(live)
            self._enqueue(live)

        # 2. Decode: one token each, one more KV slot each.
        for rid in plan.decode:
            live = self._by_id[rid]
            live.kv += 1
            live.generated += 1

        # 3. Prefill: admits waiting requests, and completing a prefill
        #    produces the request's first output token.
        for op in plan.prefill:
            live = self._by_id[op.id]
            if live in self.waiting:
                self.waiting.remove(live)
                self.admitted.append(live)
            live.kv += op.tokens
            live.prefill_remaining -= op.tokens
            if live.prefill_remaining == 0:
                live.generated += 1

        # 4. Anything that hit max_tokens is done; its memory frees after the step.
        finished = []
        for live in list(self.admitted):
            if live.generated >= live.max_tokens:
                self.admitted.remove(live)
                live.kv = 0
                finished.append(live.id)
                self.finished.append(live.id)
        return tuple(finished)
