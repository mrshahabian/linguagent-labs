"""Render a run as the same step table the level specs use.

Being able to print your own trace next to the spec's worked example is the
fastest debugging loop there is: you see the exact step where they diverge
instead of guessing from a failed assertion.

    from scheduler import Arrival, Engine, render
    from solutions.level1 import plan_step

    engine = Engine([Arrival("A", 6, 3), Arrival("B", 3, 2)], max_work=10)
    print(render(engine.run(plan_step)))
"""

from __future__ import annotations

from .engine import StepRecord

DASH = "-"


def _cell(values) -> str:
    return ", ".join(values) if values else DASH


def render(records: list[StepRecord], kv: bool = False) -> str:
    """Return a markdown-ish table of a run. Set kv=True for the memory levels."""
    headers = ["step", "waiting", "admitted"]
    if kv:
        headers.append("kv held")
    headers += ["preempt", "prefill", "decode", "finished"]

    rows = []
    for record in records:
        row = [str(record.step), _cell(record.waiting), _cell(record.admitted)]
        if kv:
            row.append(str(record.kv_held))
        row += [
            _cell(record.preempt),
            _cell(record.prefill),
            _cell(record.decode),
            _cell(record.finished),
        ]
        rows.append(row)

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    lines = [
        "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)).rstrip(),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
    return "\n".join(lines)


def as_rows(records: list[StepRecord], kv: bool = False) -> list[tuple]:
    """The same data as tuples, for comparing against an expected table in a test."""
    rows = []
    for record in records:
        row = [record.step, record.waiting, record.admitted]
        if kv:
            row.append(record.kv_held)
        row += [record.preempt, record.prefill, record.decode, record.finished]
        rows.append(tuple(row))
    return rows
