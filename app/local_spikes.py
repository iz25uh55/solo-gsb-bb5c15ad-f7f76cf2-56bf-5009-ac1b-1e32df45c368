"""Read-only local spike detection over one source's published readings.

``GET /sessions/{session_id}/local-spikes`` applies a robust local outlier
test to a single declared source's **published** readings inside a
**closed** event-time interval ``[from, to]``.  Only ``published`` rows
participate -- buffered and quarantined readings are never visible -- and
the whole query/computation runs over one consistent deferred SQLite read
snapshot without writing anything.

A reading participates only when its value (or the named top-level object
key) is a **finite, non-boolean number**; missing payloads/keys, strings,
booleans and NaN/Infinity are skipped.  For every participating reading
the neighbours are the *other* participating readings whose event-time
distance is no greater than ``window_seconds`` (both endpoints count).
With fewer than three neighbours the reading is not judged; otherwise the
neighbours' median and median absolute deviation (MAD) describe the local
baseline and the reading is a spike when its absolute deviation from the
median is **strictly greater** than ``sensitivity * MAD``.  When the MAD is
zero the rule still applies: any strictly positive deviation is a spike.

To keep the neighbourhood work bounded, an interval containing more than
200 published readings (valid values or not) is rejected up front.
"""
from __future__ import annotations

import json
import math
import sqlite3
import statistics
from typing import Any, Optional

from .alerts import extract_numeric
from .database import read_transaction
from .errors import BadRequestError, UnprocessableEntityError
from .service import _load_session
from .time_utils import format_event_time, parse_event_time, store_event_time

__all__ = ["local_spikes", "MAX_READINGS"]

MAX_READINGS = 200
# Minimum number of *other* in-window readings required to judge one.
MIN_NEIGHBOURS = 3

_EPOCH = parse_event_time("1970-01-01T00:00:00Z")


# ---------------------------------------------------------------------------
# request validation
# ---------------------------------------------------------------------------

def _required_time(value: Any, field: str) -> str:
    """Validate/normalise a required inclusive event-time bound."""
    if value is None:
        raise BadRequestError(f"{field} is required")
    if not isinstance(value, str) or not value.strip():
        raise BadRequestError(f"{field} must be a non-empty ISO 8601 string")
    # TimestampError is mapped to HTTP 400 by the application handler.
    return store_event_time(parse_event_time(value))


def _validate_positive_number(value: Any, field: str) -> float:
    """Parse a required strictly-positive finite query parameter (400 on error).

    Query parameters arrive as strings; a missing value, junk, NaN,
    Infinity, zero or a negative number is a bad request rather than a
    Pydantic 422.
    """
    if value is None:
        raise BadRequestError(f"{field} is required")
    if not isinstance(value, str) or not value.strip():
        raise BadRequestError(f"{field} must be a positive finite number")
    try:
        number = float(value.strip())
    except ValueError as exc:
        raise BadRequestError(
            f"{field} must be a positive finite number, got {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise BadRequestError(f"{field} must be a finite number")
    if number <= 0:
        raise BadRequestError(f"{field} must be strictly greater than 0, got {number}")
    return number


def _epoch_micros(observed_at: str) -> int:
    """Exact integer microseconds since the UTC epoch for a stored timestamp."""
    dt = parse_event_time(observed_at)
    delta = dt - _EPOCH
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


# ---------------------------------------------------------------------------
# spike computation
# ---------------------------------------------------------------------------

def local_spikes(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    source_id: Any,
    from_: Any,
    to: Any,
    window_seconds: Any,
    sensitivity: Any,
    value_key: Any = None,
) -> dict[str, Any]:
    """Return the local spikes of one source's published readings.

    Read-only: every value comes from one deferred read snapshot and no
    table is ever written.  Malformed parameters raise
    :class:`~app.errors.BadRequestError` / :class:`TimestampError` (400);
    an interval holding more than :data:`MAX_READINGS` published readings
    raises :class:`~app.errors.UnprocessableEntityError` (422); an unknown
    session raises 404; an undeclared ``source_id`` raises 400.
    """
    if source_id is None:
        raise BadRequestError("source_id is required")
    if not isinstance(source_id, str) or not source_id.strip():
        raise BadRequestError("source_id must be a non-empty string")
    if value_key is not None:
        if not isinstance(value_key, str) or not value_key.strip():
            raise BadRequestError("value_key must be a non-empty string")
        value_key = value_key.strip()

    from_store = _required_time(from_, "from")
    to_store = _required_time(to, "to")
    # The interval is closed [from, to]; an inverted interval is invalid, but
    # a single-instant interval (from == to) is allowed.
    if from_store > to_store:
        raise BadRequestError(
            f"'from' ({format_event_time(parse_event_time(from_store))}) must not"
            f" be later than 'to' ({format_event_time(parse_event_time(to_store))})"
        )
    window = _validate_positive_number(window_seconds, "window_seconds")
    sensitivity_value = _validate_positive_number(sensitivity, "sensitivity")
    window_micros = int(round(window * 1_000_000))

    with read_transaction(conn):
        session = _load_session(conn, session_id)
        allowed_sources = set(json.loads(session["source_ids"]))
        if source_id not in allowed_sources:
            raise BadRequestError(
                f"unknown source_id {source_id!r} for this session;"
                f" declared sources: {sorted(allowed_sources)}"
            )

        # Stable (observed_at, seq) order.  The published PK
        # (session_id, source_id, seq) makes seq unique per source, so the
        # ordering is total; buffered/quarantined rows are excluded.
        rows = conn.execute(
            "SELECT observed_at, seq, payload FROM readings"
            " WHERE session_id = ? AND source_id = ? AND status = 'published'"
            " AND observed_at >= ? AND observed_at <= ?"
            " ORDER BY observed_at, seq",
            (session_id, source_id, from_store, to_store),
        ).fetchall()

        # Guard the O(n^2) neighbourhood scan: every published reading in the
        # closed interval counts, whether its value is usable or not.
        if len(rows) > MAX_READINGS:
            raise UnprocessableEntityError(
                f"interval contains {len(rows)} published readings for source"
                f" {source_id!r}; at most {MAX_READINGS} are allowed per request"
                " (narrow the from/to interval)"
            )

        # Valid (finite, non-boolean) readings only.  Invalid rows are simply
        # absent: they neither get judged nor act as anybody's neighbour.
        readings: list[dict[str, Any]] = []
        for row in rows:
            number = extract_numeric(row["payload"], value_key)
            if number is None:
                continue
            readings.append(
                {
                    "observed_at": row["observed_at"],
                    "ts_micros": _epoch_micros(row["observed_at"]),
                    "seq": row["seq"],
                    "value": number,
                }
            )

        spikes_out: list[dict[str, Any]] = []
        insufficient_neighbours = 0
        n = len(readings)
        for i, reading in enumerate(readings):
            # Expand outwards in both directions from i: readings is sorted by
            # event time, so each side can stop as soon as the distance exceeds
            # the window.  The reading itself is never its own neighbour.
            neighbours: list[float] = []
            j = i - 1
            while j >= 0:
                distance = reading["ts_micros"] - readings[j]["ts_micros"]
                if distance > window_micros:
                    break
                neighbours.append(readings[j]["value"])
                j -= 1
            j = i + 1
            while j < n:
                distance = readings[j]["ts_micros"] - reading["ts_micros"]
                if distance > window_micros:
                    break
                neighbours.append(readings[j]["value"])
                j += 1

            if len(neighbours) < MIN_NEIGHBOURS:
                # Not enough local context -- no judgement either way.
                insufficient_neighbours += 1
                continue

            median = statistics.median(neighbours)
            mad = statistics.median(abs(v - median) for v in neighbours)
            deviation = abs(reading["value"] - median)
            # Strictly greater; with MAD == 0 any strictly positive deviation
            # spikes (0 > 0 is false, so an exactly-equal reading never does).
            if deviation > sensitivity_value * mad:
                spikes_out.append(
                    {
                        "observed_at": format_event_time(
                            parse_event_time(reading["observed_at"])
                        ),
                        "seq": reading["seq"],
                        "value": reading["value"],
                        "local_median": median,
                        "mad": mad,
                        "deviation": deviation,
                    }
                )

    return {
        "session_id": session_id,
        "source_id": source_id,
        "from": format_event_time(parse_event_time(from_store)),
        "to": format_event_time(parse_event_time(to_store)),
        "window_seconds": window,
        "sensitivity": sensitivity_value,
        "value_key": value_key,
        "valid_readings": n,
        "insufficient_neighbours": insufficient_neighbours,
        "spikes": spikes_out,
    }
