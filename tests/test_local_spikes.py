"""End-to-end tests for the read-only local spike detection endpoint."""
from __future__ import annotations

import json
import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("SENSOR_DB_PATH", db_path)
    from app import database, main

    main.app.state.db_path = db_path
    database.init_db(db_path)
    with TestClient(main.app) as c:
        yield c, db_path


def create_session(client, sources=("a", "b"), lateness=10.0, session_id="s"):
    r = client.post(
        "/sessions",
        json={
            "session_id": session_id,
            "source_ids": list(sources),
            "allowed_lateness_seconds": lateness,
        },
    )
    assert r.status_code == 201, r.text
    return session_id


def t(value: str) -> str:
    from app.time_utils import parse_event_time, store_event_time

    return store_event_time(parse_event_time(value))


def insert_row(
    db_path,
    sid,
    source,
    seq,
    observed_at,
    payload,
    *,
    status="published",
):
    """Insert one reading row with an explicit stored JSON payload."""
    if not isinstance(payload, str):
        payload = json.dumps(payload)
    with sqlite3.connect(db_path) as raw:
        raw.execute(
            "INSERT INTO readings (session_id, source_id, seq, observed_at,"
            " payload, status, first_received_at, published_watermark,"
            " quarantined_watermark)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                sid,
                source,
                seq,
                t(observed_at),
                payload,
                status,
                t(observed_at),
                t(observed_at),
                None,
            ),
        )
        raw.commit()


def insert_series(db_path, sid, source, points, *, start="2026-01-01T10:00:00Z"):
    """Insert ``points`` as seq 1..n spaced one second apart from ``start``."""
    from datetime import timedelta

    from app.time_utils import parse_event_time, store_event_time

    base = parse_event_time(start)
    for i, value in enumerate(points, start=1):
        ts = store_event_time(base + timedelta(seconds=i - 1))
        payload = json.dumps(value)
        with sqlite3.connect(db_path) as raw:
            raw.execute(
                "INSERT INTO readings (session_id, source_id, seq, observed_at,"
                " payload, status, first_received_at, published_watermark,"
                " quarantined_watermark)"
                " VALUES (?, ?, ?, ?, ?, 'published', ?, ?, NULL)",
                (sid, source, i, ts, payload, ts, ts),
            )
            raw.commit()


def spikes_call(client, sid, **params):
    return client.get(f"/sessions/{sid}/local-spikes", params=params)


def call(
    client,
    sid,
    source="a",
    lo="2026-01-01T10:00:00Z",
    hi="2026-01-01T11:00:00Z",
    window="10",
    sensitivity="3",
    **extra,
):
    params = {
        "source_id": source,
        "from": lo,
        "to": hi,
        "window_seconds": window,
        "sensitivity": sensitivity,
    }
    params.update(extra)
    return spikes_call(client, sid, **params)


class TestSpikeDetection:
    def test_clear_spike_flagged(self, client):
        c, db_path = client
        sid = create_session(c, sources=("a",))
        insert_series(db_path, sid, "a", [10, 10, 10, 10, 50, 10, 10, 10, 10])
        body = call(c, sid, window="2").json()
        # The 50 has 4 neighbours (two each side within 2s): 10,10,10,10 ->
        # median 10, MAD 0, deviation 40 > 0 -> spike.  The two outermost
        # readings each see only two neighbours and are not judged.
        assert body["valid_readings"] == 9
        assert body["insufficient_neighbours"] == 2
        assert len(body["spikes"]) == 1
        spike = body["spikes"][0]
        assert spike["seq"] == 5
        assert spike["value"] == 50.0
        assert spike["local_median"] == 10.0
        assert spike["mad"] == 0.0
        assert spike["deviation"] == 40.0

    def test_normal_variation_not_flagged(self, client):
        c, db_path = client
        sid = create_session(c, sources=("a",))
        # Clear +/-20% oscillation: every reading's deviation is at most the
        # MAD, so nothing is ever 3*MAD away.  A wide window also avoids
        # edge-of-window effects.
        insert_series(db_path, sid, "a", [10, 12, 8, 10, 12, 8, 10, 12, 8])
        body = call(c, sid, window="100").json()
        assert body["valid_readings"] == 9
        assert body["insufficient_neighbours"] == 0
        assert body["spikes"] == []

    def test_mad_zero_equal_value_not_a_spike(self, client):
        c, db_path = client
        sid = create_session(c, sources=("a",))
        # Constant series: MAD 0 and deviation 0 -> 0 > 0 is false.
        insert_series(db_path, sid, "a", [5, 5, 5, 5, 5])
        body = call(c, sid, window="10").json()
        assert body["spikes"] == []
        assert body["insufficient_neighbours"] == 0

    def test_strict_comparison_boundary_not_spike(self, client):
        c, db_path = client
        sid = create_session(c, sources=("a",))
        # Centre 13; neighbours 9,10,11,12,14,15,16,17 sorted:
        # median = 13, abs devs: 4,3,2,1,1,2,3,4 -> MAD 2.5.
        # deviation 0 is not a spike; set centre to 20.5 -> dev 7.5 exactly
        # equals 3 * 2.5 = 7.5: strict greater means NOT a spike.
        insert_series(
            db_path,
            sid,
            "a",
            [9, 10, 11, 12, 20.5, 14, 15, 16, 17],
        )
        body = call(c, sid, window="4", sensitivity="3").json()
        assert body["spikes"] == []
        # Just over the boundary is a spike.
        with sqlite3.connect(db_path) as raw:
            raw.execute(
                "UPDATE readings SET payload = ? WHERE seq = 5",
                (json.dumps(20.500001),),
            )
            raw.commit()
        body = call(c, sid, window="4", sensitivity="3").json()
        assert [s["seq"] for s in body["spikes"]] == [5]

    def test_sensitivity_scales_the_threshold(self, client):
        c, db_path = client
        sid = create_session(c, sources=("a",))
        # Four calm 9s, a 16, then four calm 11s.  Neighbours of the 16 within
        # 2s are 9,9,11,11 -> median 10, MAD 1, deviation 6.  A level-10
        # multiplier tolerates it (6 <= 10); a level-5 multiplier flags it
        # (6 > 5).  The surrounding readings have positive-MAD neighbourhoods
        # so they never spike themselves.
        insert_series(
            db_path,
            sid,
            "a",
            [9, 9, 9, 9, 16, 11, 11, 11, 11],
        )
        body = call(c, sid, window="2", sensitivity="10").json()
        assert body["spikes"] == []
        body = call(c, sid, window="2", sensitivity="5").json()
        assert [s["seq"] for s in body["spikes"]] == [5]
        spike = body["spikes"][0]
        assert spike["local_median"] == 10.0
        assert spike["mad"] == 1.0
        assert spike["deviation"] == 6.0


class TestNeighbourhood:
    def test_fewer_than_three_neighbours_not_judged(self, client):
        c, db_path = client
        sid = create_session(c, sources=("a",))
        insert_series(db_path, sid, "a", [10, 50, 10])
        body = call(c, sid, window="10").json()
        assert body["valid_readings"] == 3
        # Each reading sees the other two (< 3): nothing judged, no spikes.
        assert body["insufficient_neighbours"] == 3
        assert body["spikes"] == []

    def test_exactly_three_neighbours_judged(self, client):
        c, db_path = client
        sid = create_session(c, sources=("a",))
        # Irregular spacing with a 1s window: one reading has exactly three
        # neighbours (distance <= 1s) and is judged; every other valid reading
        # has fewer and is left unjudged.
        from app.time_utils import store_event_time, parse_event_time
        from datetime import timedelta

        base = parse_event_time("2026-01-01T10:00:00Z")
        points = [(0, 10), (1, 10), (2, 50), (3, 10), (3.5, 10), (4, 10)]
        for seq, (offset, value) in enumerate(points, start=1):
            ts = store_event_time(base + timedelta(seconds=offset))
            with sqlite3.connect(db_path) as raw:
                raw.execute(
                    "INSERT INTO readings (session_id, source_id, seq,"
                    " observed_at, payload, status, first_received_at,"
                    " published_watermark, quarantined_watermark)"
                    " VALUES ('s','a',?,?,?,'published',?,?,NULL)",
                    (seq, ts, json.dumps(value), ts, ts),
                )
                raw.commit()
        body = call(c, sid, window="1").json()
        # seq=4 @:03 sees :02(50), :04(10), :04.5(10) -> exactly 3
        # neighbours; it is the only judged reading (median 10, MAD 0,
        # deviation 0 -> not itself a spike).
        assert body["valid_readings"] == 6
        assert body["insufficient_neighbours"] == 5
        assert body["spikes"] == []

    def test_window_is_symmetric_and_bounded(self, client):
        c, db_path = client
        sid = create_session(c, sources=("a",))
        # Seven readings spaced 2s apart; the 50 sits at :06.  A 2s window
        # gives it only the two adjacent readings (:04, :08) -> not judged.
        from app.time_utils import store_event_time, parse_event_time
        from datetime import timedelta

        base = parse_event_time("2026-01-01T10:00:00Z")
        for i, value in enumerate([10, 10, 10, 50, 10, 10, 10], start=1):
            ts = store_event_time(base + timedelta(seconds=2 * (i - 1)))
            with sqlite3.connect(db_path) as raw:
                raw.execute(
                    "INSERT INTO readings (session_id, source_id, seq,"
                    " observed_at, payload, status, first_received_at,"
                    " published_watermark, quarantined_watermark)"
                    " VALUES ('s','a',?,?,?,'published',?,?,NULL)",
                    (i, ts, json.dumps(value), ts, ts),
                )
                raw.commit()
        body = call(c, sid, window="2").json()
        assert all(s["seq"] != 4 for s in body["spikes"])
        seq4_status = next(
            (s for s in body["spikes"] if s["seq"] == 4), None
        )
        assert seq4_status is None
        # A 4s window reaches the next readings out (:02 and :10), giving the
        # 50 four neighbours -> judged, MAD 0 -> spike.
        body = call(c, sid, window="4").json()
        assert [s["seq"] for s in body["spikes"]] == [4]

    def test_window_endpoint_is_inclusive(self, client):
        c, db_path = client
        sid = create_session(c, sources=("a",))
        # Readings every 2 seconds: 10 @:00,:02,:04 and 50 @:06, then 10
        # @:08,:10,:12.  With window=4s the 50 sees exactly three neighbours
        # at distance 2 and 4 (:02,:04,:08,:10 are four actually) -- use
        # window=2 to test the inclusive edge: :04 and :08 at distance 2.
        from app.time_utils import store_event_time, parse_event_time
        from datetime import timedelta

        base = parse_event_time("2026-01-01T10:00:00Z")
        for i, value in enumerate([10, 10, 10, 50, 10, 10, 10], start=1):
            ts = store_event_time(base + timedelta(seconds=2 * (i - 1)))
            with sqlite3.connect(db_path) as raw:
                raw.execute(
                    "INSERT INTO readings (session_id, source_id, seq,"
                    " observed_at, payload, status, first_received_at,"
                    " published_watermark, quarantined_watermark)"
                    " VALUES ('s','a',?,?,?,'published',?,?,NULL)",
                    (i, ts, json.dumps(value), ts, ts),
                )
                raw.commit()
        # window=2: 50@:06 sees :04 and :08 (distance exactly 2s) -> only
        # 2 neighbours -> not judged.
        body = call(c, sid, window="2").json()
        assert body["insufficient_neighbours"] >= 1
        assert all(s["seq"] != 4 for s in body["spikes"])
        # window=2.000001: the same two are in, still only 2; window=4 brings
        # in four neighbours (:02 distance 4 inclusive) -> judged, spike.
        body = call(c, sid, window="4").json()
        assert [s["seq"] for s in body["spikes"]] == [4]

    def test_interval_bounds_are_inclusive(self, client):
        c, db_path = client
        sid = create_session(c, sources=("a",))
        insert_series(db_path, sid, "a", [10, 10, 10, 50, 10, 10, 10])
        # Restrict to exactly [03:00, 05:00]: the 50 at :04 has only two
        # neighbours (:03, :05) -> insufficient.
        body = call(
            c,
            sid,
            lo="2026-01-01T10:00:03Z",
            hi="2026-01-01T10:00:05Z",
            window="10",
        ).json()
        assert body["valid_readings"] == 3
        assert body["insufficient_neighbours"] == 3
        assert body["spikes"] == []

    def test_single_instant_interval_allowed(self, client):
        c, db_path = client
        sid = create_session(c, sources=("a",))
        insert_series(db_path, sid, "a", [10, 10, 50])
        body = call(
            c,
            sid,
            lo="2026-01-01T10:00:02Z",
            hi="2026-01-01T10:00:02Z",
            window="10",
        )
        assert body.status_code == 200, body.text
        data = body.json()
        assert data["valid_readings"] == 1
        assert data["insufficient_neighbours"] == 1
        assert data["spikes"] == []

    def test_neighbours_only_come_from_interval_valid_readings(self, client):
        c, db_path = client
        sid = create_session(c, sources=("a",))
        # Baseline 10 before the interval; a 50 at the interval start cannot
        # use readings outside the closed interval as neighbours.
        insert_series(
            db_path,
            sid,
            "a",
            [10, 10, 10, 50, 50, 50],
        )
        body = call(
            c,
            sid,
            lo="2026-01-01T10:00:03Z",
            hi="2026-01-01T10:00:06Z",
            window="100",
        ).json()
        assert body["valid_readings"] == 3
        # All three are 50: median 50, MAD 0, deviation 0 -> no spikes.
        assert body["spikes"] == []

    def test_median_uses_average_of_two_middle_values(self, client):
        c, db_path = client
        sid = create_session(c, sources=("a",))
        # 4 neighbours exactly: 10, 12, 14, 100 -> median 13.
        # Reading 20 at :04; readings at :02,:03,:05,:06 within window 2.
        from app.time_utils import store_event_time, parse_event_time
        from datetime import timedelta

        base = parse_event_time("2026-01-01T10:00:00Z")
        points = {2: 10, 3: 12, 4: 20, 5: 14, 6: 100}
        for seq, value in points.items():
            ts = store_event_time(base + timedelta(seconds=seq))
            with sqlite3.connect(db_path) as raw:
                raw.execute(
                    "INSERT INTO readings (session_id, source_id, seq,"
                    " observed_at, payload, status, first_received_at,"
                    " published_watermark, quarantined_watermark)"
                    " VALUES ('s','a',?,?,?,'published',?,?,NULL)",
                    (seq, ts, json.dumps(value), ts, ts),
                )
                raw.commit()
        body = call(
            c,
            sid,
            window="2",
            lo="2026-01-01T10:00:00Z",
            hi="2026-01-01T10:00:10Z",
        ).json()
        # :04 (20) neighbours 10,12,14,100: median 13; abs devs 3,1,1,87 ->
        # MAD = 2.0; deviation 7 > 3*2=6 -> spike.  median reported 13.0.
        spike4 = next(s for s in body["spikes"] if s["seq"] == 4)
        assert spike4["local_median"] == 13.0
        assert spike4["mad"] == 2.0
        assert spike4["deviation"] == 7.0


class TestInvalidValues:
    def test_non_finite_boolean_missing_skipped(self, client):
        c, db_path = client
        sid = create_session(c, sources=("a",))
        insert_row(db_path, sid, "a", 1, "2026-01-01T10:00:00Z", 10)
        insert_row(db_path, sid, "a", 2, "2026-01-01T10:00:01Z", float("nan"))
        insert_row(db_path, sid, "a", 3, "2026-01-01T10:00:02Z", float("inf"))
        insert_row(db_path, sid, "a", 4, "2026-01-01T10:00:03Z", True)
        insert_row(db_path, sid, "a", 5, "2026-01-01T10:00:04Z", None)
        insert_row(db_path, sid, "a", 6, "2026-01-01T10:00:05Z", "hot")
        insert_row(db_path, sid, "a", 7, "2026-01-01T10:00:06Z", 10)
        insert_row(db_path, sid, "a", 8, "2026-01-01T10:00:07Z", 10)
        insert_row(db_path, sid, "a", 9, "2026-01-01T10:00:08Z", 10)
        body = call(c, sid, window="100").json()
        assert body["valid_readings"] == 4
        # The four 10s are seconds 0, 6, 7, 8.  Each sees the other three
        # (window 100): median 10, MAD 0, deviation 0 -> no spikes; all
        # judged.
        assert body["insufficient_neighbours"] == 0
        assert body["spikes"] == []

    def test_value_key_extracts_top_level_key(self, client):
        c, db_path = client
        sid = create_session(c, sources=("a",))
        for i, temp in enumerate([20, 20, 20, 80, 20, 20, 20], start=1):
            insert_row(
                db_path,
                sid,
                "a",
                i,
                f"2026-01-01T10:00:0{i-1}Z",
                {"temp": temp, "unit": "C"},
            )
        body = call(c, sid, window="2", value_key="temp").json()
        assert body["valid_readings"] == 7
        assert [s["seq"] for s in body["spikes"]] == [4]
        assert body["spikes"][0]["value"] == 80.0

    def test_value_key_missing_or_boolean_skipped(self, client):
        c, db_path = client
        sid = create_session(c, sources=("a",))
        insert_row(db_path, sid, "a", 1, "2026-01-01T10:00:00Z", {"temp": 20})
        insert_row(db_path, sid, "a", 2, "2026-01-01T10:00:01Z", {"other": 1})
        insert_row(db_path, sid, "a", 3, "2026-01-01T10:00:02Z", {"temp": True})
        insert_row(db_path, sid, "a", 4, "2026-01-01T10:00:03Z", 20)
        insert_row(db_path, sid, "a", 5, "2026-01-01T10:00:04Z", {"temp": 20})
        insert_row(db_path, sid, "a", 6, "2026-01-01T10:00:05Z", {"temp": 20})
        insert_row(db_path, sid, "a", 7, "2026-01-01T10:00:06Z", {"temp": 20})
        body = call(c, sid, window="100", value_key="temp").json()
        # Valid: seqs 1, 5, 6, 7 (seq 4 is a bare number, not an object).
        assert body["valid_readings"] == 4
        assert body["insufficient_neighbours"] == 0
        assert body["spikes"] == []

    def test_invalid_rows_count_toward_200_limit(self, client):
        c, db_path = client
        sid = create_session(c, sources=("a",))
        for i in range(1, 202):
            insert_row(
                db_path,
                sid,
                "a",
                i,
                f"2026-01-01T10:{i // 60:02d}:{i % 60:02d}Z",
                None,
            )
        r = call(c, sid, window="10")
        assert r.status_code == 422
        assert "200" in r.json()["detail"]


class TestStatusesAndOrder:
    def test_buffered_and_quarantined_never_participate(self, client):
        c, db_path = client
        sid = create_session(c, sources=("a",))
        insert_row(db_path, sid, "a", 1, "2026-01-01T10:00:00Z", 10)
        insert_row(db_path, sid, "a", 2, "2026-01-01T10:00:01Z", 10)
        insert_row(db_path, sid, "a", 3, "2026-01-01T10:00:02Z", 50,
                   status="buffered")
        insert_row(db_path, sid, "a", 4, "2026-01-01T10:00:03Z", 50,
                   status="quarantined")
        insert_row(db_path, sid, "a", 5, "2026-01-01T10:00:04Z", 10)
        insert_row(db_path, sid, "a", 6, "2026-01-01T10:00:05Z", 10)
        body = call(c, sid, window="100").json()
        assert body["valid_readings"] == 4
        assert body["spikes"] == []

    def test_spikes_sorted_by_observed_at_then_seq(self, client):
        c, db_path = client
        sid = create_session(c, sources=("a",))
        # Two spikes separated by calm readings.
        insert_series(
            db_path,
            sid,
            "a",
            [10, 10, 10, 50, 10, 10, 10, 60, 10, 10, 10],
        )
        body = call(c, sid, window="2").json()
        assert [s["seq"] for s in body["spikes"]] == [4, 8]
        assert body["spikes"][0]["observed_at"] < body["spikes"][1]["observed_at"]

    def test_same_event_time_ordered_by_seq(self, client):
        c, db_path = client
        sid = create_session(c, sources=("a",))
        ts = "2026-01-01T10:00:00Z"
        # Five readings at the same timestamp; distance 0 keeps every pair in
        # the window.  One odd one out among four 10s is a spike.
        insert_row(db_path, sid, "a", 1, ts, 10)
        insert_row(db_path, sid, "a", 2, ts, 10)
        insert_row(db_path, sid, "a", 3, ts, 42)
        insert_row(db_path, sid, "a", 4, ts, 10)
        insert_row(db_path, sid, "a", 5, ts, 10)
        body = call(c, sid, window="1").json()
        assert [s["seq"] for s in body["spikes"]] == [3]
        # Each 10 has neighbours: four others including 42; with 4
        # neighbours [10,10,10,42] median 10, MAD 0 -> deviation 0, no spike.
        assert body["insufficient_neighbours"] == 0

    def test_other_source_readings_never_neighbours(self, client):
        c, db_path = client
        sid = create_session(c, sources=("a", "b"))
        insert_row(db_path, sid, "a", 1, "2026-01-01T10:00:00Z", 50)
        for i in range(2, 9):
            insert_row(
                db_path,
                sid,
                "a",
                i,
                f"2026-01-01T10:00:0{i-1}Z",
                10,
            )
        # Lots of b readings that must not rescue a's 50.
        for i in range(1, 20):
            insert_row(
                db_path,
                sid,
                "b",
                i,
                f"2026-01-01T10:00:{i % 10:02d}Z",
                10,
            )
        body = call(c, sid, source="a", window="100").json()
        assert body["valid_readings"] == 8
        # a's 50 is at :00 with neighbours only later a-readings (:01..:06 in
        # the window) -> judged and spiking.
        assert [s["seq"] for s in body["spikes"]] == [1]


class TestErrors:
    def test_unknown_session_404(self, client):
        c, _ = client
        r = call(c, "nope")
        assert r.status_code == 404

    def test_unknown_source_400(self, client):
        c, _ = client
        sid = create_session(c, sources=("a",))
        r = call(c, sid, source="zzz")
        assert r.status_code == 400
        assert "source" in r.json()["detail"]

    @pytest.mark.parametrize("field", ["source_id", "from", "to",
                                       "window_seconds", "sensitivity"])
    def test_missing_params_400(self, client, field):
        c, _ = client
        sid = create_session(c, sources=("a",))
        params = {
            "source_id": "a",
            "from": "2026-01-01T10:00:00Z",
            "to": "2026-01-01T11:00:00Z",
            "window_seconds": "10",
            "sensitivity": "3",
        }
        del params[field]
        r = spikes_call(c, sid, **params)
        assert r.status_code == 400, r.text

    @pytest.mark.parametrize("bad", ["0", "-1", "-0.5", "nan", "inf",
                                     "-inf", "abc", " ", "1e999"])
    def test_bad_window_400(self, client, bad):
        c, _ = client
        sid = create_session(c, sources=("a",))
        r = call(c, sid, window=bad)
        assert r.status_code == 400

    @pytest.mark.parametrize("bad", ["0", "-1", "-0.5", "nan", "inf",
                                     "-inf", "abc", " "])
    def test_bad_sensitivity_400(self, client, bad):
        c, _ = client
        sid = create_session(c, sources=("a",))
        r = call(c, sid, sensitivity=bad)
        assert r.status_code == 400

    def test_inverted_interval_400(self, client):
        c, _ = client
        sid = create_session(c, sources=("a",))
        r = call(
            c,
            sid,
            lo="2026-01-01T11:00:00Z",
            hi="2026-01-01T10:00:00Z",
        )
        assert r.status_code == 400

    @pytest.mark.parametrize("field", ["from", "to"])
    def test_unparsable_bounds_400(self, client, field):
        c, _ = client
        sid = create_session(c, sources=("a",))
        kwargs = {field: "not-a-time"}
        r = call(c, sid, **kwargs)
        assert r.status_code == 400

    def test_blank_source_400(self, client):
        c, _ = client
        sid = create_session(c, sources=("a",))
        r = call(c, sid, source="   ")
        assert r.status_code == 400

    def test_blank_value_key_400(self, client):
        c, _ = client
        sid = create_session(c, sources=("a",))
        r = call(c, sid, value_key="  ")
        assert r.status_code == 400

    def test_offset_timestamp_normalised_in_response(self, client):
        c, db_path = client
        sid = create_session(c, sources=("a",))
        insert_series(db_path, sid, "a", [10, 10, 10, 10, 10])
        r = call(
            c,
            sid,
            lo="2026-01-01T12:00:00+02:00",
            hi="2026-01-01T13:00:00+02:00",
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["from"] == "2026-01-01T10:00:00Z"
        assert body["to"] == "2026-01-01T11:00:00Z"


class TestReadOnlySnapshot:
    def test_call_changes_nothing(self, client):
        c, db_path = client
        sid = create_session(c, sources=("a", "b"))
        insert_series(db_path, sid, "a", [10, 10, 10, 50, 10, 10, 10])
        insert_row(db_path, sid, "b", 1, "2026-01-01T10:00:00Z", 1,
                   status="buffered")

        def snapshot():
            with sqlite3.connect(db_path) as raw:
                raw.row_factory = sqlite3.Row
                return [
                    dict(r)
                    for r in raw.execute(
                        "SELECT source_id, seq, status, payload FROM readings"
                        " ORDER BY source_id, seq"
                    )
                ]

        before = snapshot()
        r = call(c, sid, source="a", window="2")
        assert r.status_code == 200, r.text
        assert snapshot() == before

    def test_reads_one_snapshot_during_uncommitted_write(self, client):
        c, db_path = client
        sid = create_session(c, sources=("a",))
        insert_series(db_path, sid, "a", [10, 10, 10, 50, 10, 10, 10])

        from app import database

        holder = {}

        def worker():
            writer = database.connect(db_path)
            try:
                writer.execute("BEGIN IMMEDIATE")
                writer.execute(
                    "INSERT INTO readings (session_id, source_id, seq,"
                    " observed_at, payload, status, first_received_at,"
                    " published_watermark, quarantined_watermark)"
                    " VALUES ('s','a',99,?,?, 'published', ?, ?, NULL)",
                    (
                        t("2026-01-01T10:00:03Z"),
                        json.dumps(999),
                        t("2026-01-01T10:00:03Z"),
                        t("2026-01-01T10:00:03Z"),
                    ),
                )
                holder["started"].set()
                holder["release"].wait(2)
                writer.rollback()
            finally:
                writer.close()

        holder["started"] = threading.Event()
        holder["release"] = threading.Event()
        thread = threading.Thread(target=worker)
        thread.start()
        holder["started"].wait(2)
        try:
            # The writer's uncommitted 999 must not be visible in the snapshot.
            body = call(c, sid, source="a", window="2").json()
            assert body["valid_readings"] == 7
            assert all(s["value"] != 999 for s in body["spikes"])
        finally:
            holder["release"].set()
            thread.join()
