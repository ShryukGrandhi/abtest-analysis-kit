import json

import numpy as np
import pytest

from abkit.cli import main
from abkit.data import aggregate_rows, load_events
from abkit.report import analyze, to_markdown
from abkit.simulate import SimConfig, simulate_users, write_events_csv


def test_aggregate_rows_quality_accounting():
    rows = [
        ["u1", "A", "assign", "", ""],
        ["u2", "B", "assign", "", ""],
        ["u3", "A", "assign", "", ""],
        ["u3", "B", "assign", "", ""],         # conflicting assignment
        ["u1", "A", "post", "rev", "2.5"],
        ["u1", "A", "post", "rev", "1.5"],
        ["u1", "A", "pre", "rev", "4"],
        ["u2", "B", "post", "rev", "7"],
        ["u2", "A", "post", "rev", "9"],      # variant mismatch
        ["u9", "A", "post", "rev", "1"],      # never assigned
        ["u3", "A", "post", "rev", "1"],      # conflicted user
        ["u1", "A", "post", "rev", "abc"],    # malformed value
        ["u1", "A", "during", "rev", "1"],    # unknown period
        None,                                  # short row
    ]
    t = aggregate_rows(rows)
    assert t.user_ids == ["u1", "u2"]
    assert t.counts() == {"A": 1, "B": 1}
    assert t.post["rev"].tolist() == [4.0, 7.0]
    assert t.pre["rev"].tolist() == [4.0, 0.0]
    q = t.quality
    assert q.rows_read == 14
    assert q.multi_assigned_users == 1
    assert q.variant_mismatch_rows == 2
    assert q.unassigned_user_rows == 1
    assert q.malformed_rows == 2
    assert q.unknown_period_rows == 1


def test_csv_roundtrip_matches_simulated_totals(tmp_path):
    cfg = SimConfig(n_users=800)
    path = tmp_path / "ev.csv"
    users = write_events_csv(path, cfg, seed=3)
    t = load_events(path)
    assert t.n_users == 800
    for m in ("sessions", "orders", "revenue"):
        np.testing.assert_allclose(t.post[m], users[f"{m}_post"], atol=1e-4)
        np.testing.assert_allclose(t.pre[m], users[f"{m}_pre"], atol=1e-4)


def test_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("user_id,variant\nu1,A\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_events(p)


def test_analyze_finds_planted_effect(tmp_path):
    cfg = SimConfig(n_users=60000, conversion_lift={"treatment": 1.3})
    path = tmp_path / "ev.csv"
    write_events_csv(path, cfg, seed=11, dirty_fraction=0.01)
    t = load_events(path)
    a = analyze(t, "control", ratios=[("orders", "sessions")])
    by = {r.metric: r for r in a.results}
    assert not a.srm_mismatch
    assert by["orders"].p_value < 1e-6
    assert by["orders/sessions"].rel_ci_low < 0.3 < by["orders/sessions"].rel_ci_high
    assert by["sessions"].cuped and by["sessions"].variance_reduction > 0.2
    assert "unassigned_user_rows" in a.quality
    md = to_markdown(a)
    assert "orders/sessions" in md


def test_analyze_flags_srm():
    rng = np.random.default_rng(0)
    users = simulate_users(SimConfig(n_users=40000, weights=(0.52, 0.48)), rng)
    rows = []
    for i, code in enumerate(users["variant"]):
        rows.append([f"u{i}", ("control", "treatment")[code], "assign", "", ""])
    t = aggregate_rows(rows)
    a = analyze(t, "control", metrics=[])
    assert a.srm_mismatch


def test_cli_json(tmp_path, capsys):
    path = tmp_path / "ev.csv"
    assert main(["simulate", str(path), "--users", "3000", "--seed", "1"]) == 0
    assert main(["analyze", str(path), "--control", "control", "--json",
                 "--ratio", "revenue/sessions"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["counts"]["control"] + out["counts"]["treatment"] == 3000
    assert {r["metric"] for r in out["results"]} == {"orders", "revenue", "sessions", "revenue/sessions"}
