"""Tests for publication figure rendering helpers."""

from __future__ import annotations

import pandas as pd
import numpy as np

import paper.scripts.render_paper_figures as renderer
from paper.scripts.render_paper_figures import _line_style, _pivot_heatmap


def test_pivot_heatmap_orders_budget_and_restore_budget() -> None:
    df = pd.DataFrame(
        [
            {"base_context_budget": 18432, "k": 96, "idlekv_lift": 0.7, "gold_headroom": 0.1},
            {"base_context_budget": 14336, "k": 16, "idlekv_lift": -0.01, "gold_headroom": 0.4},
            {"base_context_budget": 14336, "k": 96, "idlekv_lift": 0.6, "gold_headroom": 0.2},
            {"base_context_budget": 18432, "k": 16, "idlekv_lift": 0.02, "gold_headroom": 0.3},
        ]
    )

    budgets, ks, values, headroom = _pivot_heatmap(df)

    assert budgets == [14336, 18432]
    assert ks == [16, 96]
    np.testing.assert_allclose(values, [[0.0, 0.75], [0.0625, 0.875]])
    assert headroom.tolist() == [[0.4, 0.2], [0.3, 0.1]]


def test_line_styles_use_distinct_marker_channels() -> None:
    styles = {name: _line_style(name) for name in ["IdleKV", "Gold-K", "Matched", "Random-K", "Oldest-K"]}

    assert styles["IdleKV"]["marker"] != styles["Gold-K"]["marker"]
    assert styles["Matched"]["linestyle"] != styles["Random-K"]["linestyle"]
    assert styles["Oldest-K"]["color"] != styles["Random-K"]["color"]


def test_policy_breadth_renderer_requires_streamingllm_full_grid(tmp_path, monkeypatch) -> None:
    figure_dir = tmp_path / "figures"
    phase11_dir = tmp_path / "phase11"
    phase12_dir = tmp_path / "phase12"
    figure_dir.mkdir()
    phase11_dir.mkdir()
    phase12_dir.mkdir()

    snapkv_csv = figure_dir / "phase7_clean_suite_b16384_exact_overall.csv"
    h2o_csv = phase11_dir / "h2o_4q_fullgrid_n24.csv"
    for path in (snapkv_csv, h2o_csv):
        path.write_text(
            "k,b_match,idlekv,random_k,oldest_k,gold_k\n"
            "8,0.20,0.21,0.20,0.20,0.40\n"
            "32,0.20,0.45,0.20,0.21,0.70\n"
            "96,0.20,0.80,0.20,0.20,1.00\n"
            "128,0.20,0.90,0.21,0.20,1.00\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(renderer, "FIGURE_DIR", figure_dir)
    monkeypatch.setattr(renderer, "PHASE11_DIR", phase11_dir)
    monkeypatch.setattr(renderer, "PHASE12_DIR", phase12_dir)
    renderer.configure_matplotlib()

    assert renderer.render_policy_breadth_delta() is False
    assert not (figure_dir / "policy_breadth_delta.pdf").exists()

    (phase12_dir / "streamingllm_4q_fullgrid_n24_b16384.csv").write_text(
        "k,b_match,idlekv,random_k,oldest_k,gold_k\n"
        "8,0.30,0.30,0.30,0.30,0.35\n"
        "32,0.30,0.40,0.30,0.31,0.60\n"
        "96,0.30,0.70,0.30,0.30,0.90\n"
        "128,0.30,0.85,0.31,0.30,0.95\n",
        encoding="utf-8",
    )

    assert renderer.render_policy_breadth_delta() is True
    assert (figure_dir / "policy_breadth_delta.pdf").exists()
    assert (figure_dir / "policy_breadth_delta.png").exists()


def test_multiturn_renderer_prefers_phase13_locked_query_only_controls(tmp_path, monkeypatch) -> None:
    figure_dir = tmp_path / "figures"
    phase10_dir = tmp_path / "phase10"
    phase13_dir = tmp_path / "phase13"
    figure_dir.mkdir()
    phase10_dir.mkdir()
    phase13_dir.mkdir()

    rows = [
        ("Matched", 64, 0, 0.50),
        ("Matched", 64, 1, 0.00),
        ("IdleKV", 64, 0, 0.50),
        ("IdleKV", 64, 1, 0.75),
        ("CurrentQOnly-K", 64, 0, 0.50),
        ("CurrentQOnly-K", 64, 1, 0.70),
        ("StaleQOnly-K", 64, 0, 0.50),
        ("StaleQOnly-K", 64, 1, 0.10),
        ("Random-K", 64, 0, 0.50),
        ("Random-K", 64, 1, 0.00),
        ("Oldest-K", 64, 0, 0.50),
        ("Oldest-K", 64, 1, 0.00),
        ("Gold-K", 64, 0, 0.50),
        ("Gold-K", 64, 1, 1.00),
    ]
    csv_lines = ["example_index,turn,k,condition,score"]
    csv_lines.extend(f"0,{turn},{k},{condition},{score}" for condition, k, turn, score in rows)
    (phase13_dir / "multiturn_hard_locked_rows_n24_k64.csv").write_text(
        "\n".join(csv_lines) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(renderer, "FIGURE_DIR", figure_dir)
    monkeypatch.setattr(renderer, "PHASE10_DIR", phase10_dir)
    monkeypatch.setattr(renderer, "PHASE13_DIR", phase13_dir)
    renderer.configure_matplotlib()

    assert renderer.render_multiturn_hard_trajectory() is True
    assert (figure_dir / "multiturn_hard_trajectory.pdf").exists()
    assert (figure_dir / "multiturn_hard_trajectory.png").exists()
