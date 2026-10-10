from rcl.frontend import plots

TRAJ = {
    "time": [0.0, 0.1, 0.2],
    "states": [[0.0, 1.0], [0.1, 1.1], [0.2, 1.2]],
    "actions": [[0.5], [0.4]],
    "rewards": [1.0, 1.0],
    "state_labels": ["x", "v"],
    "state_units": ["m", "m/s"],
    "action_labels": ["force"],
    "action_units": ["N"],
}


def test_trajectory_figure_has_a_panel_per_signal_with_api_labels():
    fig = plots.trajectory_figure(TRAJ)
    assert len(fig.data) == 4  # 2 states + 1 action + reward
    assert [a.text for a in fig.layout.annotations] == ["x [m]", "v [m/s]", "force [N]", "reward"]
    assert list(fig.data[2].x) == [0.0, 0.1]  # action has T samples


def test_unlabelled_signals_get_fallback_names():
    fig = plots.trajectory_figure({**TRAJ, "state_labels": [], "state_units": []})
    assert [a.text for a in fig.layout.annotations][:2] == ["state 0", "state 1"]


def test_curve_and_tables():
    assert len(plots.curve_figure([[10, 1.0], [20, 2.0]]).data[0].x) == 2
    run = {
        "episodes": [{"index": 1, "success": True}, {"index": 0, "success": False}],
        "metrics": [
            {
                "metric_name": "return",
                "episode_index": None,
                "value": 3.0,
                "unit": "r",
                "status": "ok",
            },
            {
                "metric_name": "return",
                "episode_index": 0,
                "value": 2.0,
                "unit": "r",
                "status": "ok",
            },
            {
                "metric_name": "return",
                "episode_index": 1,
                "value": 4.0,
                "unit": "r",
                "status": "ok",
            },
        ],
    }
    assert plots.run_metrics_table(run) == [
        {"metric": "return", "value": 3.0, "unit": "r", "status": "ok"}
    ]
    assert [(r["index"], r["return"]) for r in plots.episode_table(run)] == [(0, 2.0), (1, 4.0)]
