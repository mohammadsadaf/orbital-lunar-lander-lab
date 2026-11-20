from rocketlab.evaluate import evaluate, rollout


def test_random_evaluation_is_reproducible_and_unshaped(tmp_path):
    a, replay = evaluate(None, episodes=2, seed=910, output=tmp_path)
    b, _ = evaluate(None, episodes=2, seed=910, capture=False)
    assert a == b
    assert a["wrapped"] is False
    assert a["solved"] is False
    assert len(a["results"]) == 2
    assert replay["frames"]
    assert len(replay["frames"][0]["obs"]) == 8
    assert replay["frames"][-1]["score"] == round(a["results"][0]["score"], 2)
    assert (tmp_path / "replay.json").exists()


def test_rollout_respects_time_limit_and_records_actions():
    row, replay = rollout(None, seed=800, capture=True)
    assert sum(row["actions"]) == row["length"]
    assert 1 <= row["length"] <= 1000
    assert len(replay["frames"]) == row["length"]
