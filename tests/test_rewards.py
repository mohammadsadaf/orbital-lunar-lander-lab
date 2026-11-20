import numpy as np
import pytest
import gymnasium as gym

from rocketlab.rewards import RewardConfig, ShapedReward, potential, reward_terms, make_env


def observation(y=.6):
    return np.array([0, y, 0, 0, 0, 0, 0, 0], dtype=np.float32)


def test_stationary_hover_collects_bonus_but_fix_does_not():
    obs = observation()
    bad = reward_terms(obs, obs, False, RewardConfig(mode="hover"))
    fixed = reward_terms(obs, obs, False, RewardConfig(mode="potential"))
    assert sum(bad.values()) == 1
    assert sum(fixed.values()) < 0


def test_discounted_potential_telescopes_through_terminal_state():
    config = RewardConfig(mode="potential", time_cost=0, gamma=.99)
    path = [observation(1), observation(.7), observation(.2), observation(0)]
    discounted = sum(config.gamma**i * reward_terms(path[i], path[i+1], i == 2, config)["potential"] for i in range(3))
    assert discounted == pytest.approx(-potential(path[0], config.potential_scale))


def test_time_limit_retains_continuation_potential():
    before, after = observation(.6), observation(.5)
    config = RewardConfig(mode="potential")
    # A truncation must use terminated=False and retain the nonzero next Phi.
    continuing = reward_terms(before, after, False, config)["potential"]
    terminal = reward_terms(before, after, True, config)["potential"]
    assert continuing != terminal
    assert terminal == pytest.approx(-potential(before))


@pytest.mark.parametrize("kwargs", [{"mode":"fake"}, {"hover_bonus":-1}, {"time_cost":float('nan')}, {"gamma":0}, {"gamma":1.1}])
def test_bad_config_rejected(kwargs):
    with pytest.raises(ValueError):
        RewardConfig(**kwargs)


def test_wrapper_matches_original_environment_and_reports_both_returns():
    plain = gym.make("LunarLander-v3")
    wrapped = make_env(RewardConfig(mode="time", time_cost=.1))
    try:
        a, _ = plain.reset(seed=42)
        b, _ = wrapped.reset(seed=42)
        np.testing.assert_array_equal(a, b)
        base, shaped, count = 0, 0, 0
        while True:
            a, r, term, trunc, _ = plain.step(0)
            b, s, term2, trunc2, info = wrapped.step(0)
            np.testing.assert_array_equal(a, b)
            assert (term, trunc) == (term2, trunc2)
            assert s == pytest.approx(r-.1)
            base += r
            shaped += s
            count += 1
            if term or trunc:
                break
        assert info["lab_episode"]["base"] == pytest.approx(base)
        assert info["lab_episode"]["shaped"] == pytest.approx(shaped)
        assert info["lab_episode"]["length"] == count
        wrapped.reset(seed=43)
        assert wrapped.length == 0
        assert wrapped.base_return == 0
    finally:
        plain.close()
        wrapped.close()
