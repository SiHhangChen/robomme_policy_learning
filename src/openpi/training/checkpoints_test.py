import dataclasses

from openpi.training.checkpoints import _merge_params
from openpi.training.checkpoints import _split_params


@dataclasses.dataclass
class _State:
    params: dict
    ema_params: dict | None


def test_split_and_merge_params_without_ema():
    state = _State(params={"weight": 1}, ema_params=None)

    train_state, params = _split_params(state)
    restored = _merge_params(train_state, {"params": params})

    assert train_state.params == {}
    assert restored == state


def test_split_and_merge_params_with_ema():
    state = _State(params={"weight": 1}, ema_params={"weight": 2})

    train_state, params = _split_params(state)
    restored = _merge_params(train_state, {"params": params})

    assert train_state.params == state.params
    assert train_state.ema_params is None
    assert restored == state
