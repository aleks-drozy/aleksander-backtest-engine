import pytest
import pandas as pd
from engine.base_strategy import Strategy


def test_cannot_instantiate_abstract_strategy():
    with pytest.raises(TypeError):
        Strategy()  # type: ignore


def test_concrete_strategy_must_implement_generate_signals():
    class Incomplete(Strategy):
        id = "INCOMPLETE"
        name = "Incomplete"
        description = "Missing generate_signals."
        direction = "long_only"
        params = {}

    with pytest.raises(TypeError):
        Incomplete()


def test_concrete_strategy_valid():
    class AlwaysLong(Strategy):
        id = "ALWAYS_LONG"
        name = "Always Long"
        description = "Always holds long."
        direction = "long_only"
        params = {}

        def generate_signals(self, df: pd.DataFrame) -> pd.Series:
            return pd.Series(1, index=df.index)

    s = AlwaysLong()
    assert s.id == "ALWAYS_LONG"
    assert s.direction == "long_only"
