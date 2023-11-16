import pytest
from src.tests.v1.conftest import get_order_number
from src.tests.v1.conftest import dataholder
from src.models.lstm import LSTM


@pytest.mark.order(get_order_number("test_lstm"))
def test_lstm():
    lstm = LSTM(1, 12)
    assert str(lstm.__class__) == "<class 'src.models.lstm.LSTM'>"
    assert lstm.horizon == 1
    assert lstm.window_size == 12
    assert lstm.model == None
    
    lstm.build()
    assert str(type(lstm.model)) == "<class 'keras.src.engine.functional.Functional'>"

    # After fitting the model should tested against some values etc...
    # we better check all the functionality for the models including saving, loading etc, as well as the results

