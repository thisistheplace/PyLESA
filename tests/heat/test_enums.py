from enum import Enum
import pytest

from pylesa.heat.enums import HP, ModelName, DataInput
from pylesa.io.enums import SingleTypeCheck


class TestHP:
    def test_metaclass(self):
        assert isinstance(HP, SingleTypeCheck)

    @pytest.mark.parametrize("hp", ["ASHP", "GSHP", "WSHP"])
    def test_hp_options(self, hp):
        assert hp in HP


class TestModelName:
    def test_metaclass(self):
        assert isinstance(ModelName, SingleTypeCheck)

    @pytest.mark.parametrize(
        "model", ["Simple", "Lorentz", "Generic regression", "Standard test regression"]
    )
    def test_model_options(self, model):
        assert model.upper() in ModelName


class TestDataInput:
    def test_metaclass(self):
        assert isinstance(DataInput, SingleTypeCheck)

    @pytest.mark.parametrize("data", ["INTEGRATED PERFORMANCE", "PEAK PERFORMANCE"])
    def test_data_options(self, data):
        assert data in DataInput
