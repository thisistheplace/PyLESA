from enum import EnumMeta
import pytest

from pylesa.storage.enums import AmbientLocation, Insulation, ChargingState
from pylesa.io.enums import SingleTypeCheck


class TestInsulation:
    def test_metaclass(self):
        assert isinstance(Insulation, SingleTypeCheck)

    @pytest.mark.parametrize("insultype", ["POLYURETHANE", "FIBREGLASS", "POLYSTYRENE"])
    def test_insulation_options(self, insultype):
        assert insultype in Insulation


class TestAmbientLocation:
    def test_metaclass(self):
        assert isinstance(AmbientLocation, SingleTypeCheck)

    @pytest.mark.parametrize("location", ["INSIDE", "OUTSIDE"])
    def test_model_options(self, location):
        assert location in AmbientLocation


class TestChargingState:
    def test_metaclass(self):
        assert isinstance(ChargingState, EnumMeta)

    @pytest.mark.parametrize("state", [0, 1, 2])
    def test_model_options(self, state):
        options = [
            ChargingState.STANDBY,
            ChargingState.CHARGING,
            ChargingState.DISCHARGING,
        ]
        assert state == options[state]
