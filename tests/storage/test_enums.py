import pytest

from pylesa.storage.enums import AmbientLocation, Insulation
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
