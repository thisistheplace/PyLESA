from enum import Enum
import pytest

from pylesa.io.enums import SingleTypeCheck


class Dummy(str, Enum, metaclass=SingleTypeCheck):
    FIRST = "first"
    SECOND = "second"


class TestSingleTypeCheck:
    def test_contains(self):
        assert "first" in Dummy
        assert "second" in Dummy
        assert "third" not in Dummy

    def test_from_value(self):
        assert Dummy.from_value("first") == Dummy.FIRST
        assert Dummy.from_value("second") == Dummy.SECOND
        with pytest.raises(KeyError):
            Dummy.from_value("third")
