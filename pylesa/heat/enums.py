from enum import Enum
import logging

from ..io.enums import SingleTypeCheck

LOG = logging.getLogger(__name__)


class HP(str, Enum, metaclass=SingleTypeCheck):
    ASHP = "ASHP"
    GSHP = "GSHP"
    WSHP = "WSHP"


class ModelName(str, Enum, metaclass=SingleTypeCheck):
    SIMPLE = "SIMPLE"
    LORENTZ = "LORENTZ"
    GENERIC = "GENERIC REGRESSION"
    STANDARD = "STANDARD TEST REGRESSION"


class DataInput(str, Enum, metaclass=SingleTypeCheck):
    INTEGRATED = "INTEGRATED PERFORMANCE"
    PEAK = "PEAK PERFORMANCE"


class Fuel(str, Enum, metaclass=SingleTypeCheck):
    GAS = "GAS"
    WOOD = "WOOD CHIPS"
    KEROSENE = "KEROSENE"
    ELECTRIC = "ELECTRIC"
