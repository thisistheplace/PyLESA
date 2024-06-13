from enum import Enum
import logging

from ..io.enums import SingleTypeCheck

LOG = logging.getLogger(__name__)


class Insulation(str, Enum, metaclass=SingleTypeCheck):
    POLYURETHANE = "POLYURETHANE"
    FIBREGLASS = "FIBREGLASS"
    POLYSTYRENE = "POLYSTYRENE"


class AmbientLocation(str, Enum, metaclass=SingleTypeCheck):
    INSIDE = "INSIDE"
    OUTSIDE = "OUTSIDE"
