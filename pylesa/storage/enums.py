from enum import Enum, IntEnum
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


class ChargingState(IntEnum):
    STANDBY = 0
    CHARGING = 1
    DISCHARGING = 2