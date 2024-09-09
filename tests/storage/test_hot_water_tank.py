from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, Mock

from pylesa.storage.enums import AmbientLocation, Insulation, ChargingState
from pylesa.storage.hot_water_tank import HotWaterTank


@dataclass
class CoeffSpec:
    tank: HotWaterTank
    state: str
    node: int
    flow: float
    source: float
    source_delta_t: float
    treturn: float
    timestep: int
    thermal_output: float
    demand: float

    @property
    def cf(self):
        if self.state == ChargingState.CHARGING:
            temp = self.source
        else:
            temp = self.flow
        return self.tank.charging_function(self.state, self.nodes_temp, temp)

    @property
    def mass_flow(self):
        return self.tank.node_mass

    @property
    def nodes_temp(self):
        return self.tank.init_temps(10.0)

    @property
    def temp_tank_bottom(self):
        return self.nodes_temp[-1]

    @property
    def temp_tank_top(self):
        return self.nodes_temp[0]


@pytest.fixture
def tank() -> HotWaterTank:
    return HotWaterTank(
        capacity=100,
        insulation=Insulation.POLYURETHANE,
        location=AmbientLocation.INSIDE,
        number_nodes=4,
        dimensions={"width": 5},  # height and insulation_thickness are overwritten!
        tank_openings={
            "tank_opening": 1.0,
            "tank_opening_diameter": 1.5,
            "uninsulated_connections": 2,
            "uninsulated_connections_diameter": 0.5,
            "insulated_connections": 2,
            "insulated_connections_diameter": 0.2,
        },
        correction_factors={"insulation_factor": 2.0, "overall_factor": 3.0},
        air_temperature=None,
    )


class TestTank:
    def test_init(self, tank: HotWaterTank):
        assert isinstance(tank, HotWaterTank)

    def test_node_mass(self, tank: HotWaterTank):
        assert tank.node_mass == 100 / 4.0

    def test_insulation_k_value(self, tank: HotWaterTank):
        assert tank.insulation_k_value == 0.025 * 3600

    def test_insulation_k_value_error(self, tank: HotWaterTank):
        tank.insulation = "bad"
        with pytest.raises(KeyError):
            tank.insulation_k_value

    def test_specific_heat(self, tank: HotWaterTank):
        # test values from Isobaric Cp here:
        # https://www.engineeringtoolbox.com/specific-heat-capacity-water-d_660.html
        assert round(tank.specific_heat_water(100), 0) == 4216.0
        assert round(tank.specific_heat_water(70), 0) == 4190.0
        assert round(tank.specific_heat_water(60), 0) == 4185.0
        assert round(tank.specific_heat_water("default")) == 4180.0

    @pytest.mark.parametrize("temp", [-1.0, 101.0])
    def test_specific_heat_bad_temp(self, tank: HotWaterTank, temp: float):
        with pytest.raises(ValueError):
            tank.specific_heat_water(temp)

    def test_internal_radius(self, tank: HotWaterTank):
        assert tank.internal_radius == (
            2 * (tank.capacity / (2.5 * math.pi)) ** (1.0 / 3)
        ) * (7 / 8)


class TestOutside:
    def test_outside_missing_air_temp(self):
        # Raises value error due to missing air temperature with OUTSIDE location
        with pytest.raises(ValueError):
            HotWaterTank(
                capacity=100,
                insulation=Insulation.POLYURETHANE,
                location=AmbientLocation.OUTSIDE,
                number_nodes=4,
                dimensions={
                    "width": 5
                },  # height and insulation_thickness are overwritten!
                tank_openings={
                    "tank_opening": 1.0,
                    "tank_opening_diameter": 1.5,
                    "uninsulated_connections": 2,
                    "uninsulated_connections_diameter": 0.5,
                    "insulated_connections": 2,
                    "insulated_connections_diameter": 0.2,
                },
                correction_factors={"insulation_factor": 2.0, "overall_factor": 3.0},
                air_temperature=None,
            )


class TestAmbientTemp:
    def test_inside_ambient_temp(self, tank: HotWaterTank):
        assert tank.amb_temp(0) == 15.0

    def test_outside_ambient_temp(self, tank: HotWaterTank):
        tank.location = AmbientLocation.OUTSIDE
        tank.air_temperature = pd.DataFrame.from_dict(
            {"air_temperature": [1.0, 2.0, 3.0]}
        )
        assert tank.amb_temp(1) == 2.0

    def test_ambient_temp_bad_location(self, tank: HotWaterTank):
        tank.location = None
        with pytest.raises(ValueError):
            tank.amb_temp(0)


class TestChargingFunction:
    def test_top_charging(self, tank: HotWaterTank):
        node_temps = [60.0, 60.0, 60.0, 60.0]
        state = tank.charging_function(ChargingState.CHARGING, node_temps, 70.0)
        # Top node is CHARGING
        assert np.allclose(state, [1, 0, 0, 0])

    def test_mid_charging(self, tank: HotWaterTank):
        node_temps = [71.0, 71.0, 60.0, 60.0]
        state = tank.charging_function(ChargingState.CHARGING, node_temps, 70.0)
        # Mid node is CHARGING
        assert np.allclose(state, [0, 0, 1, 0])

    def test_bottom_charging(self, tank: HotWaterTank):
        node_temps = [71.0, 71.0, 71.0, 60.0]
        state = tank.charging_function(ChargingState.CHARGING, node_temps, 70.0)
        # Mid node is CHARGING
        assert np.allclose(state, [0, 0, 0, 1])

    def test_no_charging(self, tank: HotWaterTank):
        node_temps = [71.0, 71.0, 71.0, 71.0]
        state = tank.charging_function(ChargingState.CHARGING, node_temps, 70.0)
        # Mid node is CHARGING
        assert np.allclose(state, [0, 0, 0, 0])

    def test_top_discharging(self, tank: HotWaterTank):
        node_temps = [60.0, 60.0, 60.0, 60.0]
        state = tank.charging_function(ChargingState.DISCHARGING, node_temps, 50.0)
        # Top node is discharging
        assert np.allclose(state, [2, 0, 0, 0])

    def test_mid_discharging(self, tank: HotWaterTank):
        node_temps = [49.0, 49.0, 60.0, 60.0]
        state = tank.charging_function(ChargingState.DISCHARGING, node_temps, 50.0)
        # Mid node is discharging
        assert np.allclose(state, [0, 0, 2, 0])

    def test_bottom_discharging(self, tank: HotWaterTank):
        node_temps = [49.0, 49.0, 49.0, 60.0]
        state = tank.charging_function(ChargingState.DISCHARGING, node_temps, 50.0)
        # Mid node is discharging
        assert np.allclose(state, [0, 0, 0, 2])

    def test_no_discharging(self, tank: HotWaterTank):
        node_temps = [49.0, 49.0, 49.0, 49.0]
        state = tank.charging_function(ChargingState.DISCHARGING, node_temps, 50.0)
        # Mid node is discharging
        assert np.allclose(state, [0, 0, 0, 0])

    def test_standby(self, tank: HotWaterTank):
        node_temps = [60.0, 60.0, 60.0, 60.0]
        got = tank.charging_function(ChargingState.STANDBY, node_temps, 70.0)
        assert np.allclose(got, 0)


class TestCoefficients:
    @pytest.fixture
    def spec(self, tank: HotWaterTank):
        return CoeffSpec(
            tank=tank,
            state=ChargingState.CHARGING,
            node=1,
            flow=65.0,
            source=70.0,
            source_delta_t=10.0,
            treturn=40.0,
            timestep=1,
            demand=350,
            thermal_output=350,
        )

    @patch.object(HotWaterTank, "mass_flow_calc")
    def test_set_of_coefficients(
        self, mock_mass_flow: Mock, spec: CoeffSpec, tank: HotWaterTank
    ):
        # Mock mass_flow_calc to return the same value as node mass
        mock_mass_flow.return_value = tank.node_mass

        assert tank.set_of_max_coefficients(
            spec.state,
            spec.nodes_temp,
            spec.source,
            spec.flow,
            spec.treturn,
            spec.timestep,
        ) == tank.set_of_coefficients(
            spec.state,
            spec.nodes_temp,
            spec.source,
            spec.source_delta_t,
            spec.flow,
            spec.treturn,
            spec.thermal_output,
            spec.demand,
            spec.temp_tank_bottom,
            spec.temp_tank_top,
            spec.timestep,
        )

    def test_coefficient_A(self, spec: CoeffSpec, tank: HotWaterTank):
        assert tank.coefficient_A(
            spec.state, spec.node, spec.nodes_temp, spec.mass_flow, spec.cf
        )

    def test_coefficient_B(self, spec: CoeffSpec, tank: HotWaterTank):
        assert tank.coefficient_B(spec.state, spec.node, spec.mass_flow, spec.cf) == 1

    def test_coefficient_C(self, spec: CoeffSpec, tank: HotWaterTank):
        assert tank.coefficient_C(spec.state, spec.node, spec.mass_flow, spec.cf) == 0

    def test_coefficient_D(self, spec: CoeffSpec, tank: HotWaterTank):
        assert tank.coefficient_D(
            spec.node,
            spec.nodes_temp,
            spec.mass_flow,
            spec.source,
            spec.flow,
            spec.treturn,
            spec.timestep,
            spec.cf,
        )