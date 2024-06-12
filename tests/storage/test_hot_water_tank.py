from dataclasses import dataclass
import pytest
from unittest.mock import patch, Mock

from pylesa.storage.hot_water_tank import HotWaterTank, AmbientLocation, Insulation


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
        return self.tank.charging_function(
            self.state, self.nodes_temp, self.source, self.node
        )

    @property
    def df(self):
        return self.tank.discharging_function(
            self.state, self.nodes_temp, self.flow, self.node
        )

    @property
    def mass_flow(self):
        return self.tank.calc_node_mass()

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
        assert tank.calc_node_mass() == 100 / 4.0


class TestCoefficients:
    @pytest.fixture
    def spec(self, tank: HotWaterTank):
        return CoeffSpec(
            tank=tank,
            state="charging",
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
        mock_mass_flow.return_value = tank.calc_node_mass()

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
            spec.state, spec.node, spec.nodes_temp, spec.mass_flow, spec.cf, spec.df
        )

    def test_coefficient_B(self, spec: CoeffSpec, tank: HotWaterTank):
        assert (
            tank.coefficient_B(spec.state, spec.node, spec.mass_flow, spec.cf, spec.df)
            == 0
        )

    def test_coefficient_C(self, spec: CoeffSpec, tank: HotWaterTank):
        assert (
            tank.coefficient_C(spec.state, spec.node, spec.mass_flow, spec.cf, spec.df)
            == 0
        )

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
            spec.df,
        )
