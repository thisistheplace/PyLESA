"""modelling hot watertanks

lowest capacity is 100L, this can be used as
"no" thermal storage simulation as long as demand is large
"""

from importlib.resources import files as ifiles
import logging
import pandas as pd
import math
from typing import Dict, List

from scipy.integrate import odeint

from .enums import Insulation, AmbientLocation
from ..environment import weather

LOG = logging.getLogger(__name__)


INSULATION_K = {
    Insulation.POLYURETHANE: 0.025,
    Insulation.FIBREGLASS: 0.04,
    Insulation.POLYSTYRENE: 0.035,
}


class HotWaterTank(object):

    def __init__(
        self,
        capacity: float,
        insulation: Insulation,
        location: AmbientLocation,
        number_nodes: int,
        dimensions: Dict[str, float],
        tank_openings: Dict[str, float],
        correction_factors: Dict[str, float],
        air_temperature: pd.DataFrame = None,
    ):
        """hot water tank class object

        Args:
            capacity, capacity in L of tank
            insulation, type of insulation of tank
            location, outside or inside
            number_nodes, number of nodes
            dimensions, {height, width, insul thickness}
            tank_openings, {'tank_opening'
                                    'tank_opening_diameter'
                                    'uninsulated_connections'
                                    'uninsulated_connections_diameter'
                                    'insulated_connections'
                                    'insulated_connections_diameter'}
            correction_factors, insulation factor and overall factor
            air_temperature, Dataframe containing "air_temperature" column defining
                hourly air temperatures, default: None
        """

        # float or str inputs
        self.capacity = capacity
        self.insulation = insulation
        self.location = location
        self.number_nodes = number_nodes
        self.node_list = list(range(self.number_nodes))

        # dic inputs
        self.dimensions = dimensions
        # using new calc for dimensions
        # assuming a ratio of height to width of 2.5
        factor = 2.5
        self.dimensions["width"] = 2 * (self.capacity / (factor * math.pi)) ** (1.0 / 3)
        self.dimensions["height"] = 0.5 * factor * self.dimensions["width"]
        # assuming a ratio of width to insulation thickness
        ins_divider = 8
        self.dimensions["insulation_thickness"] = self.dimensions["width"] / ins_divider

        self.tank_openings = tank_openings
        self.correction_factors = correction_factors

        # optional input, needed if location is set to outside
        self.air_temperature = air_temperature

        self.cp_spec = pd.read_pickle(
            # Use importlib.resources to manage files required by package
            ifiles("pylesa").joinpath("data", "water_spec_heat.pkl")
        )
        # stored in j/kg deg
        self.cp = {
            temp: cp * 1000 for (temp, cp) in zip(self.cp_spec["t"], self.cp_spec["Cp"])
        }

    def init_temps(self, initial_temp):
        return [initial_temp for _ in range(self.number_nodes)]

    def calc_node_mass(self) -> float:
        """Calculates the mass of one node in kg"""
        return float(self.capacity) / self.number_nodes

    def insulation_k_value(self):
        """selects k for insulation

        Returns:
            float -- k-value of insulation W/mK
        """
        if not self.insulation in INSULATION_K:
            msg = f"Insulation {self.insulation} is not valid, must be one of {list(INSULATION_K.keys())}"
            LOG.error(msg)
            raise ValueError(msg)

        # units of k need to be adjusted from W to joules
        # over the hour, and this requires
        # minuts in hour * seconds in minute (60*60)
        return INSULATION_K[self.insulation] * 3600

    def specific_heat_water(self, temp):
        """cp of water

        Arguments:
            temp {float} -- temperature of water

        Returns:
            float -- cp of water at given temp - j/(kg deg C)
        """
        # input temp must be between 0 and 100 deg
        if isinstance(temp, (int, float)):
            if 100.0 >= temp >= 0.0:
                T = round(float(temp), -1)
                cp = self.cp[T]
            else:
                msg = f"Water temperature {temp} is outside of allowable range of 0<=temp<=100"
                LOG.error(msg)
                raise ValueError(msg)
        else:
            cp = 4180

        return cp

    def internal_radius(self):
        """calculates internal radius

        Returns:
            float -- internal radius, m
        """

        r1 = self.dimensions["width"] - self.dimensions["insulation_thickness"]
        return r1

    def amb_temp(self, timestep):
        """ambient temperature surrounding tank

        If location of storage is inside then a 15 deg ambient
        condition is assumed else if location is outside then
        outdoor temperature is used.

        Arguments:
            timestep {int} --

        Returns:
            float -- ambient temp surrounding tank degC
        """
        if self.location == AmbientLocation.OUTSIDE:

            w = weather.Weather(air_temperature=self.air_temperature).hot_water_tank()
            ambient_temp = w["air_temperature"]["air_temperature"][timestep]

        elif self.location == AmbientLocation.INSIDE:
            ambient_temp = 15.0

        else:
            msg = f"Location {self.location} not valid, must be one of {[_.value for _ in AmbientLocation]}"
            LOG.error(msg)
            raise ValueError(msg)
        return ambient_temp

    def discharging_function(
        self, state: str, nodes_temp: List[float], flow_temp: float, node: int = None
    ) -> List[int]:
        """Determine which nodes in the tank are discharging

        If the in mass exceeds the node volume then next node also charged.

        Args:
            state, "charging" or "discharging"
            total nodes, is the number of nodes being modelled
            nodes_temp, is a dict of the nodes and their temperatures
            flow_temp, is the temperature from the storage going into the system
            node, only run function for this node

        Returns:
            list of 1 (discharging) or 0
        """
        total_nodes = self.number_nodes
        out = list(range(len(self.node_list)))

        if node is not None:
            node_list = [node]
        else:
            node_list = self.node_list

        if state == "discharging":

            for idx, node in enumerate(node_list):

                # this asks if we are looking at the top node
                # and if the charging water is above this nodes temp
                if node == 0 and flow_temp <= nodes_temp[0]:
                    out[idx] = 1

                # if the source temp is lower than
                elif node == 0 and flow_temp >= nodes_temp[0]:
                    out[idx] = 0

                # top node then goes in other node
                elif flow_temp < nodes_temp[node] and flow_temp >= nodes_temp[node - 1]:
                    out[idx] = 1

                # for out of bounds nodes, shouldnt occur
                elif node < 0 or node == total_nodes + 1:
                    out[idx] = 0

                else:
                    out[idx] = 0

        elif state == "charging" or state == "standby":
            for idx, node in enumerate(node_list):
                out[idx] = 0

        return out

    def discharging_bottom_node(self, nodes_temp, flow_temp, df):

        total_nodes = self.number_nodes
        out = list(range(len(self.node_list)))
        bottom_node = total_nodes - 1

        if 1 in df:

            for idx, node in enumerate(self.node_list):

                # this asks if we are looking at the bottom node
                if node == bottom_node and nodes_temp[0] >= flow_temp:
                    out[idx] = 1

                elif node == bottom_node and nodes_temp[0] < flow_temp:
                    out[idx] = 0

                else:
                    out[idx] = 0

        else:
            for idx, node in enumerate(self.node_list):
                out[idx] = 0

        return out

    def charging_function(
        self, state: str, nodes_temp: List[float], source_temp: float, node: int = None
    ) -> List[int]:
        """Determine which nodes in the tank are charging

        If the in mass exceeds the node volume then next node also charged.

        Args:
            state, "charging" or "discharging"
            total nodes, is the number of nodes being modelled
            nodes_temp, is a dict of the nodes and their temperatures
            source_temp, is the temperature from the source going into the storage
            node, only run function for this node

        Returns:
            list of 1 (charging) or 0
        """
        total_nodes = self.number_nodes
        out = list(range(len(self.node_list)))

        if node is not None:
            node_list = [node]
        else:
            node_list = self.node_list

        if state == "charging":

            for idx, node in enumerate(node_list):

                # this asks if we are looking at the top node
                # and if the charging water is above this nodes temp
                if node == 0 and source_temp >= nodes_temp[0]:
                    out[idx] = 1

                # if the source temp is lower than
                elif node == 0 and source_temp <= nodes_temp[0]:
                    out[idx] = 0

                # top node then goes in other node
                elif (
                    source_temp >= nodes_temp[node]
                    and source_temp <= nodes_temp[node - 1]
                ):
                    out[idx] = 1

                # for out of bounds nodes, shouldnt occur
                elif node < 0 or node == total_nodes + 1:
                    out[idx] = 0

                else:
                    out[idx] = 0

        elif state == "discharging" or "standby":
            for idx, node in enumerate(node_list):
                out[idx] = 0
        return out

    def charging_top_node(self, state):

        function = {}
        for node in range(self.number_nodes):

            if state == "charging" and node == self.number_nodes - 1:
                function[node] = 1
            else:
                function[node] = 0

        return function

    def mixing_function(self, state, node, cf, df):

        total_nodes = self.number_nodes
        bottom_node = total_nodes - 1

        mf = {}

        if 1 in cf:
            for n in range(self.number_nodes):
                if cf[n] == 1:
                    node_charging = n
        else:
            node_charging = bottom_node + 1
        if 1 in df:
            for n in range(self.number_nodes):
                if df[n] == 1:
                    node_discharging = n
        else:
            node_discharging = bottom_node + 1

        if state == "charging":
            if node <= node_charging:
                mf["Fcnt"] = 0
                mf["Fdnt"] = 0
            else:
                mf["Fcnt"] = 1
                mf["Fdnt"] = 0

            if node == bottom_node or node < node_charging:
                mf["Fcnb"] = 0
                mf["Fdnb"] = 0
            else:
                mf["Fcnb"] = 1
                mf["Fdnb"] = 0
        # discharging
        elif state == "discharging":
            if node == 0 or node <= node_discharging:
                mf["Fcnt"] = 0
                mf["Fdnt"] = 0
            else:
                mf["Fcnt"] = 0
                mf["Fdnt"] = 1

            if node == bottom_node or node < node_discharging:
                mf["Fcnb"] = 0
                mf["Fdnb"] = 0
            else:
                mf["Fcnb"] = 0
                mf["Fdnb"] = 1
        # standby
        else:
            mf["Fcnt"] = 0
            mf["Fdnt"] = 0
            mf["Fcnb"] = 0
            mf["Fdnb"] = 0

        return mf

    def connection_losses(self):

        tank_opening = self.tank_openings["tank_opening"]
        tank_opening_d = self.tank_openings["tank_opening_diameter"]
        uninsulated_connections = self.tank_openings["uninsulated_connections"]
        uninsulated_connections_d = self.tank_openings[
            "uninsulated_connections_diameter"
        ]
        insulated_connections = self.tank_openings["insulated_connections"]
        insulated_connections_d = self.tank_openings["insulated_connections_diameter"]

        # divided by 0.024 to convert from kWh/day to W
        tank_opening_loss = (
            ((tank_opening_d * 0.001) ** 2) * math.pi * tank_opening * 27 / (4 * 0.024)
        )

        uninsulated_connections_loss = (
            uninsulated_connections_d * 0.001 * uninsulated_connections * 5 / 0.024
        )

        insulated_connections_loss = (
            insulated_connections_d * 0.001 * insulated_connections * 3.5 / 0.024
        )

        loss = 3600 * (
            tank_opening_loss
            + uninsulated_connections_loss
            + insulated_connections_loss
        )

        return loss

    def DH_flow_post_mix(self, demand, flow_temp, return_temp):

        cp1 = self.specific_heat_water(flow_temp) / 1000.0
        mass_DH_flow = demand / (cp1 * (flow_temp - return_temp))
        return mass_DH_flow

    def source_out_pre_mix(self, thermal_output, source_temp, source_delta_t):

        cp1 = self.specific_heat_water(source_temp) / 1000.0
        mass_out_pre_mix = thermal_output / (cp1 * source_delta_t)
        return mass_out_pre_mix

    def thermal_storage_mass_charging(
        self,
        thermal_output,
        source_temp,
        source_delta_t,
        return_temp,
        flow_temp,
        demand,
        temp_tank_bottom,
    ):

        ts_mass = (
            self.source_out_pre_mix(thermal_output, source_temp, source_delta_t)
            * source_delta_t
            - self.DH_flow_post_mix(demand, flow_temp, return_temp)
            * (flow_temp - return_temp)
        ) / (source_temp - temp_tank_bottom)
        return ts_mass

    def thermal_storage_mass_discharging(
        self,
        thermal_output,
        source_temp,
        source_delta_t,
        return_temp,
        flow_temp,
        demand,
        temp_tank_top,
    ):

        # mass discharged in every tank timestep
        ts_mass = (
            self.DH_flow_post_mix(demand, flow_temp, return_temp)
            * (flow_temp - return_temp)
            - self.source_out_pre_mix(thermal_output, source_temp, source_delta_t)
            * (source_delta_t)
        ) / (temp_tank_top - return_temp)
        return abs(ts_mass)

    def mass_flow_calc(
        self,
        state,
        flow_temp,
        return_temp,
        source_temp,
        source_delta_t,
        thermal_output,
        demand,
        temp_tank_bottom,
        temp_tank_top,
    ):
        if state == "charging":
            mass_ts = self.thermal_storage_mass_charging(
                thermal_output,
                source_temp,
                source_delta_t,
                return_temp,
                flow_temp,
                demand,
                temp_tank_bottom,
            )
        elif state == "discharging":
            mass_ts = self.thermal_storage_mass_discharging(
                thermal_output,
                source_temp,
                source_delta_t,
                return_temp,
                flow_temp,
                demand,
                temp_tank_top,
            )
        elif state == "standby":
            mass_ts = 0

        return mass_ts

    def coefficient_A(
        self,
        state: str,
        node: int,
        nodes_temp: List[float],
        mass_flow: float,
        cf: List[int],
        df: List[int],
    ):
        """Calculate coefficient A

        Args:
            state, 'charging' or 'discharging'
            node, index in nodes_temp list to calculate coefficient for
            nodes_temp, list of nodal temperatures
            mass_flow, mass flow rate
            cf, list of nodes where value of 1 is charging
            df, list of nodes where value of 1 is discharging

        Returns:
            Value of coefficient A
        """
        node_mass = self.calc_node_mass()

        # specific heat at temperature of node i
        cp = self.specific_heat_water(nodes_temp[node])

        # thermal conductivity of insulation material
        k = self.insulation_k_value()

        # dimensions
        r1 = self.internal_radius()
        r2 = self.dimensions["width"]
        h = self.dimensions["height"]

        # correction factors
        Fi = self.correction_factors["insulation_factor"]
        Fe = self.correction_factors["overall_factor"]
        Fd = df[node]
        mf = self.mixing_function(state, node, cf, df)
        Fco = self.charging_top_node(state)[node]

        A = (
            -Fd * mass_flow * cp
            - mf["Fdnt"] * mass_flow * cp
            - mf["Fcnb"] * mass_flow * cp
            - Fco * mass_flow * cp
            - Fe * Fi * k * ((1) / (r2 - r1)) * math.pi * ((r1**2) + h * (r2 + r1))
        ) / (node_mass * cp)

        return A

    def coefficient_B(self, state, node, mass_flow, cf, df):

        node_mass = self.calc_node_mass()
        mf = self.mixing_function(state, node, cf, df)

        B = mf["Fcnt"] * mass_flow / node_mass

        return B

    def coefficient_C(self, state, node, mass_flow, cf, df):
        node_mass = self.calc_node_mass()
        mf = self.mixing_function(state, node, cf, df)

        C = mf["Fdnb"] * mass_flow / node_mass
        return C

    def coefficient_D(
        self,
        node,
        nodes_temp,
        mass_flow,
        source_temp,
        flow_temp,
        return_temp,
        timestep,
        cf,
        df,
    ):

        node_mass = self.calc_node_mass()

        # specific heat at temperature of node i
        cp = self.specific_heat_water(nodes_temp[node])

        # thermal conductivity of insulation material
        k = self.insulation_k_value()

        # dimensions
        r1 = self.internal_radius()
        r2 = self.dimensions["width"]
        h = self.dimensions["height"]

        # correction factors
        Fi = self.correction_factors["insulation_factor"]
        Fe = self.correction_factors["overall_factor"]

        Fc = cf[node]
        Fdi = self.discharging_bottom_node(nodes_temp, flow_temp, df)[node]
        Ta = self.amb_temp(timestep)

        cl = self.connection_losses()

        D = (
            Fc * mass_flow * cp * source_temp
            + Fdi * mass_flow * cp * return_temp
            + Fe * Fi * k * ((Ta) / (r2 - r1)) * math.pi * ((r1**2) + h * (r2 + r1))
            + Fe * cl
        ) / (node_mass * cp)

        return D

    def set_of_coefficients(
        self,
        state,
        nodes_temp,
        source_temp,
        source_delta_t,
        flow_temp,
        return_temp,
        thermal_output,
        demand,
        temp_tank_bottom,
        temp_tank_top,
        timestep,
    ):

        mass_flow1 = self.mass_flow_calc(
            state,
            flow_temp,
            return_temp,
            source_temp,
            source_delta_t,
            thermal_output,
            demand,
            temp_tank_bottom,
            temp_tank_top,
        )
        # errors may lead to slight overestimation of maximum mass flow
        # this accounts for this and ensures not going over node mass
        mass_flow = min(mass_flow1, self.calc_node_mass())

        return self._set_of_coefficients(
            state, nodes_temp, source_temp, flow_temp, return_temp, timestep, mass_flow
        )

    def set_of_max_coefficients(
        self, state, nodes_temp, source_temp, flow_temp, return_temp, timestep
    ):
        # Mass flow rate
        node_mass = self.calc_node_mass()
        return self._set_of_coefficients(
            state, nodes_temp, source_temp, flow_temp, return_temp, timestep, node_mass
        )

    def _set_of_coefficients(
        self,
        state,
        nodes_temp,
        source_temp,
        flow_temp,
        return_temp,
        timestep,
        mass_flow,
    ):
        # Charging and discharging data
        cf = self.charging_function(state, nodes_temp, source_temp)
        df = self.discharging_function(state, nodes_temp, flow_temp)

        # Initialize output list
        out = list(range(self.number_nodes))
        for idx, node in enumerate(range(self.number_nodes)):
            coefficients = {
                "A": self.coefficient_A(state, node, nodes_temp, mass_flow, cf, df),
                "B": self.coefficient_B(state, node, mass_flow, cf, df),
                "C": self.coefficient_C(state, node, mass_flow, cf, df),
                "D": self.coefficient_D(
                    node,
                    nodes_temp,
                    mass_flow,
                    source_temp,
                    flow_temp,
                    return_temp,
                    timestep,
                    cf,
                    df,
                ),
            }
            out[idx] = coefficients
        return out

    def new_nodes_temp(
        self,
        state,
        nodes_temp,
        source_temp,
        source_delta_t,
        flow_temp,
        return_temp,
        thermal_output,
        demand,
        timestep,
    ):
        if self.capacity == 0:
            return nodes_temp

        check = 0.0
        for node in range(len(nodes_temp)):
            check += nodes_temp[node]
        if check == source_temp * len(nodes_temp) and state == "charging":
            return nodes_temp * len(nodes_temp)

        def model_temp(z, t, c):
            dzdt = list(range(self.number_nodes))
            for idx, node in enumerate(range(self.number_nodes)):

                if node == 0:
                    Ti = nodes_temp[node]
                    Ti_b = nodes_temp[node + 1]

                    dTdt = c[node]["A"] * Ti + c[node]["C"] * Ti_b + c[node]["D"]

                    dzdt[idx] = dTdt

                elif node == (self.number_nodes - 1):
                    Ti = nodes_temp[node]
                    Ti_a = nodes_temp[node - 1]

                    dTdt = c[node]["A"] * Ti + c[node]["B"] * Ti_a + c[node]["D"]

                    dzdt[idx] = dTdt

                else:
                    Ti = nodes_temp[node]
                    Ti_b = nodes_temp[node + 1]
                    Ti_a = nodes_temp[node - 1]

                    dTdt = (
                        c[node]["A"] * Ti
                        + c[node]["B"] * Ti_a
                        + c[node]["C"] * Ti_b
                        + c[node]["D"]
                    )

                    dzdt[idx] = dTdt

            return dzdt

        # node indexes
        top = 0
        bottom = self.number_nodes - 1

        mass_flow_tot = (
            self.mass_flow_calc(
                state,
                flow_temp,
                return_temp,
                source_temp,
                source_delta_t,
                thermal_output,
                demand,
                nodes_temp[bottom],
                nodes_temp[top],
            )
            * 3600
        )
        # number of internal timesteps function of node mass and charging/dis mass
        t_ = math.ceil((mass_flow_tot / self.calc_node_mass()))
        # maximum internal timesteps is the number of nodes
        t = min(self.number_nodes, t_)
        # minimum internal timesteps is 1
        t = max(t, 1)

        # initial condition of coefficients
        coefficients = []
        # divide thermal output and demand accross timesteps
        # convert from kWh to kJ
        thermal_output = thermal_output * 3600 / float(t)
        demand = demand * 3600 / float(t)

        node_temp_list = list(range(1, t + 1))

        # solve ODE
        for i in range(1, t + 1):
            # span for next time step
            tspan = [i - 1, i]
            # solve for next step
            # new coefficients
            coefficients = self.set_of_coefficients(
                state,
                nodes_temp,
                source_temp,
                source_delta_t,
                flow_temp,
                return_temp,
                thermal_output,
                demand,
                nodes_temp[bottom],
                nodes_temp[top],
                timestep,
            )

            z = odeint(model_temp, nodes_temp, tspan, args=(coefficients,))

            nodes_temp = z[1]
            nodes_temp = sorted(nodes_temp, reverse=True)
            node_temp_list[i - 1] = nodes_temp

        return node_temp_list

    def max_energy_in_out(
        self, state, nodes_temp, source_temp, flow_temp, return_temp, timestep
    ):
        nodes_temp_sum = 0.0
        for node in range(len(nodes_temp)):
            nodes_temp_sum += nodes_temp[node]
        if nodes_temp_sum >= source_temp * len(nodes_temp) and state == "charging":
            return 0.0

        if nodes_temp_sum <= return_temp * len(nodes_temp) and state == "discharging":
            return 0.0

        if self.capacity == 0:
            return 0.0

        def model_temp(z, t, c):
            dzdt = list(range(self.number_nodes))
            for idx, node in enumerate(range(self.number_nodes)):

                if node == 0:
                    Ti = nodes_temp[node]
                    Ti_b = nodes_temp[node + 1]

                    dTdt = c[node]["A"] * Ti + c[node]["C"] * Ti_b + c[node]["D"]

                    dzdt[idx] = dTdt

                elif node == (self.number_nodes - 1):
                    Ti = nodes_temp[node]
                    Ti_a = nodes_temp[node - 1]

                    dTdt = c[node]["A"] * Ti + c[node]["B"] * Ti_a + c[node]["D"]

                    dzdt[idx] = dTdt

                else:
                    Ti = nodes_temp[node]
                    Ti_b = nodes_temp[node + 1]
                    Ti_a = nodes_temp[node - 1]

                    dTdt = (
                        c[node]["A"] * Ti
                        + c[node]["B"] * Ti_a
                        + c[node]["C"] * Ti_b
                        + c[node]["D"]
                    )

                    dzdt[idx] = dTdt

            return dzdt

        # number of time points
        t = self.number_nodes - 1

        # initial condition of coefficients
        coefficients = []
        coefficients.append(
            (
                self.set_of_max_coefficients(
                    state, nodes_temp, source_temp, flow_temp, return_temp, timestep
                )
            )
        )

        energy_list = []
        mass_flow = self.calc_node_mass()
        cp = self.specific_heat_water(source_temp)

        # solve ODE
        for i in range(1, t + 1):
            if state == "charging" and source_temp > nodes_temp[self.number_nodes - 1]:
                energy = (
                    mass_flow * cp * (source_temp - nodes_temp[self.number_nodes - 1])
                )
            elif state == "discharging" and nodes_temp[0] > flow_temp:
                energy = mass_flow * cp * (nodes_temp[0] - return_temp)
            else:
                energy = 0
            energy_list.append(energy)
            # span for next time step
            tspan = [i - 1, i]
            # solve for next step
            # new coefficients
            coefficients.append(
                (
                    self.set_of_max_coefficients(
                        state, nodes_temp, source_temp, flow_temp, return_temp, timestep
                    )
                )
            )

            z = odeint(model_temp, nodes_temp, tspan, args=(coefficients[i],))
            nodes_temp = z[1]

        # convert J to kWh by divide by 3600000
        energy_total = sum(energy_list) / 3600000
        return energy_total
