from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd

from .models import (
    HP,
    ModelName,
    Lorentz,
    StandardTestRegression,
    GenericRegression,
    PerformanceArray,
    PerformanceModel,
    PerformanceValue,
    Simple,
)
from ..constants import ANNUAL_HOURS
from ..environment import weather

LOG = logging.getLogger(__name__)


@dataclass
class HpDemand:
    demand: float
    elec: float


class HeatPump(object):
    def __init__(
        self,
        hp_type: HP,
        modelling_approach: ModelName,
        capacity: float,
        ambient_delta_t: float,
        minimum_runtime: float,
        minimum_output: float,
        data_input: str,
        flow_temp_source: pd.DataFrame,
        return_temp: pd.DataFrame,
        hp_ambient_temp: pd.DataFrame,
        simple_cop: float = None,
        lorentz_inputs: dict = None,
        standard_inputs: dict = None,
    ):
        """heat pump class object

        Args:
            hp_type, type of heatpump, ASHP, WSHP, GSHP
            modelling_approach, selects performance model
            capacity, thermal capacity of heat pump
            ambient_delta_t, drop in ambient source temperature
                from inlet to outlet
            minimum_runtime, fixed or variable speed compressor
            data_input, type of data input, peak or integrated
            flow_temp, required temperatures out of HP
            return_temp, inlet temp to HP
            hp_ambient_temp, ambient conditions of heat source

        Kwargs: all these are for inputs, bar simple, for different modelling approaches
            simple_cop, only COP for simple, default: None
            lorentz_inputs, default: None
            standard_inputs, default: None
        """
        self.hp_type = hp_type
        self.capacity = capacity
        self.ambient_delta_t = ambient_delta_t
        self.minimum_runtime = minimum_runtime
        self.minimum_output = minimum_output
        self.data_input = data_input
        self.flow_temp_source = flow_temp_source
        self.return_temp = return_temp
        self.hp_ambient_temp = hp_ambient_temp

        self.simple_cop = simple_cop
        self.lorentz_inputs = lorentz_inputs
        self.standard_inputs = standard_inputs

        self._model = None
        self.model = modelling_approach

    @property
    def model(self) -> PerformanceModel:
        return self._model

    @model.setter
    def model(self, approach: ModelName):
        """Generate PerformanceModel from modelling approach"""
        match approach:
            case ModelName.SIMPLE:
                if self.simple_cop is None:
                    msg = f"Heat pump performance model ({Simple}) cannot be initiated with simple_cop=None"
                    LOG.error(msg)
                    raise ValueError(msg)
                self._model = Simple(self.simple_cop, self.capacity)

            case ModelName.LORENTZ:
                if self.lorentz_inputs is None:
                    msg = f"Heat pump performance model ({Lorentz}) cannot be initiated with lorentz_inputs=None"
                    LOG.error(msg)
                    raise ValueError(msg)
                self._model = Lorentz(
                    self.lorentz_inputs["cop"],
                    self.lorentz_inputs["flow_temp_spec"],
                    self.lorentz_inputs["return_temp_spec"],
                    self.lorentz_inputs["temp_ambient_in_spec"],
                    self.lorentz_inputs["temp_ambient_out_spec"],
                    self.lorentz_inputs["elec_capacity"],
                )

            case ModelName.GENERIC:
                # Raises error if hp_type is invalid
                self._model = GenericRegression(self.hp_type)

            case ModelName.STANDARD:
                if self.standard_inputs is None:
                    msg = f"Heat pump performance model ({StandardTestRegression}) cannot be initiated with standard_inputs=None"
                    LOG.error(msg)
                    raise ValueError(msg)
                self._model = StandardTestRegression(
                    self.standard_inputs["data_x"],
                    self.standard_inputs["data_COSP"],
                    self.standard_inputs["data_duty"],
                )

            case _:
                msg = f"Performance model {approach} is not valid"
                LOG.error(msg)
                raise KeyError(msg)

    def heat_resource(self):
        """accessing the heat resource

        takes the hp resource from the weather class

        Returns:
            dataframe -- ambient temperature for heat source of heat pump
        """

        HP_resource = weather.Weather(
            air_temperature=self.hp_ambient_temp["air_temperature"],
            water_temperature=self.hp_ambient_temp["water_temperature"],
        ).heatpump()

        if self.hp_type == HP.ASHP:

            HP_resource = HP_resource.rename(
                columns={"air_temperature": "ambient_temp"}
            )
            return HP_resource[["ambient_temp"]]

        elif self.hp_type == HP.WSHP:

            HP_resource = HP_resource.rename(
                columns={"water_temperature": "ambient_temp"}
            )
            return HP_resource[["ambient_temp"]]

        else:
            msg = f"Invalid heat pump type: {self.hp_type}, must be {HP.ASHP} or {HP.WSHP}"
            LOG.error(msg)
            raise ValueError(msg)

    def performance(self) -> PerformanceArray:
        """performance over year of heat pump

        Returns:
            Performance object defining cop and duty for each hour timestep in year
        """
        if self.capacity == 0:
            # TODO: change mpc to be clearer about how it uses cop and duty
            # cop needs to be low to not break the mpc solver
            # duty being zero means it won't choose it anyway
            cop = np.empty((ANNUAL_HOURS,)).fill(0.5)
            duty = np.zeros((ANNUAL_HOURS,))
            return PerformanceArray(cop, duty)

        ambient_temp = self.heat_resource()["ambient_temp"]

        duty_x = self.capacity

        match self.model:
            case Simple():
                return PerformanceArray(
                    np.empty((ANNUAL_HOURS,)).fill(self.model.cop),
                    np.empty((ANNUAL_HOURS,)).fill(self.model.duty),
                )

            case Lorentz():
                return PerformanceArray(
                    self.model.cop(
                        self.flow_temp_source.values,
                        self.return_temp.values,
                        ambient_temp.values,
                        ambient_temp.values - self.ambient_delta_t,
                    ),
                    np.empty((ANNUAL_HOURS,)).fill(self.model.duty(self.capacity)),
                )

            case GenericRegression():
                return PerformanceArray(
                    self.model.cop(self.flow_temp_source.values, ambient_temp.values),
                    np.empty((ANNUAL_HOURS,)).fill(duty_x),
                )

            case StandardTestRegression():
                cop = self.model.cop(ambient_temp.values, self.flow_temp_source.values)

                # 15% reduction in performance if
                # data not done to standards
                # TODO: check this logic against original code
                factor = np.ones(ambient_temp.shape)
                if self.data_input == "Integrated performance":
                    factor = 1.0
                elif self.data_input == "Peak performance":
                    if self.hp_type == HP.ASHP:
                        factor[ambient_temp <= 5.0] = 0.9
                    else:
                        msg = (
                            f"Peak performance option not available for {self.hp_type}"
                        )
                        LOG.error(msg)
                        raise ValueError(msg)
                else:
                    msg = f"{self.data_input} is not valid for {self.model.__class__}"
                    LOG.error(msg)
                    raise ValueError(msg)

                return PerformanceArray(
                    cop * factor, self.model.duty(ambient_temp, self.flow_temp_source)
                )

            case _:
                msg = f"Performance of {self.model} cannot be calculated"
                LOG.error(msg)
                raise ValueError(msg)

    def elec_usage(self, demand: float, hp_performance: PerformanceValue) -> float:
        """electricity usage of hp for timestep given a thermal demand

        calculates the electrical usage of the heat pump given a heat demand
        outputs a dataframe of heat demand, heat pump heat demand,
        heat pump elec demand, cop, duty, and leftover
        (only non-zero for fixed speed HP)

        Args:
            demand, thermal demand to be met by heat pump
            hp_performance, PerformanceValue at specific timestep

        Returns:
            Electricity usage
        """
        if self.capacity == 0:
            return 0.0
        max_elec_usage = demand / hp_performance.cop
        max_elec_cap = hp_performance.duty / hp_performance.cop
        return min(max_elec_usage, max_elec_cap)

    def thermal_output(
        self, elec_supply: float, hp_performance: PerformanceValue, heat_demand: float
    ) -> HpDemand:
        """thermal output from a given electricity supply

        Args:
            elec_supply, electricity supply used by heat pump
            hp_performance, PerformanceValue at specific timestep
            heat_demand, heat demand to be met of timestep

        Returns:
            HpDemand object defining heat demand met by hp and electricity usage
        """

        if self.capacity == 0:
            return HpDemand(0.0, 0.0)

        # maximum thermal output given elec supply
        max_thermal_output = elec_supply * hp_performance.cop

        # demand met by hp is min of three arguments
        hp_demand = min(max_thermal_output, heat_demand, hp_performance.duty)
        hp_elec = hp_demand / hp_performance.cop

        return HpDemand(hp_demand, hp_elec)
