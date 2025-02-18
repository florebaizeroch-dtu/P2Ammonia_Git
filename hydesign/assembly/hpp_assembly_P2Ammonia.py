import os
import numpy as np
import pandas as pd
import openmdao.api as om
import yaml
import matplotlib.pyplot as plt

from hydesign.weather.weather import ABL
from hydesign.wind.wind import genericWT_surrogate, genericWake_surrogate, wpp, get_rotor_d # , wpp_with_degradation, get_rotor_area
from hydesign.pv.pv import pvp #, pvp_with_degradation
from hydesign.ems.ems_P2Ammonia import ems_P2Ammonia as ems #, ems_long_term_operation
from hydesign.costs.costs_P2Ammonia import wpp_cost, pvp_cost, battery_cost, shared_cost, ptg_cost, NH3_cost
from hydesign.finance.finance_P2Ammonia import finance_P2Ammonia
from hydesign.assembly.hpp_assembly import hpp_base


class hpp_model_P2Ammonia(hpp_base):
    """HPP design evaluator"""

    def __init__(
        self, 
        sim_pars_fn,
        H2_demand_fn = None,
        NH3_demand_fn = None,
        **kwargs
        ):
        """Initialization of the hybrid power plant evaluator

        Parameters
        ----------
        sims_pars_fn : Case study input values of the HPP
        """
        defaults = {'electrolyzer_eff_curve_name': 'PEM electrolyzer H2 production',
                    'electrolyzer_eff_curve_type': 'production', 
                    }
        
        hpp_base.__init__(self,
                          sim_pars_fn=sim_pars_fn,
                          defaults=defaults,
                          **kwargs
                          )
        
        N_time = self.N_time
        N_ws = self.N_ws
        wpp_efficiency = self.wpp_efficiency
        sim_pars = self.sim_pars
        # life_h = self.life_h
        # N_life = self.N_life
        price = self.price
        life_y = self.life_y
        genWT_fn = sim_pars['genWT_fn']
        genWake_fn = sim_pars['genWake_fn']
        ems_type = sim_pars['ems_type']
        electrolyzer_eff_fn = os.path.join(os.path.dirname(sim_pars_fn), 'Electrolyzer_efficiency_curves.csv')
        df = pd.read_csv(electrolyzer_eff_fn)
        electrolyzer_eff_curve_name = sim_pars['electrolyzer_eff_curve_name']
        electrolyzer_eff_curve_type = sim_pars['electrolyzer_eff_curve_type']
        if electrolyzer_eff_curve_type =='production':
            eff_curve_col_name = electrolyzer_eff_curve_name + ' H2 production'
        elif electrolyzer_eff_curve_type =='efficiency':
            eff_curve_col_name = electrolyzer_eff_curve_name + ' efficiency'
        col_no = df.columns.get_loc(eff_curve_col_name)
        my_df = df.iloc[:,col_no:col_no+2].dropna()
        eff_curve = my_df[1:].values.astype(float)
        eta_col_no = df.columns.get_loc(electrolyzer_eff_curve_name + ' efficiency')
        eta_curve = df.iloc[:,eta_col_no:eta_col_no+2].dropna()
        eta_H2_curve = eta_curve[1:].values.astype(float)
        self.eta_H2 = np.max(eta_H2_curve)
       
        
        input_ts_fn = sim_pars['input_ts_fn']
        altitude = sim_pars['altitude']
        longitude = sim_pars['longitude']
        latitude = sim_pars['latitude']
        weather = pd.read_csv(input_ts_fn, index_col=0, parse_dates=True)
        H2_demand_data = pd.read_csv(H2_demand_fn, index_col=0, parse_dates=True).loc[weather.index,:]
        NH3_demand_data = pd.read_csv(NH3_demand_fn, index_col=0, parse_dates=True).loc[weather.index,:]
     
        model = om.Group()
        
        model.add_subsystem(
            'abl', 
            ABL(
                weather_fn=input_ts_fn, 
                N_time=N_time),
            promotes_inputs=['hh']
            )
        model.add_subsystem(
            'genericWT', 
            genericWT_surrogate(
                genWT_fn=genWT_fn,
                N_ws = N_ws),
            promotes_inputs=[
               'hh',
               'd',
               'p_rated',
            ])
        
        model.add_subsystem(
            'genericWake', 
            genericWake_surrogate(
                genWake_fn=genWake_fn,
                N_ws = N_ws),
            promotes_inputs=[
                'Nwt',
                'Awpp',
                'd',
                'p_rated',
                ])
        
        model.add_subsystem(
            'wpp', 
            wpp(
                N_time = N_time,
                N_ws = N_ws,
                wpp_efficiency = wpp_efficiency,)
                )
        
        model.add_subsystem(
            'pvp', 
            pvp(
                weather_fn = input_ts_fn, 
                N_time = N_time,
                latitude = latitude,
                longitude = longitude,
                altitude = altitude,
                tracking = sim_pars['tracking']
               ),
            promotes_inputs=[
                'surface_tilt',
                'surface_azimuth',
                'DC_AC_ratio',
                'solar_MW',
                'land_use_per_solar_MW',
                ])
        model.add_subsystem(
            'ems', 
            ems(
                N_time = N_time,
                eff_curve=eff_curve,
                # life_h = life_h, 
                ems_type=ems_type,
                electrolyzer_eff_curve_type=electrolyzer_eff_curve_type
                ),
            promotes_inputs=[
                'price_t',
                'b_P',
                'b_E',
                'G_MW',
                'battery_depth_of_discharge',
                'battery_charge_efficiency',
                'peak_hr_quantile',
                'cost_of_battery_P_fluct_in_peak_price_ratio',
                'n_full_power_hours_expected_per_day_at_peak_price',
                'price_H2',
                'price_NH3',
                'water_treatment_cost',
                'water_cost',
                'water_consumption',
                'ptg_MW',
                'NH3_tpd',
                'HSS_kg',
                'NH3SS_kg',
                'H2_storage_eff',
                'NH3_storage_eff',
                'ptg_deg',
                'P_elec_NH3_kg',
                'P_elec_N2_kg',
                'hhv_H2',
                'lhv_H2',
                'hhv_NH3',
                'lhv_NH3',
                'm_H2_demand_t',
                'm_NH3_demand_t',
                'penalty_factor_H2',
                'penalty_factor_NH3',
                'min_power_standby',
                ],
            promotes_outputs=[
                'total_curtailment'
            ])

        
        model.add_subsystem(
            'wpp_cost',
            wpp_cost(
                wind_turbine_cost=sim_pars['wind_turbine_cost'],
                wind_civil_works_cost=sim_pars['wind_civil_works_cost'],
                wind_fixed_onm_cost=sim_pars['wind_fixed_onm_cost'],
                wind_variable_onm_cost=sim_pars['wind_variable_onm_cost'],
                d_ref=sim_pars['d_ref'],
                hh_ref=sim_pars['hh_ref'],
                p_rated_ref=sim_pars['p_rated_ref'],
                N_time = N_time, 
            ),
            promotes_inputs=[
                'Nwt',
                'Awpp',
                'hh',
                'd',
                'p_rated'])
        model.add_subsystem(
            'pvp_cost',
            pvp_cost(
                solar_PV_cost=sim_pars['solar_PV_cost'],
                solar_hardware_installation_cost=sim_pars['solar_hardware_installation_cost'],
                solar_inverter_cost=sim_pars['solar_inverter_cost'],
                solar_fixed_onm_cost=sim_pars['solar_fixed_onm_cost'],
            ),
            promotes_inputs=['solar_MW', 'DC_AC_ratio'])

        model.add_subsystem(
            'battery_cost',
            battery_cost(
                battery_energy_cost=sim_pars['battery_energy_cost'],
                battery_power_cost=sim_pars['battery_power_cost'],
                battery_BOP_installation_commissioning_cost=sim_pars['battery_BOP_installation_commissioning_cost'],
                battery_control_system_cost=sim_pars['battery_control_system_cost'],
                battery_energy_onm_cost=sim_pars['battery_energy_onm_cost'],
                # N_life = N_life,
                # life_h = life_h
                life_y = life_y,
            ),
            promotes_inputs=[
                'b_P',
                'b_E',
                'battery_price_reduction_per_year'])

        model.add_subsystem(
            'shared_cost',
            shared_cost(
                hpp_BOS_soft_cost=sim_pars['hpp_BOS_soft_cost'],
                hpp_grid_connection_cost=sim_pars['hpp_grid_connection_cost'],
                land_cost=sim_pars['land_cost'],
            ),
            promotes_inputs=[
                'G_MW',
                'Awpp',
            ])
        model.add_subsystem(
            'ptg_cost',
            ptg_cost(
                ptg_type = sim_pars['electrolyzer_eff_curve_name'],
                AEC_capex_cost = sim_pars['AEC_capex_cost'],
                AEC_opex_cost = sim_pars['AEC_opex_cost'],
                HTP_AEC_capex_cost = sim_pars['HTP_AEC_capex_cost'],
                HTP_AEC_opex_cost = sim_pars['HTP_AEC_opex_cost'],
                LTP_AEC_capex_cost = sim_pars['LTP_AEC_capex_cost'],
                LTP_AEC_opex_cost = sim_pars['LTP_AEC_opex_cost'],
                PEM_capex_cost = sim_pars['PEM_capex_cost'],
                PEM_opex_cost = sim_pars['PEM_opex_cost'],
                LP_SOEC_capex_cost = sim_pars['LP_SOEC_capex_cost'],
                LP_SOEC_opex_cost = sim_pars['LP_SOEC_opex_cost'],
                HP_SOEC_capex_cost = sim_pars['HP_SOEC_capex_cost'],
                HP_SOEC_opex_cost = sim_pars['HP_SOEC_opex_cost'],
                water_cost = sim_pars['water_cost'],
                water_treatment_cost = sim_pars['water_treatment_cost'],
                water_consumption = sim_pars['water_consumption'],
                H2_storage_capex_cost = sim_pars['H2_storage_capex_cost'],
                H2_storage_opex_cost = sim_pars['H2_storage_opex_cost'],
                N_time = N_time,
                # life_h = life_h,
                ),
            promotes_inputs=[
            'ptg_MW',
            'HSS_kg',
            ])
        
        model.add_subsystem(
            'NH3_cost',
            NH3_cost(
                NH3_HB_capex_cost = sim_pars['NH3_HB_capex_cost'],
                NH3_HB_opex_cost = sim_pars['NH3_HB_opex_cost'],
                ASU_capex_cost = sim_pars['ASU_capex_cost'],
                NH3_storage_capex_cost = sim_pars['NH3_storage_capex_cost'],
                N_time = N_time,
                # life_h = life_h,
                ),
            promotes_inputs=[
            'NH3_tpd',
            'NH3SS_kg',
            ])
        
        model.add_subsystem(
            'finance_P2Ammonia', 
            finance_P2Ammonia(
                N_time = N_time, 
                # Depreciation curve
                depreciation_yr = sim_pars['depreciation_yr'],
                depreciation = sim_pars['depreciation'],
                # Inflation curve
                inflation_yr = sim_pars['inflation_yr'],
                inflation = sim_pars['inflation'],
                ref_yr_inflation = sim_pars['ref_yr_inflation'],
                # Early paying or CAPEX Phasing
                phasing_yr = sim_pars['phasing_yr'],
                phasing_CAPEX = sim_pars['phasing_CAPEX'],
                # life_h = life_h
                ),
            promotes_inputs=['price_H2',
                             'price_NH3',
                             'wind_WACC',
                             'solar_WACC', 
                             'battery_WACC',
                             'ptg_WACC',
                             'NH3_WACC',
                             'tax_rate',
                             'penalty_factor_H2',
                             'penalty_factor_NH3'
                            ],
            promotes_outputs=['NPV',
                              'IRR',
                              'NPV_over_CAPEX',
                              'LCOE',
                              'LCOH',
                              'LCONH3',
                              'Revenue',
                              'mean_AEP',
                            #   'mean_Power2Grid',
                              'annual_H2',
                              'annual_NH3',
                              'annual_P_ptg',
                              'annual_Q',
                              'annual_curtailment',
                              # 'annual_P_ptg_H2',
                              'penalty_lifetime',
                              'CAPEX',
                              'OPEX',
                              'break_even_H2_price',
                              'break_even_PPA_price',
                              'break_even_NH3_price',
                              ],
        )
                  
                      
        model.connect('genericWT.ws', 'genericWake.ws')
        model.connect('genericWT.pc', 'genericWake.pc')
        model.connect('genericWT.ct', 'genericWake.ct')
        model.connect('genericWT.ws', 'wpp.ws')

        model.connect('genericWake.pcw', 'wpp.pcw')

        model.connect('abl.wst', 'wpp.wst')
        
        model.connect('wpp.wind_t', 'ems.wind_t')
        model.connect('pvp.solar_t', 'ems.solar_t')
        
        
        model.connect('wpp.wind_t', 'wpp_cost.wind_t')
        
        
        model.connect('pvp.Apvp', 'shared_cost.Apvp')
        
        model.connect('wpp_cost.CAPEX_w', 'finance_P2Ammonia.CAPEX_w')
        model.connect('wpp_cost.OPEX_w', 'finance_P2Ammonia.OPEX_w')

        model.connect('pvp_cost.CAPEX_s', 'finance_P2Ammonia.CAPEX_s')
        model.connect('pvp_cost.OPEX_s', 'finance_P2Ammonia.OPEX_s')

        model.connect('battery_cost.CAPEX_b', 'finance_P2Ammonia.CAPEX_b')
        model.connect('battery_cost.OPEX_b', 'finance_P2Ammonia.OPEX_b')

        model.connect('shared_cost.CAPEX_sh', 'finance_P2Ammonia.CAPEX_el')
        model.connect('shared_cost.OPEX_sh', 'finance_P2Ammonia.OPEX_el')

        model.connect('ptg_cost.CAPEX_ptg', 'finance_P2Ammonia.CAPEX_ptg')
        model.connect('ptg_cost.OPEX_ptg', 'finance_P2Ammonia.OPEX_ptg')
        model.connect('ptg_cost.water_consumption_cost', 'finance_P2Ammonia.water_consumption_cost')

        model.connect('NH3_cost.CAPEX_NH3', 'finance_P2Ammonia.CAPEX_NH3')
        model.connect('NH3_cost.OPEX_NH3', 'finance_P2Ammonia.OPEX_NH3')
    

        model.connect('ems.price_t_ext', 'finance_P2Ammonia.price_t_ext')
        model.connect('ems.hpp_t', 'finance_P2Ammonia.hpp_t')
        model.connect('ems.penalty_t', 'finance_P2Ammonia.penalty_t')
        model.connect('ems.hpp_curt_t', 'finance_P2Ammonia.hpp_curt_t')
        model.connect('ems.m_H2_t', 'finance_P2Ammonia.m_H2_t')
        model.connect('ems.m_H2_t', 'ptg_cost.m_H2_t' )
        model.connect('ems.P_ptg_t', 'finance_P2Ammonia.P_ptg_t')
        model.connect('ems.m_H2_demand_t_ext', 'finance_P2Ammonia.m_H2_demand_t_ext')
        model.connect('ems.m_H2_offtake_t', 'finance_P2Ammonia.m_H2_offtake_t')
        model.connect('ems.m_H2_demand_t_ext', 'ptg_cost.m_H2_demand_t_ext')
        model.connect('ems.m_H2_offtake_t', 'ptg_cost.m_H2_offtake_t')
        model.connect('ems.P_HB_t', 'finance_P2Ammonia.P_HB_t')
        model.connect('ems.P_ASU_t', 'finance_P2Ammonia.P_ASU_t')
        model.connect('ems.m_NH3_t', 'finance_P2Ammonia.m_NH3_t')
        model.connect('ems.m_NH3_offtake_t', 'finance_P2Ammonia.m_NH3_offtake_t')
        model.connect('ems.Q_t', 'finance_P2Ammonia.Q_t')

        prob = om.Problem(
            model,
            reports=None
        )

        prob.setup()        
        
        # Additional parameters
        prob.set_val('price_t', price)
        prob.set_val('m_H2_demand_t', H2_demand_data['H2_demand'])
        prob.set_val('m_NH3_demand_t', NH3_demand_data['NH3_demand'])
        prob.set_val('G_MW', sim_pars['G_MW'])
        #prob.set_val('pv_deg_per_year', sim_pars['pv_deg_per_year'])
        prob.set_val('battery_depth_of_discharge', sim_pars['battery_depth_of_discharge'])
        prob.set_val('battery_charge_efficiency', sim_pars['battery_charge_efficiency'])      
        prob.set_val('peak_hr_quantile',sim_pars['peak_hr_quantile'] )
        prob.set_val('n_full_power_hours_expected_per_day_at_peak_price',
                     sim_pars['n_full_power_hours_expected_per_day_at_peak_price'])        
        #prob.set_val('min_LoH', sim_pars['min_LoH'])
        prob.set_val('wind_WACC', sim_pars['wind_WACC'])
        prob.set_val('solar_WACC', sim_pars['solar_WACC'])
        prob.set_val('battery_WACC', sim_pars['battery_WACC'])
        prob.set_val('ptg_WACC', sim_pars['ptg_WACC'])
        prob.set_val('NH3_WACC', sim_pars['NH3_WACC'])
        prob.set_val('tax_rate', sim_pars['tax_rate'])
        prob.set_val('land_use_per_solar_MW', sim_pars['land_use_per_solar_MW'])
        prob.set_val('hhv_H2', sim_pars['hhv_H2'])
        prob.set_val('lhv_H2', sim_pars['lhv_H2'])
        prob.set_val('hhv_NH3', sim_pars['hhv_NH3'])
        prob.set_val('lhv_NH3', sim_pars['lhv_NH3'])
        prob.set_val('min_power_standby', sim_pars['min_power_standby'])
        prob.set_val('ptg_deg', sim_pars['ptg_deg'])
        prob.set_val('price_H2', sim_pars['price_H2'])
        prob.set_val('price_NH3', sim_pars['price_NH3'])
        prob.set_val('water_treatment_cost', sim_pars['water_treatment_cost'])
        prob.set_val('water_cost', sim_pars['water_cost'])
        prob.set_val('water_consumption', sim_pars['water_consumption'])
        prob.set_val('penalty_factor_H2', sim_pars['penalty_factor_H2'])
        prob.set_val('penalty_factor_NH3', sim_pars['penalty_factor_NH3'])
        prob.set_val('H2_storage_eff', sim_pars['H2_storage_eff'])
        prob.set_val('NH3_storage_eff', sim_pars['NH3_storage_eff'])
    
        self.prob = prob
        
        self.list_out_vars = [
            'NPV_over_CAPEX',
            'NPV [MEuro]',
            'IRR',
            'LCOE [Euro/MWh]',
            'LCOH [Euro/kg]',
            'LCONH3 [Euro/kg]',
            'Revenue [MEuro]',
            'CAPEX [MEuro]',
            'OPEX [MEuro]',
            'penalty lifetime [MEuro]',
            'AEP [GWh]',
            # 'annual_Power2Grid [GWh]',
            'GUF',
            'annual_H2 [tons]',
            'annual_NH3 [tons]',
            'annual_Q [MW]',
            'annual_P_ptg [GWh]',
            # 'annual_P_ptg_H2 [GWh]',
            'grid [MW]',
            'wind [MW]',
            'solar [MW]',
            'PtG [MW]',
            'HSS [kg]',
            'NH3 [TDP]',
            'NH3SS [kg]',
            'Battery Energy [MWh]',
            'Battery Power [MW]',
            'Annual curtailment [GWh]',
            'Awpp [km2]',
            'Apvp [km2]',
            'Rotor diam [m]',
            'Hub height [m]',
            'Number of batteries used in lifetime',
            'Break-even H2 price [Euro/kg]',
            'Break-even NH3 price [Euro/kg]',
            'Break-even PPA price [Euro/MWh]',
            'Capacity factor wind [-]'
            ]

        self.list_vars = [
            'clearance [m]', 
            'sp [W/m2]', 
            'p_rated [MW]', 
            'Nwt', 
            'wind_MW_per_km2 [MW/km2]', 
            'solar_MW [MW]', 
            'surface_tilt [deg]', 
            'surface_azimuth [deg]', 
            'DC_AC_ratio', 
            'b_P [MW]', 
            'b_E_h [h]',
            'cost_of_battery_P_fluct_in_peak_price_ratio',
            'ptg_MW [MW]',
            'HSS_kg [kg]',
            'NH3SS [kg]',
            ]   
    
    
    def evaluate(
        self,
        # Wind plant design
        clearance, sp, p_rated, Nwt, wind_MW_per_km2,
        # PV plant design
        solar_MW,  surface_tilt, surface_azimuth, DC_AC_ratio,
        # Energy storage & EMS price constrains
        b_P, b_E_h, cost_of_battery_P_fluct_in_peak_price_ratio,
        # PtG plant design
        ptg_MW, 
        # Hydrogen storage capacity
        HSS_kg,
        #Ammonia storage 
        NH3SS_kg,
        ):
        """Calculating the financial metrics of the hybrid power plant project.

        Parameters
        ----------
        clearance : Distance from the ground to the tip of the blade [m]
        sp : Specific power of the turbine [MW/m2] 
        p_rated : Rated powe of the turbine [MW] 
        Nwt : Number of wind turbines
        wind_MW_per_km2 : Wind power installation density [MW/km2]
        solar_MW : Solar AC capacity [MW]
        surface_tilt : Surface tilt of the PV panels [deg]
        surface_azimuth : Surface azimuth of the PV panels [deg]
        DC_AC_ratio : DC  AC ratio
        b_P : Battery power [MW]
        b_E_h : Battery storage duration [h]
        cost_of_battery_P_fluct_in_peak_price_ratio : Cost of battery power fluctuations in peak price ratio [Eur]
        ptg_MW: Electrolyzer capacity [MW]
        HSS_kg: Hydrogen storgae capacity [kg]
        NH3SS_kg: Ammonia storage [kg]

        Returns
        -------
        prob['NPV_over_CAPEX'] : Net present value over the capital expenditures
        prob['NPV'] : Net present value
        prob['IRR'] : Internal rate of return
        prob['LCOE'] : Levelized cost of energy
        prob['LCOH'] : Levelized cost of hydrogen
        prob['LCONH3'] : Levelized cost of Ammonia
        prob['Revenue'] : Revenue of HPP
        prob['CAPEX'] : Total capital expenditure costs of the HPP
        prob['OPEX'] : Operational and maintenance costs of the HPP
        prob['penalty_lifetime'] : Lifetime penalty
        prob['AEP']: Annual energy production injected to the grid
        prob['mean_AEP']/(self.sim_pars['G_MW']*365*24) : Grid utilization factor
        prob['annual_H2']: Annual H2 production
        prob['annual_NH3']: Annual NH3 production
        prob['annual_Q']: Annual Heat production
        prob['annual_P_ptg']: Annual power converted to hydrogen
        self.sim_pars['G_MW'] : Grid connection [MW]
        wind_MW : Wind power plant installed capacity [MW]
        solar_MW : Solar power plant installed capacity [MW]
        ptg_MW: Electrolyzer capacity [MW]
        HSS_kg: Hydrogen storgae capacity [kg]
        NH3_tpd: Haber-Bosch capacity [tpd]
        NH3SS_kg: Ammonia storage [kg]
        b_E : Battery power [MW]
        b_P : Battery energy [MW]
        prob['total_curtailment']/1e3 : Total curtailed power [GMW]
        d : wind turbine diameter [m]
        hh : hub height of the wind turbine [m]
        num_batteries : Number of batteries
        """
        prob = self.prob
       
        d = get_rotor_d(p_rated*1e6/sp)
        hh = (d/2)+clearance
        wind_MW = Nwt * p_rated
        Awpp = wind_MW / wind_MW_per_km2 
        #Awpp = Awpp + 1e-10*(Awpp==0)
        b_E = b_E_h * b_P
        
        # pass design variables        
        prob.set_val('hh', hh)
        prob.set_val('d', d)
        prob.set_val('p_rated', p_rated)
        prob.set_val('Nwt', Nwt)
        prob.set_val('Awpp', Awpp)
        #Apvp = solar_MW * self.sim_pars['land_use_per_solar_MW']
        #prob.set_val('Apvp', Apvp)

        prob.set_val('surface_tilt', surface_tilt)
        prob.set_val('surface_azimuth', surface_azimuth)
        prob.set_val('DC_AC_ratio', DC_AC_ratio)
        prob.set_val('solar_MW', solar_MW)
        prob.set_val('ptg_MW', ptg_MW)
        prob.set_val('HSS_kg', HSS_kg)
        NH3_tpd = 4.08 * ptg_MW * self.eta_H2    
        prob.set_val('NH3_tpd', NH3_tpd)
        prob.set_val('NH3SS_kg', NH3SS_kg)        
        prob.set_val('b_P', b_P)
        prob.set_val('b_E', b_E)
        prob.set_val('cost_of_battery_P_fluct_in_peak_price_ratio',cost_of_battery_P_fluct_in_peak_price_ratio)        
        
        prob.run_model()
        
        self.prob = prob
         
        if Nwt == 0:
            cf_wind = np.nan
        else:
            cf_wind = prob.get_val('wpp.wind_t').mean() / p_rated / Nwt  # Capacity factor of wind only

        return np.hstack([
            prob['NPV_over_CAPEX'], 
            prob['NPV']/1e6,
            prob['IRR'],
            prob['LCOE'],
            prob['LCOH'],
            prob['LCONH3'],
            prob['Revenue']/1e6,
            prob['CAPEX']/1e6,
            prob['OPEX']/1e6,
            prob['penalty_lifetime']/1e6,
            prob['mean_AEP']/1e3, #[GWh]
            # prob['mean_Power2Grid']/1e3, #GWh
            # Grid Utilization factor
            prob['mean_AEP']/(self.sim_pars['G_MW']*365*24),
            prob['annual_H2']/1e3, # in tons
            prob['annual_NH3']/1e3, # in tons
            prob['annual_Q'],
            prob['annual_P_ptg']/1e3, # in GWh
            # prob['annual_P_ptg_H2']/1e3, # in GWh
            self.sim_pars['G_MW'],
            wind_MW,
            solar_MW,
            ptg_MW,
            HSS_kg,
            NH3_tpd,
            NH3SS_kg,
            b_E,
            b_P,
            prob['annual_curtailment']/1e3, #[GWh]
            Awpp,
            prob.get_val('shared_cost.Apvp'),
            d,
            hh,
            1 * (b_P>0),
            prob['break_even_H2_price'],
            prob['break_even_NH3_price'],
            prob['break_even_PPA_price'],
            cf_wind,
            ])
    
    def plot_P2Ammonia_results(self, n_hours=1 * 24, index_hour_start=0):
        # get data from solved hpp model
        prob = self.prob
        #price
        price_el = prob.get_val('ems.price_t')
        price_H2 = prob.get_val('ems.price_H2')
        price_NH3 = prob.get_val('ems.price_NH3')
        #electricity production 
        wind_t = prob.get_val('ems.wind_t')
        solar_t = prob.get_val('ems.solar_t')
        hpp_t = prob.get_val('ems.hpp_t')
        #Energy consumption
        p_ptg_t = prob.get_val('ems.P_ptg_t')
        p_hb_t = prob.get_val('ems.P_HB_t')
        p_asu_t = prob.get_val('ems.P_ASU_t')
        p_NH3_t = p_hb_t + p_asu_t
        #battery
        b_t = prob.get_val('ems.b_t')
        b_E_SOC_t = prob.get_val('ems.b_E_SOC_t')
        #H2 and ammonia production
        m_H2_t = prob.get_val('ems.m_H2_t')
        m_H2_offtake_t = prob.get_val('ems.m_H2_offtake_t')
        m_H2_to_NH3_t = prob.get_val('ems.m_H2_to_NH3_t')
        m_NH3_t = prob.get_val('ems.m_NH3_t')
        m_NH3_offtake_t = prob.get_val('ems.m_NH3_offtake_t')
        m_H2_demand_t = prob.get_val('ems.m_H2_demand_t_ext')
        m_NH3_demand_t = prob.get_val('ems.m_NH3_demand_t_ext')
        #storage ammonia
        LoS_NH3_t = prob.get_val('ems.LoS_NH3_t')
        #Heat production
        Q_t = prob.get_val('ems.Q_t')
        #Curtailment 
        p_curtail_t = prob.get_val('ems.hpp_curt_t')
        # Create a figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # ------------------------------
        # Plot 1: Prices for resources
        # ------------------------------
        axs[0][0].plot(range(len(price_el[:n_hours])), price_el[:n_hours], color='green', label='Electricity [€/MWh]')
        axs[0][0].plot(range(len(price_NH3[:n_hours])), price_NH3[:n_hours],  color='orange', linestyle=':', label='NH3 [€/kg]')
        axs[0][0].plot(range(len(price_H2[:n_hours])), price_H2[:n_hours],  color='purple', linestyle='--', label='H2 [€/kg]')

        axs[0][0].set_ylabel('Price [€]')
        axs[0][0].legend(loc='upper right')
        axs[0][0].set_xticklabels([])

        # -------------------------------------------
        # Plot 2: Electrical production & Consumption
        # -------------------------------------------
        axs[1][0].plot(range(len(wind_t[:n_hours])), wind_t[:n_hours], color='blue', label='Wind production')
        axs[1][0].plot(range(len(solar_t[:n_hours])), solar_t[:n_hours],  color='yellow', label='Solar production')
        axs[1][0].plot(range(len(p_ptg_t[:n_hours])), p_ptg_t[:n_hours], color='purple', label='H2 power consumption')
        axs[1][0].plot(range(len(p_NH3_t[:n_hours])), p_NH3_t[:n_hours],  color='green', label='NH3 power consumption')
        axs[1][0].plot(range(len(hpp_t[:n_hours])), hpp_t[:n_hours], color='red', label='Net power production')
        axs[1][0].plot(range(len(p_curtail_t[:n_hours])), p_curtail_t[:n_hours], color = 'deeppink', label = 'Curtailment')
        axs[1][0].plot(range(len(Q_t[:n_hours])), Q_t[:n_hours], color = 'orange', label = 'Heat production')
        axs[1][0].set_ylabel('Power [MWh]')
        axs[1][0].set_xlabel('Time [h]')
        axs[1][0].legend(loc='upper right')

        # ------------------------------
        # Plot 3: H2 and NH3 generation
        # ------------------------------
        axs[0][1].plot(range(len(m_NH3_t[:n_hours])), m_NH3_t[:n_hours],  color='green', label='NH3 production')
        axs[0][1].plot(range(len(m_NH3_demand_t[:n_hours])), m_NH3_demand_t[:n_hours], color='green', linestyle = ':', label='NH3 demand')
        axs[0][1].plot(range(len(m_NH3_offtake_t[:n_hours])), m_NH3_offtake_t[:n_hours], color='green', linestyle='--', label='NH3 offtake')
        axs[0][1].plot(range(len(m_H2_t[:n_hours])), m_H2_t[:n_hours],color='purple', label='H2 production')
        axs[0][1].plot(range(len(m_H2_demand_t[:n_hours])), m_H2_demand_t[:n_hours],color='purple', linestyle = ':', label='H2 demand')
        axs[0][1].plot(range(len(m_H2_offtake_t[:n_hours])), m_H2_offtake_t[:n_hours], color='purple', linestyle='--', label='H2 offtake')
        axs[0][1].plot(range(len(m_H2_to_NH3_t[:n_hours])), m_H2_offtake_t[:n_hours],  color='purple', linestyle='-.', label='H2 offtake')
        axs[0][1].set_ylabel('H2 and NH3 production [kg]')
        axs[0][1].set_xlabel('Time [h]')
        axs[0][1].legend(loc='upper right')

        # ------------------------------
        # Plot 4: Evolution of storage
        # ------------------------------
        axs[1][1].plot(range(len(LoS_NH3_t[:n_hours])), LoS_NH3_t[:n_hours], color='green', linestyle='-', label='LoS NH3')
        axs[1][1].plot(range(len(b_E_SOC_t[:n_hours])), b_E_SOC_t[:n_hours], color='blue', linestyle='-', label='Battery LoS')
        axs[1][1].set_ylim(0, 100)
        axs[1][1].set_ylabel('State of Charge[%]')
        axs[1][1].set_xlabel('Time [h]')
        axs[1][1].legend(loc='upper right')

        return fig

    def save_to_csv_results(self):
        # get data from solved hpp model
        prob = self.prob
        #price
        price_el = prob.get_val('ems.price_t')
        price_H2 = prob.get_val('ems.price_H2')

        price_NH3 = prob.get_val('ems.price_NH3')
        #electricity production 
        wind_t = prob.get_val('ems.wind_t')
        solar_t = prob.get_val('ems.solar_t')
        hpp_t = prob.get_val('ems.hpp_t')
        #Energy consumption
        p_ptg_t = prob.get_val('ems.P_ptg_t')
        p_hb_t = prob.get_val('ems.P_HB_t')
        p_asu_t = prob.get_val('ems.P_ASU_t')
        p_NH3_t = p_hb_t + p_asu_t
        #battery
        b_t = prob.get_val('ems.b_t')
        b_E_SOC_t = prob.get_val('ems.b_E_SOC_t')
        #H2 and ammonia production
        m_H2_t = prob.get_val('ems.m_H2_t')
        m_H2_offtake_t = prob.get_val('ems.m_H2_offtake_t')
        m_H2_to_NH3_t = prob.get_val('ems.m_H2_to_NH3_t')
        m_NH3_t = prob.get_val('ems.m_NH3_t')
        m_NH3_offtake_t = prob.get_val('ems.m_NH3_offtake_t')
        m_H2_demand_t = prob.get_val('ems.m_H2_demand_t_ext')
        m_NH3_demand_t = prob.get_val('ems.m_NH3_demand_t_ext')
        #storage ammonia
        LoS_NH3_t = prob.get_val('ems.LoS_NH3_t')
        #Heat production
        Q_t = prob.get_val('ems.Q_t')
        #Curtailment 
        p_curtail_t = prob.get_val('ems.hpp_curt_t')
        
        df = pd.DataFrame()
        df['Price_el'] = price_el
        df['Price_H2'] = pd.Series(price_H2[0])
        df['Price_NH3'] = pd.Series(price_NH3[0])
        df['wind_t'] = wind_t
        df['solar_t'] = solar_t
        df['hpp_t'] = hpp_t[:8760]
        df['p_ptg_t'] = p_ptg_t[:8760]
        df['p_asu_t'] = p_asu_t[:8760]
        df['p_NH3_t'] = p_NH3_t[:8760]
        df['b_t'] = b_t[:8760]
        df['b_E_SOC_t'] = b_E_SOC_t[:8760]
        df['m_H2_t'] = m_H2_t[:8760]
        df['m_H2_offtake_t']=m_H2_offtake_t[:8760]
        df['m_H2_to_NH3_t'] = m_H2_to_NH3_t[:8760]
        df['m_H2_demand_t'] = m_H2_demand_t[:8760]
        df['m_NH3_t'] = m_NH3_t[:8760]
        df['m_NH3_offtake_t'] = m_NH3_offtake_t[:8760]
        df['m_NH3_demand_t'] = m_NH3_demand_t[:8760]
        df['LoS_NH3_t'] = LoS_NH3_t[:8760]
        df['Q_t'] = Q_t[:8760]
        df['p_curtail_t'] = p_curtail_t[:8760]
        df.to_excel('Results_excel.xlsx')
