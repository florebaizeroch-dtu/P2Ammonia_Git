import os
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hydesign
from hydesign.Parallel_EGO import EfficientGlobalOptimizationDriver
from hydesign.assembly.hpp_assembly_P2Ammonia import hpp_model_P2Ammonia as hpp_model
from hydesign.Parallel_EGO import EfficientGlobalOptimizationDriver
from hydesign.examples import examples_filepath

# -----------------------------------------------------------
# Setup site-specific data for Hybrid Power Plant simulation
# -----------------------------------------------------------
latitude = 56.227322
longitude = 8.594398
altitude = 85

# Case study
input_ts_fn = examples_filepath + "Europe/GWA3/input_ts_Denmark_good_wind.csv"
#input_ts_fn = examples_filepath + "Europe/GWA3/input_ts_France_good_wind.csv"
#input_ts_fn = examples_filepath + "Europe/GWA3/input_ts_Germany_good_wind.csv"
H2_demand_fn = examples_filepath + "Europe/H2_demand.csv"
NH3_demand_fn = examples_filepath + "Europe/NH3_demand.csv"



H2_demand_data = pd.read_csv(H2_demand_fn, header=0, index_col=0, parse_dates=True)
H2_demand_data['H2_demand'] = 100_000
H2_demand_data.to_csv(H2_demand_fn)

NH3_demand_data = pd.read_csv(NH3_demand_fn, header=0, index_col=0, parse_dates=True)
NH3_demand_data['NH3_demand'] = 4_500
NH3_demand_data.to_csv(NH3_demand_fn)

# -----------------------------------------------------------
# Load simulation parameters and initialize the HPP model
# -----------------------------------------------------------

# Load simulation parameters from a YAML file
sim_pars_fn = examples_filepath + "Europe/hpp_pars.yml"
with open(sim_pars_fn) as file:
    sim_pars = yaml.load(file, Loader=yaml.FullLoader)
#print(sim_pars)

# Initialize the Hybrid Power Plant (HPP) model
batch_size = 15 * 24

hpp = hpp_model(
    sim_pars_fn=sim_pars_fn,  # Simulation parameters
    H2_demand_fn= H2_demand_fn, 
    NH3_demand_fn= NH3_demand_fn,
    latitude = latitude, 
    longitude = longitude, 
    altitude = altitude,  # Geographical data for the site
    input_ts_fn = input_ts_fn,
    batch_size=batch_size,
    work_dir='./',  # Directory for saving outputs
)
# -------------------------------------------------------
# Type of Analysis 
# -------------------------------------------------------
#type_analysis = 'sensitivity_analysis'
type_analysis = 'evaluation' 
#type_analysis = 'sizing'

# -------------------------------------------------------
# Operation parameters
# -------------------------------------------------------
 # sizing variables
    # Wind plant design
clearance = 10
sp = 360 
p_rated = 4
Nwt = 90
wind_MW_per_km2 = 5 
    #PV plant design 
solar_MW = 80
surface_tilt = 50
surface_azimuth= 210
DC_AC_ratio = 1.5
    #Energy storage & EMS price constraints
b_P = 0
b_E_h = 0
cost_of_batt_degr = 0
    #PtG plant design
ptg_MW = 35
    #Hydrogen storage design 
HSS_kg = 1000
    #Ammonia storage design 
NH3SS_kg = 0

x_design = [
    # sizing variables
    clearance, sp, p_rated, Nwt, wind_MW_per_km2,
    solar_MW, surface_tilt, surface_azimuth,
    DC_AC_ratio, b_P, b_E_h, cost_of_batt_degr,
    ptg_MW, HSS_kg, NH3SS_kg
]


# -----------------------------------------------------------
# Evaluation of HPP model 
# -----------------------------------------------------------

if type_analysis  == 'evaluation':
    #hpp.prob.set_val('price_H2', 4)
    #hpp.prob.set_val('price_NH3', 0.9)
    #hpp.prob.set_val('penalty_factor_NH3', 0.05)

    start = time.time()
    outs = hpp.evaluate(*x_design)  # Run the model evaluation
    hpp.print_design(x_design, outs)

    end = time.time()

    print(f'Execution time [min]:', round((end - start) / 60, 2))

    fig = hpp.plot_P2Ammonia_results(n_hours = batch_size)

    fig.savefig("C:/Users/Flore/Documents/GitHub/P2Ammonia_Flore/Results/fig_results.png", format = 'png', dpi = 300, bbox_inches = 'tight')
    plt.show()
    hpp.save_to_csv_results()

# -----------------------------------------------------------
# Sensitivity analysis
# -----------------------------------------------------------

elif type_analysis =='sensitivity_analysis':
    start = time.time()
    param_name = 'ptg_MW'
    param_list = np.linspace(0, 500, 10)
    dict = {'NPV' : [], 'NPV_over_CAPEX': [],  'IRR' : [], 'AH2P' : [], 'AEP' : [], 'ANH3P' : [], 'Curtail' : []}
    for param in param_list:
        print(f'{param_name} : {param} ')
        #hpp.prob.set_val(param_name, param)
        ptg_MW = param
        x_update =[clearance, sp, p_rated, Nwt, wind_MW_per_km2,
                    solar_MW, surface_tilt, surface_azimuth,
                    DC_AC_ratio, b_P, b_E_h, cost_of_batt_degr,
                    ptg_MW, HSS_kg, NH3SS_kg
                    ]
 
        outs = hpp.evaluate(*x_update)  # Run the model evaluation
        dict['NPV'].append(outs[1])
        dict['NPV_over_CAPEX'].append(outs[0])
        dict['IRR'].append(outs[2])
        dict['AH2P'].append(outs[12])
        dict['ANH3P'].append(outs[13])
        dict['AEP'].append(outs[10])
        dict['Curtail'].append(outs[25])
    end = time.time()

    print(f'Execution time [min]:', round((end - start) / 60, 2))
    df = pd.DataFrame.from_dict(dict)
    df.to_excel('Sensitivity_Analysis.xlsx')

# -----------------------------------------------------------    
# Sizing
# -----------------------------------------------------------
if type_analysis == 'sizing':
    inputs = {
        'name' : 'Denmark_good_wind',
        'longitude': longitude,
        'latitude': latitude,
        'altitude': altitude,
        'input_ts_fn': input_ts_fn,
        'sim_pars_fn': sim_pars_fn,
        'H2_demand_fn': H2_demand_fn,
        'NH3_demand_fn' : NH3_demand_fn,
        'opt_var': "NPV_over_CAPEX",
        'num_batteries': 1,
        'n_procs': 4,
        'n_doe': 32,
        'n_clusters': 1,
        'n_seed': 0,
        'max_iter': 5,
        'final_design_fn': 'hydesign_design_P2Ammonia.csv',
        'npred': 3e4,
        'tol': 1e-6,
        'min_conv_iter': 2,
        'work_dir': './',
        'hpp_model': hpp_model,
        'variables': {
        'clearance [m]':
            # {'var_type':'design',
            #   'limits':[10, 60],
            #   'types':'int'
            #   },
            {'var_type':'fixed',
              'value': 10
              },
         'sp [W/m2]':
            # {'var_type':'design',
            #  'limits':[200, 360],
            #  'types':'int'  
            #  },
            {'var_type':'fixed',
              'value': 360
              },
        'p_rated [MW]':
            # {'var_type':'design',
            #   'limits':[1, 10],
            #   'types':'int'
            #   },
            {'var_type':'fixed',
             'value': 4
             },
        'Nwt':
            # {'var_type':'design',
            #   'limits':[0, 400],
            #   'types':'int'
            #   },
            {'var_type':'fixed',
              'value': 90
              },
        'wind_MW_per_km2 [MW/km2]':
            # {'var_type':'design',
            #   'limits':[5, 9],
            #   'types':'float'
            #   },
            {'var_type':'fixed',
              'value': 5
              },
        'solar_MW [MW]':
            # {'var_type':'design',
            #   'limits':[0, 400],
            #   'types':'int'
            #   },
            {'var_type':'fixed',
              'value': 80
              },
        'surface_tilt [deg]':
            # {'var_type':'design',
            #   'limits':[0, 50],
            #   'types':'float'
            #   },
            {'var_type':'fixed',
              'value': 50
              },
        'surface_azimuth [deg]':
            # {'var_type':'design',
            #   'limits':[150, 210],
            #   'types':'float'
            #   },
            {'var_type':'fixed',
              'value': 210
              },
        'DC_AC_ratio':
            # {'var_type':'design',
            #   'limits':[1, 2.0],
            #   'types':'float'
            #   },
            {'var_type':'fixed',
              'value':1.5,
              },
        'b_P [MW]':
            # {'var_type':'design',
            #   'limits':[0, 100],
            #   'types':'int'
            #   },
            {'var_type':'fixed',
              'value': 50
              },
        'b_E_h [h]':
            # {'var_type':'design',
            #   'limits':[1, 10],5
            #   'types':'int'
            #   },
            {'var_type':'fixed',
              'value': 3
              },
        'cost_of_battery_P_fluct_in_peak_price_ratio':
            # {'var_type':'design',
            #   'limits':[0, 20],
            #   'types':'float'
            #   },
            {'var_type':'fixed',
              'value': 0},
        'ptg_MW [MW]':
            {'var_type':'design',
              'limits':[1, 200],
              'types':'int'
              },
            # {'var_type':'fixed',
            #   'value': 150
            # },
        'HSS_kg [kg]':
            #{'var_type':'design',
            #  'limits':[0, 5000],
            #  'types':'int'
            #  },
            {'var_type':'fixed',
              'value': 0
            },
        'NH3SS_kg [kg]':
            #{'var_type':'design',
            #  'limits':[0, 5000],
            #  'types':'int'
            #  },
            {'var_type':'fixed',
              'value': 0
            },
        }}
    EGOD = EfficientGlobalOptimizationDriver(**inputs)
    EGOD.run()
    result = EGOD.result
    print(result)

# -----------------------------------------------------------
# Test function for validating model outputs
# -----------------------------------------------------------

def test_outputs():
    """
    Test function to validate the outputs of the HPP model.
    Uses assertions to check if the calculated values match known values.
    """
    assert np.isclose(np.round(hpp.prob.get_val('NPV_over_CAPEX'), 3), 0.716, atol=1e-5), \
        f"Mismatch in NPV_over_CAPEX values: {np.round(hpp.prob.get_val('NPV_over_CAPEX'), 3)} != 0.716"

    assert np.isclose(np.round(hpp.prob.get_val('IRR'), 3), 0.115, atol=1e-5), \
        f"Mismatch in IRR values: {np.round(hpp.prob.get_val('IRR'), 3)} != 0.115"

    assert np.isclose(np.round(sum(hpp.prob.get_val('ems_long_term_operation.hpp_curt_t')) / 1000, 3), 27.949, atol=1e-5), \
        f"Mismatch in Total_curtailment_GWh values: {np.round(sum(hpp.prob.get_val('ems_long_term_operation.hpp_curt_t')) / 1000, 3)} != 27.949"

    assert np.isclose(np.round(sum(hpp.prob.get_val('ems_long_term_operation.hpp_curt_t_with_deg')) / 1000, 3), 4.919, atol=1e-5), \
        f"Mismatch in Total_curtailment_with_deg_GWh values: {np.round(hpp.prob.get_val('ems_long_term_operation.hpp_curt_t_with_deg'), 3)} != 4.919"

    print("All output values match the known max and min values successfully.")

# Run the test function to validate outputs
# test_outputs()
