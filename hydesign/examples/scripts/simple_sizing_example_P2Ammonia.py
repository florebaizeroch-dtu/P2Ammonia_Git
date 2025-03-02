if __name__ == '__main__':
    import sys
    # Add the new path to sys.path
    sys.path.insert(0, r"C:\Users\Flore\Documents\GitHub\P2Ammonia_Git")
    # Import the package
    import hydesign
    import inspect
    print(inspect.getfile(hydesign))
    from hydesign.assembly.hpp_assembly_P2Ammonia import hpp_model_P2Ammonia as hpp_model
    from hydesign.Parallel_EGO import EfficientGlobalOptimizationDriver
    from hydesign.examples import examples_filepath
    import pandas as pd
    
    example = 9
    examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0, sep=';')
    ex_site = examples_sites.iloc[example]
    
    H2_demand_fn = examples_filepath + "Europe/H2_demand.csv"
    NH3_demand_fn = examples_filepath + "Europe/NH3_demand.csv"
    H2_demand_data = pd.read_csv(H2_demand_fn, header=0, index_col=0, parse_dates=True)
    H2_demand_data['H2_demand'] = 100_000
    H2_demand_data.to_csv(H2_demand_fn)

    NH3_demand_data = pd.read_csv(NH3_demand_fn, header=0, index_col=0, parse_dates=True)
    NH3_demand_data['NH3_demand'] = 4_500
    NH3_demand_data.to_csv(NH3_demand_fn)

    sim_pars_fn = examples_filepath + "Europe/hpp_pars.yml"
    input_ts_fn = input_ts_fn = examples_filepath + "Europe/GWA3/input_ts_Denmark_good_wind.csv"

    inputs = {
        'name': ex_site['name'],
        'longitude': ex_site['longitude'],
        'latitude': ex_site['latitude'],
        'altitude': ex_site['altitude'],
        'input_ts_fn': input_ts_fn,
        'sim_pars_fn': sim_pars_fn,
        'H2_demand_fn': H2_demand_fn,
        'NH3_demand_fn': NH3_demand_fn,
        'price_col': ex_site['price_col'],

        'opt_var': "NPV_over_CAPEX",
        'num_batteries': 10,
        'n_procs': 4,
        'n_doe': 48,
        'n_clusters': 4,
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
            #{'var_type':'design',
            # 'limits':[10, 400],
            # 'types':'int'
            #  },
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
            #{'var_type':'design',
            #  'limits':[0, 100],
            #  'types':'int'
            #  },
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
            {'var_type':'design',
              'limits':[0, 50],
              'types':'int'
              },
            # {'var_type':'fixed',
            #  'value': 50
            # },
        'b_E_h [h]':
            {'var_type':'design',
              'limits':[0, 10],
              'types':'int'
              },
            #{'var_type':'fixed',
            #  'value': 3
            #  },
        'cost_of_battery_P_fluct_in_peak_price_ratio':
            # {'var_type':'design',
            #   'limits':[0, 20],
            #   'types':'float'
            #   },
            {'var_type':'fixed',
              'value': 0},
        'ptg_MW [MW]':
            {'var_type':'design',
              'limits':[0, 100],
              'types':'int'
              },
            #{'var_type':'fixed',
            #  'value': 45
            #},
        'HSS_kg [kg]':
            {'var_type':'design',
              'limits':[0, 10000],
              'types':'int'
             },
            #{'var_type':'fixed',
            #  'value': 0
            #},
        'NH3SS_kg [kg]':
            {'var_type':'design',
              'limits':[0, 5000],
              'types':'int'
              },
            #{'var_type':'fixed',
            # 'value': 0
            # },
        }}
    EGOD = EfficientGlobalOptimizationDriver(**inputs)
    EGOD.run()
    result = EGOD.result
