import matplotlib.pyplot as plt

def plot_P2Ammonia_results(hpp, n_hours=1*24):
    # get data from solved hpp model
    #price
    price_el = hpp.prob.get_val('ems_P2Ammonia.price_t_text')
    price_H2 = hpp.prob.get_val('ems_P2Ammonia.price_H2')
    price_NH3 = hpp.prob.get_val('ems_P2Ammonia.price_NH3')
    #electricity production 
    wind_t = hpp.prob.get_val('ems_P2Ammonia.wind_t_ext')
    solar_t = hpp.prob.get_val('ems_P2Ammonia.solar_t_ext')
    hpp_t = hpp.prob.get_val('ems_P2Ammonia.hpp_t')
    #Energy consumption
    p_ptg_t = hpp.prob.get_val('ems_P2Ammonia.P_ptg_t')
    p_hb_t = hpp.prob.get_val('ems_P2Ammonia.P_HB_t')
    p_asu_t = hpp.prob.get_val('ems_P2Ammonia.P_ASU_t')
    p_NH3_t = p_hb_t + p_asu_t
    #battery
    b_t = hpp.prob.get_val('ems_P2Ammonia.b_t')
    b_E_SOC_t = hpp.prob.get_val('ems_P2Ammonia.b_E_SOC_t')
    #H2 and ammonia production
    m_H2_t = hpp.prob.get_val('ems_P2Ammonia.m_H2_t')
    m_H2_offtake_t = hpp.prob.get_val('ems_P2Ammonia.m_H2_offtake_t')
    m_H2_to_NH3_t = hpp.prob.get_val('ems_P2Ammonia.m_H2_to_NH3_t')
    m_NH3_t = hpp.prob.get_val('ems_P2Ammonia.m_NH3_t')
    m_NH3_offtake_t = hpp.prob.get_val('ems_P2Ammonia.m_NH3_offtake_t')
    m_H2_demand_t = hpp.prob.get_val('ems_P2Ammonia.m_H2_demand_t_ext')
    m_NH3_demand_t = hpp.prob.get_val('ems_P2Ammonia.m_NH3_demand_t_ext')
    #storage ammonia
    LoS_NH3_t = hpp.prob.get_val('ems_P2Ammonia.LoS_NH3_t')
    #Heat production
    Q_t = hpp.prob.get_val('ems_P2Ammonia.Q_t')
    #Curtailment 
    P_curtail = hpp.prob.get_val('ems_P2Ammonia.total_curtailment')
    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # ------------------------------
    # Plot 1: Prices for resources
    # ------------------------------
    axs[0][0].plot(range(len(price_el[:n_hours])), price_el[:n_hours], where='mid', color='green', label='Electricity [€/MWh]')
    axs[0][0].plot(range(len(price_NH3[:n_hours])), price_NH3[:n_hours], where='mid', color='orange', linestyle=':', label='NH3 [€/kg]')
    axs[0][0].plot(range(len(price_H2[:n_hours])), price_H2[:n_hours], where='mid', color='purple', linestyle='--', label='H2 [€/kg]')

    axs[0][0].set_ylabel('Price [€]')
    axs[0][0].legend(loc='upper right')
    axs[0][0].set_xticklabels([])

    # -------------------------------------------
    # Plot 2: Electrical production & Consumption
    # -------------------------------------------
    axs[1][0].plot(range(len(wind_t[:n_hours])), wind_t[:n_hours], where='mid', color='blue', label='Wind production')
    axs[1][0].plot(range(len(solar_t[:n_hours])), solar_t[:n_hours], where='mid', color='yellow', label='Solar production')
    axs[1][0].plot(range(len(p_ptg_t[:n_hours])), p_ptg_t[:n_hours], where='mid', color='purple', label='H2 power consumption')
    axs[1][0].plot(range(len(p_NH3_t[:n_hours])), p_NH3_t[:n_hours], where='mid', color='green', label='NH3 power consumption')
    axs[1][0].plot(range(len(hpp_t[:n_hours])), hpp_t[:n_hours], where='mid', color='red', label='Net power production')
    axs[1][0].plot(range(len(P_curtail[:n_hours])), P_curtail[:n_hours], where ='mid', color = 'deeppink')
    axs[1][0].set_ylabel('Power [MWh]')
    axs[1][0].set_xlabel('Time [h]')
    axs[1][0].legend(loc='upper right')

    # ------------------------------
    # Plot 3: H2 and NH3 generation
    # ------------------------------
    axs[0][1].plot(range(len(m_NH3_t[:n_hours])), m_NH3_t[:n_hours], where='mid', color='green', label='NH3 production')
    axs[0][1].plot(range(len(m_NH3_demand_t[:n_hours])), m_NH3_demand_t[:n_hours], where='mid', color='green', linestyle = ':', label='NH3 demand')
    axs[0][1].plot(range(len(m_NH3_offtake_t[:n_hours])), m_NH3_offtake_t[:n_hours], where='mid', color='green', linestyle='--', label='NH3 offtake')
    axs[0][1].plot(range(len(m_H2_t[:n_hours])), m_H2_t[:n_hours], where='mid', color='purple', label='H2 production')
    axs[0][1].plot(range(len(m_H2_demand_t[:n_hours])), m_H2_demand_t[:n_hours], where='mid', color='purple', linestyle = ':', label='H2 demand')
    axs[0][1].plot(range(len(m_H2_offtake_t[:n_hours])), m_H2_offtake_t[:n_hours], where='mid', color='purple', linestyle='--', label='H2 offtake')
    axs[0][1].plot(range(len(m_H2_to_NH3_t[:n_hours])), m_H2_offtake_t[:n_hours], where='mid', color='purple', linestyle='-.', label='H2 offtake')
    axs[0][1].set_ylabel('H2 and NH3 production [kg]')
    axs[0][1].set_xlabel('Time [h]')
    axs[0][1].legend(loc='upper right')

    # ------------------------------
    # Plot 4: Evolution of storage
    # ------------------------------
    axs[1][1].plot(range(len(LoS_NH3_t[:n_hours])), LoS_NH3_t[:n_hours], where='mid', color='green', linestyle='-', label='LoS NH3')
    axs[1][1].plot(range(len(b_E_SOC_t[:n_hours])), b_E_SOC_t[:n_hours], where='mid', color='blue', linestyle='-', label='Battery LoS')
    axs[1][1].set_ylim(0, 100)
    axs[1][1].set_ylabel('State of Charge[%]')
    axs[1][1].set_xlabel('Time [h]')
    axs[1][1].legend(loc='upper right')

    # Show the plot
    plt.show()
