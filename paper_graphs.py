import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hydesign.examples import examples_filepath

path_name = examples_filepath + "Europe/Electrolyzer_efficiency_curves.csv"

electrolyzer_data = pd.read_csv(path_name, header=0)

efficiency_curves = ["PEM electrolyzer efficiency", 
"HTP-AEC electrolyzer efficiency",
"HP-SOEC-WFS electrolyzer efficiency",
"LTP-AEC electrolyzer efficiency", 
"LP-SOEC-WFS electrolyzer efficiency"
 ]

H2_prod_curves = ["PEM electrolyzer H2 production", 
"HTP-AEC electrolyzer H2 production",
"HP-SOEC-WFS electrolyzer H2 production",
"LTP-AEC electrolyzer H2 production", 
"LP-SOEC-WFS electrolyzer H2 production"]

names = ["PEM electrolyzer", "HTP-AEC electrolyzer", "HP-SOEC-WFS electrolyzer", "LTP-AEC electrolyzer", "LP-SOEC-WFS electrolyzer"]

plt.figure(figsize = (6,4))
n_column_PEM = electrolyzer_data.columns.get_loc("PEM electrolyzer")
p_ptg_PEM = electrolyzer_data.iloc[1:,n_column_PEM].dropna().values.astype(float)
eta_PEM = electrolyzer_data.iloc[1:,n_column_PEM +1].dropna().values.astype(float)
plt.plot(p_ptg_PEM, eta_PEM, marker = ".", label = "PEM")

n_column_HTP_AEC = electrolyzer_data.columns.get_loc("HTP-AEC electrolyzer efficiency")
p_ptg_HTP_AEC = electrolyzer_data.iloc[1:,n_column_HTP_AEC].dropna().values.astype(float)
eta_HTP_AEC = electrolyzer_data.iloc[1:,n_column_HTP_AEC +1].dropna().values.astype(float)
plt.plot(p_ptg_HTP_AEC, eta_HTP_AEC, marker = "x", label = "HTP-AEC")

n_column_LTP_AEC = electrolyzer_data.columns.get_loc("LTP-AEC electrolyzer efficiency")
p_ptg_LTP_AEC = electrolyzer_data.iloc[1:,n_column_LTP_AEC].dropna().values.astype(float)
eta_LTP_AEC = electrolyzer_data.iloc[1:,n_column_LTP_AEC +1].dropna().values.astype(float)
plt.plot(p_ptg_LTP_AEC, eta_LTP_AEC, marker = "+", label = "LTP-AEC")

n_column_HP_SOEC = electrolyzer_data.columns.get_loc("HP-SOEC-WFS electrolyzer efficiency")
p_ptg_HP_SOEC = electrolyzer_data.iloc[1:,n_column_HP_SOEC].dropna().values.astype(float)
eta_HP_SOEC = electrolyzer_data.iloc[1:,n_column_HP_SOEC +1].dropna().values.astype(float)
plt.plot(p_ptg_HP_SOEC, eta_HP_SOEC, marker = "o", label = "HP-SOEC")

n_column_LP_SOEC = electrolyzer_data.columns.get_loc("LP-SOEC-WFS electrolyzer efficiency")
p_ptg_LP_SOEC = electrolyzer_data.iloc[1:,n_column_LP_SOEC].dropna().values.astype(float)
eta_LP_SOEC = electrolyzer_data.iloc[1:,n_column_LP_SOEC +1].dropna().values.astype(float)
plt.plot(p_ptg_LP_SOEC, eta_LP_SOEC, marker = "^", label = "HP-SOEC")
plt.xlabel("Load [%]")
plt.ylabel("Efficiency [%]")
plt.grid()
plt.legend()
plt.show()

#H2 production
plt.figure(figsize = (6,4))
n_column_PEM = electrolyzer_data.columns.get_loc("PEM electrolyzer H2 production")
p_ptg_PEM = electrolyzer_data.iloc[1:,n_column_PEM].dropna().values.astype(float)[:-1]
H2prod_PEM = electrolyzer_data.iloc[1:,n_column_PEM +1].dropna().values.astype(float)[:-1]
plt.plot(p_ptg_PEM, H2prod_PEM, marker = ".", label = "PEM")

n_column_HTP_AEC = electrolyzer_data.columns.get_loc("HTP-AEC electrolyzer H2 production")
p_ptg_HTP_AEC = electrolyzer_data.iloc[1:,n_column_HTP_AEC].dropna().values.astype(float)
H2prod_HTP_AEC = electrolyzer_data.iloc[1:,n_column_HTP_AEC +1].dropna().values.astype(float)
plt.plot(p_ptg_HTP_AEC, H2prod_HTP_AEC, marker = "x", label = "HTP-AEC")

n_column_LTP_AEC = electrolyzer_data.columns.get_loc("LTP-AEC electrolyzer H2 production")
p_ptg_LTP_AEC = electrolyzer_data.iloc[1:,n_column_LTP_AEC].dropna().values.astype(float)
H2prod_LTP_AEC = electrolyzer_data.iloc[1:,n_column_LTP_AEC +1].dropna().values.astype(float)
plt.plot(p_ptg_LTP_AEC, H2prod_LTP_AEC, marker = "+", label = "LTP-AEC")

n_column_HP_SOEC = electrolyzer_data.columns.get_loc("HP-SOEC-WFS electrolyzer H2 production")
p_ptg_HP_SOEC = electrolyzer_data.iloc[1:,n_column_HP_SOEC].dropna().values.astype(float)
H2prod_HP_SOEC = electrolyzer_data.iloc[1:,n_column_HP_SOEC +1].dropna().values.astype(float)
plt.plot(p_ptg_HP_SOEC, H2prod_HP_SOEC, marker = "o", label = "HP-SOEC")

n_column_LP_SOEC = electrolyzer_data.columns.get_loc("LP-SOEC-WFS electrolyzer H2 production")
p_ptg_LP_SOEC = electrolyzer_data.iloc[1:,n_column_LP_SOEC].dropna().values.astype(float)
H2prod_LP_SOEC = electrolyzer_data.iloc[1:,n_column_LP_SOEC +1].dropna().values.astype(float)
plt.plot(p_ptg_LP_SOEC, H2prod_LP_SOEC, marker = "^", label = "HP-SOEC")
plt.xlabel("Load [MWh]")
plt.ylabel("H2 production [kg]")
plt.grid()
plt.legend()
plt.show()