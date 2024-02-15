#in progress
#predicting surface energy imbalance using data from the MOFLUX flux tower in central Missouri
#random forest regression is used

import pandas as pd
import numpy as num
import matplotlib.pyplot as plt
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from scipy.stats import randint
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import shap

starttime=time.perf_counter()

halfhourlydf = pd.read_csv('AMF_US-MOz_FLUXNET_SUBSET_HH_2004-2019_3-5.csv')
basedf = pd.read_csv('AMF_US-MOz_BASE_HH_9-5.csv')
data = pd.merge(halfhourlydf, basedf, on='TIMESTAMP_START')
data = data.replace(-9999, num.NaN)

#constants
rho_b = 856 #kg/m^3
c_ps = 750 #j/kg/K
rho_w = 1000 #kg/m^3
c_pw = 4200 #j/kg/K
delta_z = 0.1 #m

#calculate new G_0
data['c_v'] = ((rho_b * c_ps) + (rho_w * c_pw * (data['SWC_F_MDS_1'] / 100))) / 1000 #/100 is to convert percentage, /1000 is to convert m^3 to kg
data['T-1'] = (data['TS_F_MDS_1'] - data['TS_F_MDS_1'].shift()) / 1800
data['G_0'] = data['G_F_MDS'] + (data['c_v'] * data['T-1'] * delta_z)

#trimming 
data['TIMESTAMP_START'] = data['TIMESTAMP_START'].astype(str)
data['TOD'] = data['TIMESTAMP_START'].str[8:]
data['month'] = data['TIMESTAMP_START'].str[4:6]
data['year'] = data['TIMESTAMP_START'].str[:4]
data['TOD'] = data['TOD'].astype(int)
data['month'] = data['month'].astype(int)
data['year'] = data['year'].astype(int)
data = data.replace([num.inf, -num.inf], num.nan)

#energy calcs
data['A_LE_H'] = data['LE_F_MDS'] + data['H_F_MDS'] + data['SH_1_1_1'] + data['SLE_1_1_1']
data['A_Rnet'] = data['NETRAD'] - data['G_0']
data['SEB'] = (data['A_LE_H'] / data['A_Rnet'])
data['SEB'] = data.loc[(data['SEB'] >= 0) & (data['SEB'] <= 1), 'SEB']

# seb = data['SEB'].values.tolist()
# seb = [x for x in seb if str(x) != 'nan']

# seb = sorted(seb)
# # plt.scatter(data['A_Rnet'], data['A_LE_H'])
# plt.plot(seb)
# plt.show()

rfdata = data[['SEB', 'TOD', 'month', 'year','VPD_F', 'WS_F', 'WD', 'USTAR', 'SWC_F_MDS_1', 'TA_F', 'TS_F_MDS_1', 'NEE_VUT_REF', 'GPP_NT_VUT_REF']].copy()
rfdata = rfdata.dropna()

rfdata = rfdata.loc[rfdata['year']==2015]
x = rfdata.drop('SEB', axis=1)
y = rfdata.loc[:, 'SEB'].values
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

rf = RandomForestRegressor()
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("R^2: ", r2)

# plt.barh(x.columns, rf.feature_importances_)
# plt.show()

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values, x_test, plot_type="bar")
shap.summary_plot(shap_values, x_test)


endtime=time.perf_counter()
print(f"Execution Time: {endtime - starttime: 0.3f}")
