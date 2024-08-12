import pandas as pd
import numpy as num
import matplotlib.pyplot as plt
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from scipy.stats import randint
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, mean_absolute_error
import shap
import matplotlib
import matplotlib.pyplot as plt
from pylr2 import regress2

starttime=time.perf_counter()

#reading in data
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

#calculate new G_0 for imbalance eq
#data product gives you G from depth of heat plate, but we want G at the surface, so we need to correct for heat transfer
#between heat plat and SFC, so we get G_0
data['c_v'] = ((rho_b * c_ps) + (rho_w * c_pw * (data['SWC_F_MDS_1'] / 100))) / 1000 #/100 is to convert percentage, /1000 is to convert m^3 to kg
data['T-1'] = (data['TS_F_MDS_1'] - data['TS_F_MDS_1'].shift()) / 1800
data['G_0'] = data['G_F_MDS'] + (data['c_v'] * data['T-1'] * delta_z)

#reformatting some columns
data['TIMESTAMP_START'] = data['TIMESTAMP_START'].astype(str)
data['TOD'] = data['TIMESTAMP_START'].str[8:]
data['month'] = data['TIMESTAMP_START'].str[4:6]
data['year'] = data['TIMESTAMP_START'].str[:4]
data['TOD'] = data['TOD'].astype(int)
data['month'] = data['month'].astype(int)
data['year'] = data['year'].astype(int)
#any infinite numbers are errors of data collection or data processing, so you can ignore them
data = data.replace([num.inf, -num.inf], num.nan)

#energy calcs

#this first way was how I was doing it first, but it is not as good as the second way (the uncommented way)
# data['A_LE_H'] = data['LE_F_MDS'] + data['H_F_MDS'] + data['SH_1_1_1'] + data['SLE_1_1_1']
# data['A_Rnet'] = data['NETRAD'] - data['G_0']
# data['SEBperc'] = (data['A_LE_H'] / data['A_Rnet'])
# data['SEBperc'] = data.loc[(data['SEBperc'] >= 0) & (data['SEBperc'] <= 1), 'SEBperc']

#this way handles numbers closer to zero much much better
data['SEB'] = data['NETRAD']-data['LE_F_MDS']-data['H_F_MDS']-data['SH_1_1_1']-data['SLE_1_1_1']-data['G_0']

# graphdf = pd.DataFrame({'ypred': data['SEB']})

#there were a lot of negative imb values, so I wanted to see at what point in the day were those occuring
# negdf = data[(data['SEB']<0) & (data['SEB']>-50)]
# uniq = negdf['TOD'].value_counts()
# print(uniq)

#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------
#code to see the distribution of SEB

# seb = data['SEB'].values.tolist()
# seb = [x for x in seb if str(x) != 'nan']

# seb = sorted(seb)
# # plt.scatter(data['A_Rnet'], data['A_LE_H'])
# plt.plot(seb)
# plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------

#line 83 for when you want to run the model on just one year, makes things faster for when you're playing around with stuff
# data = data.loc[data['year']==2019]
#isolating only the columns I was interested in, helps cuts down run time
rfdata = data[['SEB','TA_F','VPD_F','WS_F','WD','USTAR','SWC_F_MDS_1','month','NEE_VUT_REF', 'TOD']].copy()
rfdata = rfdata.dropna()

x = rfdata.drop('SEB', axis=1)
y = rfdata.loc[:, 'SEB'].values

#split data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------

#isolating drought data by using growing season data from 2012
datadrought = data.loc[(data['year'] == 2012) & (data['month'].isin([5,6,7,8,9,10]))]
#then the same process
rfdatadrought = datadrought[['SEB','TA_F','VPD_F','WS_F','WD','USTAR','SWC_F_MDS_1','month','NEE_VUT_REF', 'TOD']].copy()
rfdatadrought = rfdatadrought.dropna()
xdrog = rfdatadrought.drop('SEB', axis=1)
ydrog = rfdatadrought.loc[:, 'SEB'].values
x_traindrog, x_testdrog, y_traindrog, y_testdrog = train_test_split(xdrog, ydrog, test_size=0.2, random_state=42)

#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------

#same for isolating wet periods, year was determined by PLWP integrals (i think?)
datawet = data.loc[(data['year'] == 2009) & (data['month'].isin([5,6,7,8,9,10]))]
rfdatawet = datawet[['SEB','TA_F','VPD_F','WS_F','WD','USTAR','SWC_F_MDS_1','month','NEE_VUT_REF', 'TOD']].copy()
rfdatawet.loc[rfdatawet['NEE_VUT_REF']>20, 'NEE_VUT_REF'] = num.nan
rfdatawet = rfdatawet.dropna()
xwet = rfdatawet.drop('SEB', axis=1)
ywet = rfdatawet.loc[:, 'SEB'].values
x_trainwet, x_testwet, y_trainwet, y_testwet = train_test_split(xwet, ywet, test_size=0.2, random_state=42)

print('all data prep complete')

#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------

#this next block takes FOREVER to run, so run it once and do the hyperparameter tuning, then don't do it ever again lol

# rf = RandomForestRegressor()

# #finding the best parameters for model
# param_grid = {
#     'n_estimators': [100, 200, 300, 500],  # Number of trees in the forest
#     'max_depth': [20, 30, 40, 50],  # Maximum depth of the tree
#     'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
#     'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
# }
# scorer = make_scorer(r2_score)
# print('scorer done')

# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring=scorer, cv=5)
# print('grid search prepped')

# grid_search.fit(x_train, y_train)
# print('grid search fitted')

# print("Best Parameters:", grid_search.best_params_)
# best_rf_model = grid_search.best_estimator_
# y_pred = best_rf_model.predict(x_test)
# r2 = r2_score(y_test, y_pred)
# print("R-squared on Test Set:", r2)

#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------
#parameters determined from earlier grid search cv
rf = RandomForestRegressor(max_depth=50, min_samples_leaf=1, min_samples_split=2, n_estimators=500)

#time to actually fit the model
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

#statistics to check how well the model performed
r2 = r2_score(y_test, y_pred)
rmse = num.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
bias_error = num.mean(y_pred - y_test)
print("Bias Error:", bias_error)
print("MAE:", mae)
print("R^2: ", r2)
print("RMSE: ", rmse)

graphdf = pd.DataFrame({'ypred': y_pred, 'ytest': y_test})
graphdf=graphdf.to_csv('graphdf.csv', index=False)

# #graph
# x = num.arange(-500,500)
# ybase = x
# matplotlib.rc('xtick', labelsize = 20)
# matplotlib.rc('ytick', labelsize = 20)
# bestfit = regress2(y_test, y_pred, _method_type_2="reduced major axis")
# bestfit = bestfit['slope']*x + bestfit['intercept']
# fig, ax = plt.subplots()
# ax.scatter(y_test, y_pred, s=10, alpha=0.25, color='blue')
# ax.plot(x, bestfit, color = 'blue', linewidth = 3, label='Drought')
# ax.plot(x, ybase, 'k--', linewidth = 3, label='1:1')
# ax.set_xlabel(r'Observed SEB', fontsize=20)
# ax.set_ylabel(r'Model SEB', fontsize=20)
# ax.legend(fontsize=20)
# plt.show()

# plt.barh(x.columns, rf.feature_importances_)
# plt.show()

# feat_df = pd.DataFrame({'Feature': x.columns, 'Importance': rf.feature_importances_})
# feat_df = feat_df.sort_values(by='Importance', ascending=False)
# print(feat_df)

#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------

# # rf = RandomForestRegressor(max_depth=50, min_samples_leaf=1, min_samples_split=2, n_estimators=500)
# rf.fit(x_traindrog, y_traindrog)
# y_preddrog = rf.predict(x_testdrog)
# r2drog = r2_score(y_testdrog, y_preddrog)
# print("R^2 drought: ", r2drog)
# feat_dfdrog = pd.DataFrame({'Feature': xdrog.columns, 'Importance': rf.feature_importances_})
# feat_dfdrog = feat_dfdrog.sort_values(by='Importance', ascending=False)
# print()
# print(feat_dfdrog)
# r2 = r2_score(y_testdrog, y_preddrog)
# rmse = num.sqrt(mean_squared_error(y_testdrog, y_preddrog))
# mae = mean_absolute_error(y_testdrog, y_preddrog)
# bias_error = num.mean(y_preddrog - y_testdrog)
# print("Bias Error drog:", bias_error)
# print("MAE drog:", mae)
# print("R^2 drog: ", r2)
# print("RMSE drog: ", rmse)


# rf.fit(x_trainwet, y_trainwet)
# y_predwet = rf.predict(x_testwet)
# r2wet = r2_score(y_testwet, y_predwet)
# print("R^2 wet: ", r2wet)
# feat_dfwet = pd.DataFrame({'Feature': xwet.columns, 'Importance': rf.feature_importances_})
# feat_dfwet = feat_dfwet.sort_values(by='Importance', ascending=False)
# print()
# print(feat_dfwet)
# r2 = r2_score(y_testwet, y_predwet)
# rmse = num.sqrt(mean_squared_error(y_testwet, y_predwet))
# mae = mean_absolute_error(y_testwet, y_predwet)
# bias_error = num.mean(y_predwet - y_testwet)
# print("Bias Error wet:", bias_error)
# print("MAE wet:", mae)
# print("R^2 wet: ", r2)
# print("RMSE wet: ", rmse)


#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------

#permutation feature importance ranking

#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------

#https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/NHANES%20I%20Survival%20Model.html#SHAP-Summary-Plot

# explainer = shap.TreeExplainer(rf)  
# shap_values = explainer.shap_values(x_test)
# shap.dependence_plot('NEE_VUT_REF', shap_values,x_test, interaction_index='TOD')
# shap.summary_plot(shap_values, x_test)

# explainer = shap.TreeExplainer(rf)  
# shap_values = explainer.shap_values(x_testdrog)
# shap.dependence_plot('NEE_VUT_REF', shap_values,x_testdrog, interaction_index='SWC_F_MDS_1')
# shap.summary_plot(shap_values, x_testdrog)

# explainer = shap.TreeExplainer(rf)  
# shap_values = explainer.shap_values(x_testwet)
# shap.dependence_plot('NEE_VUT_REF', shap_values,x_testwet, interaction_index='TOD')
# shap.dependence_plot('NEE_VUT_REF', shap_values,x_testwet, interaction_index='SWC_F_MDS_1')
# shap.summary_plot(shap_values, x_testwet)

#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------

endtime=time.perf_counter()
print(f"Execution Time: {endtime - starttime: 0.3f}")

