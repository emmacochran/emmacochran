import pandas as pd
import numpy as num
import matplotlib
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.metrics import r2_score, make_scorer, mean_squared_error
import tensorflow as tf
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPRegressor
from scikeras.wrappers import KerasRegressor
from pylr2 import regress2
import shap


starttime=time.perf_counter()

halfhourlydf = pd.read_csv('AMF_US-MOz_FLUXNET_SUBSET_HH_2004-2019_3-5.csv')
basedf = pd.read_csv('AMF_US-MOz_BASE_HH_9-5.csv')
data = pd.merge(halfhourlydf, basedf, on='TIMESTAMP_START')
data = data.replace(-9999, num.NaN)

#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------

#constants
rho_b = 856 #kg/m^3
c_ps = 750 #j/kg/K
rho_w = 1000 #kg/m^3
c_pw = 4200 #j/kg/K
delta_z = 0.1 #m

#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------

#calculate new G_0
data['c_v'] = ((rho_b * c_ps) + (rho_w * c_pw * (data['SWC_F_MDS_1'] / 100))) / 1000 #/100 is to convert percentage, /1000 is to convert m^3 to kg
data['T-1'] = (data['TS_F_MDS_1'] - data['TS_F_MDS_1'].shift()) / 1800
data['G_0'] = data['G_F_MDS'] + (data['c_v'] * data['T-1'] * delta_z)
data['TIMESTAMP_START'] = data['TIMESTAMP_START'].astype(str)
data['TOD'] = data['TIMESTAMP_START'].str[8:]
data['month'] = data['TIMESTAMP_START'].str[4:6]
data['year'] = data['TIMESTAMP_START'].str[:4]
data['TOD'] = data['TOD'].astype(int)
data['month'] = data['month'].astype(int)
data['year'] = data['year'].astype(int)
data = data.replace([num.inf, -num.inf], num.nan)

#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------

#energy calcs
# data['A_LE_H'] = data['LE_F_MDS'] + data['H_F_MDS'] + data['SH_1_1_1'] + data['SLE_1_1_1']
# data['A_Rnet'] = data['NETRAD'] - data['G_0']
# data['SEB'] = (data['A_LE_H'] / data['A_Rnet'])
data['SEB'] = data['NETRAD']-data['LE_F_MDS']-data['H_F_MDS']-data['SH_1_1_1']-data['SLE_1_1_1']-data['G_0']
# data['SEB'] = data.loc[(data['SEB'] >= 0) & (data['SEB'] <= 1), 'SEB']

#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------

anndataALL = data[['SEB','TA_F','VPD_F','WS_F','WD','USTAR','SWC_F_MDS_1','month','NEE_VUT_REF', 'TOD']].copy()
anndataALL = anndataALL.dropna()
#scale data for model
#you don't have to do this for RF, but you ABSOLUTELY have to do it for neural networks
columns_to_scale = anndataALL.columns.difference(['SEB'])
anndataALL_scaled = anndataALL.copy()
anndataALL_scaled[columns_to_scale] = (2 * (anndataALL_scaled[columns_to_scale] - anndataALL_scaled[columns_to_scale].min()) / 
                                 (anndataALL_scaled[columns_to_scale].max() - anndataALL_scaled[columns_to_scale].min())) - 1
x = anndataALL_scaled.drop('SEB', axis=1)
y = anndataALL_scaled.loc[:, 'SEB'].values
#split data into training and testing
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.66, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------

datadrought = data.loc[(data['year'] == 2012) & (data['month'].isin([5,6,7,8,9,10]))]
# anndatadrought = datadrought[['SEB','TA_F','VPD_F','PA_F','WS_F','WD','USTAR','CO2_F_MDS','TS_F_MDS_1','SWC_F_MDS_1','H2O_1_1_1','TOD','month','NEE_VUT_REF']].copy()
anndatadrought = datadrought[['SEB','TA_F','VPD_F','WS_F','WD','USTAR','SWC_F_MDS_1','month','NEE_VUT_REF', 'TOD']].copy()
# anndatadrought = datadrought[['SEB','TA_F','VPD_F','PA_F','WS_F','WD','USTAR','RH','CO2_F_MDS','TS_F_MDS_1','SWC_F_MDS_1','H2O_1_1_1','FC_1_1_1','SC_1_1_1','TOD','month','NEE_VUT_REF','RECO_NT_VUT_REF','GPP_NT_VUT_REF']].copy()
anndatadrought = anndatadrought.dropna()
#scale data for model
columns_to_scale = anndatadrought.columns.difference(['SEB'])
anndatadrought_scaled = anndatadrought.copy()
anndatadrought_scaled[columns_to_scale] = (2 * (anndatadrought_scaled[columns_to_scale] - anndatadrought_scaled[columns_to_scale].min()) / 
                                 (anndatadrought_scaled[columns_to_scale].max() - anndatadrought_scaled[columns_to_scale].min())) - 1
xdrog = anndatadrought_scaled.drop('SEB', axis=1)
ydrog = anndatadrought_scaled.loc[:, 'SEB'].values
x_traindrog, x_tempdrog, y_traindrog, y_tempdrog = train_test_split(xdrog, ydrog, test_size=0.66, random_state=42)
x_valdrog, x_testdrog, y_valdrog, y_testdrog = train_test_split(x_tempdrog, y_tempdrog, test_size=0.5, random_state=42)

#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------

datawet = data.loc[(data['year'] == 2009) & (data['month'].isin([5,6,7,8,9,10]))]
# anndatawet = datawet[['SEB','TA_F','VPD_F','PA_F','WS_F','WD','USTAR','CO2_F_MDS','TS_F_MDS_1','SWC_F_MDS_1','H2O_1_1_1','TOD','month','NEE_VUT_REF']].copy()
anndatawet = datawet[['SEB','TA_F','VPD_F','WS_F','WD','USTAR','SWC_F_MDS_1','month','NEE_VUT_REF', 'TOD']].copy()
# anndatawet = datawet[['SEB','TA_F','VPD_F','PA_F','WS_F','WD','USTAR','RH','CO2_F_MDS','TS_F_MDS_1','SWC_F_MDS_1','H2O_1_1_1','FC_1_1_1','SC_1_1_1','TOD','month','NEE_VUT_REF','RECO_NT_VUT_REF','GPP_NT_VUT_REF']].copy()
anndatawet = anndatawet.dropna()
#scale data for model
columns_to_scale = anndatawet.columns.difference(['SEB'])
anndatawet_scaled = anndatawet.copy()
anndatawet_scaled[columns_to_scale] = (2 * (anndatawet_scaled[columns_to_scale] - anndatawet_scaled[columns_to_scale].min()) / 
                                 (anndatawet_scaled[columns_to_scale].max() - anndatawet_scaled[columns_to_scale].min())) - 1
xwet = anndatawet_scaled.drop('SEB', axis=1)
ywet = anndatawet_scaled.loc[:, 'SEB'].values
x_trainwet, x_tempwet, y_trainwet, y_tempwet = train_test_split(xwet, ywet, test_size=0.66, random_state=42)
x_valwet, x_testwet, y_valwet, y_testwet = train_test_split(x_tempwet, y_tempwet, test_size=0.5, random_state=42)

#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------
print("data preparation complete")
#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------

#best ANN model with best arch, hyperparameters, and 10 initializations each run
def its_giving_best_model(x_train, y_train, x_test, y_test, x_val, y_val, num_initializations=10):
    best_model = None
    best_error = float('inf')
    
    for i in range(num_initializations):
        # Initialize model with varying weights
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(x_train.shape[1]*4, activation='relu', input_dim=x_train.shape[1], kernel_initializer='random_uniform'),
            tf.keras.layers.Dense(x_train.shape[1]*2, activation='relu', kernel_initializer='random_uniform'),
            tf.keras.layers.Dense(x_train.shape[1], activation='relu', kernel_initializer='random_uniform'),
            tf.keras.layers.Dense(1)
        ])
        # Compile and train the model
        model.compile(optimizer='rmsprop', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=0, validation_data=(x_val, y_val))
        # Evaluate the model
        y_pred = model.predict(x_test)
        error = mean_squared_error(y_test, y_pred)
        # Update the best model and error
        if error < best_error:
            best_error = error
            best_model = model
    y_pred = best_model.predict(x_test)
    r_squared = r2_score(y_test, y_pred)
    loss = best_model.evaluate(x_test, y_test)
    return best_model, best_error, r_squared, loss, y_test, y_pred

#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------

best_model, best_error, r_squared, loss, y_test, y_pred = its_giving_best_model(x_train, y_train, x_test, y_test, x_val, y_val, num_initializations=10)
print("Best mean squared error:", best_error)
print("best r2: ", r_squared)

#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------

# # Example usage:
# best_modeldrog, best_errordrog, r_squareddrog, lossdrog, y_testdrog, y_preddrog = its_giving_best_model(x_traindrog, y_traindrog, x_testdrog, y_testdrog, x_valdrog, y_valdrog, num_initializations=10)
# print("Best mean squared error DROUGHT:", best_errordrog)
# print("best r2 DROUGHT: ", r_squareddrog)
# best_modelwet, best_errorwet, r_squaredwet, losswet, y_testwet, y_predwet = its_giving_best_model(x_trainwet, y_trainwet, x_testwet, y_testwet, x_valwet, y_valwet, num_initializations=10)
# print("Best mean squared error WET:", best_errorwet)
# print("best r2 WET: ", r_squaredwet)

#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------

# #feature ranking for permutation feature importance

perm_import = permutation_importance(best_model, x_test, y_test, scoring='r2',n_repeats = 10, random_state=42)
feat_import = perm_import.importances_mean
feat_names = x.columns
feat_df = pd.DataFrame({'Feature': feat_names, 'Importance': feat_import})
feat_df = feat_df.sort_values(by='Importance', ascending=False)

# perm_importdrog = permutation_importance(best_modeldrog, x_testdrog, y_testdrog, scoring='r2',n_repeats = 10, random_state=42)
# feat_importdrog = perm_importdrog.importances_mean
# feat_namesdrog = xdrog.columns
# feat_dfdrog = pd.DataFrame({'Feature': feat_namesdrog, 'Importance': feat_importdrog})
# feat_dfdrog = feat_dfdrog.sort_values(by='Importance', ascending=False)

# perm_importwet = permutation_importance(best_modelwet, x_testwet, y_testwet, scoring='r2',n_repeats = 10, random_state=42)
# feat_importwet = perm_importwet.importances_mean
# feat_nameswet = xwet.columns
# feat_dfwet = pd.DataFrame({'Feature': feat_nameswet, 'Importance': feat_importwet})
# feat_dfwet = feat_dfwet.sort_values(by='Importance', ascending=False)

print('Feature ranking:')
print(feat_df)
# print('Drought feature ranking:')
# print(feat_dfdrog)
# print('wet feature ranking:')
# print(feat_dfwet)

#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------

# #graph
# x = num.arange(-100,200)
# ybase = x
# matplotlib.rc('xtick', labelsize = 20)
# matplotlib.rc('ytick', labelsize = 20)
# drogbest = regress2(y_testdrog, y_preddrog, _method_type_2="reduced major axis")
# drogbest = drogbest['slope']*x + drogbest['intercept']
# wetbest = regress2(y_testwet, y_predwet, _method_type_2="reduced major axis")
# wetbest = wetbest['slope']*x + wetbest['intercept']
# fig, ax = plt.subplots()
# ax.scatter(y_testdrog, y_preddrog, s=10, alpha=0.25, color='red')
# ax.scatter(y_testwet, y_predwet, s=10, alpha=0.25, color='blue')
# ax.plot(x, drogbest, color = 'red', linewidth = 3, label='Drought')
# ax.plot(x, wetbest, color = 'blue', linewidth = 3, label='Wet')
# ax.plot(x, ybase, 'k--', linewidth = 3, label='1:1')
# ax.set_xlabel(r'Observed SEB', fontsize=20)
# ax.set_ylabel(r'Model SEB', fontsize=20)
# ax.legend(fontsize=20)
# plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------

# #SHAP
# # Assuming you have a trained model named 'model' and your test data 'x_test'
# # Function to wrap the model
# def model_wrapper(x):
#     # Assuming your model expects input shape (batch_size, input_dim)
#     return best_modeldrog.predict(x)
# # Create a KernelExplainer instance
# explainerdrog = shap.KernelExplainer(model_wrapper, shap.sample(x_traindrog, 100))
# # Compute SHAP values for the test data
# shap_valuesdrog = explainerdrog.shap_values(x_testdrog)
# # Plot the summary plot
# shap.summary_plot(shap_valuesdrog, x_testdrog)

# def model_wrapperwet(x):
#     return best_modelwet.predict(x)
# explainerwet = shap.KernelExplainer(model_wrapperwet, shap.sample(x_trainwet, 100))  
# shap_valueswet = explainerwet.shap_values(x_testwet)
# shap.summary_plot(shap_valueswet, x_testwet)

#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------

endtime=time.perf_counter()
print(f"Execution Time: {endtime - starttime: 0.3f}")

#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------

# #base ANN model with best arch and hyperparameters--- NO init
# def its_giving_model():
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Dense(x_train.shape[1]*4, activation='relu', input_dim=x_train.shape[1]),
#         tf.keras.layers.Dense(x_train.shape[1]*2, activation='relu'),
#         tf.keras.layers.Dense(x_train.shape[1], activation='relu'),
#         tf.keras.layers.Dense(1)
#     ])
#     model.compile(optimizer='rmsprop', loss='mean_squared_error')
#     model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=0, validation_data=(x_test, y_test))
#     y_pred = model.predict(x_test)
#     r_squared = r2_score(y_test, y_pred)
#     loss = model.evaluate(x_test, y_test)
#     return model, r_squared, loss
# #initialize model
# # model, r_squared, loss = its_giving_model()

# #testing for best arch
# #defining the archs
# archs = [
#      #40
#     [tf.keras.layers.Dense(x_train.shape[1]*4, activation='relu', input_dim=x_train.shape[1]),
#      tf.keras.layers.Dense(1)],
#      #40 > 20
#     [tf.keras.layers.Dense(x_train.shape[1]*4, activation='relu', input_dim=x_train.shape[1]),
#      tf.keras.layers.Dense(x_train.shape[1]*2, activation='relu'),
#      tf.keras.layers.Dense(1)],
#      #40 > 20 > 10
#      #ideal
#     [tf.keras.layers.Dense(x_train.shape[1]*4, activation='relu', input_dim=x_train.shape[1]),
#      tf.keras.layers.Dense(x_train.shape[1]*2, activation='relu'),
#      tf.keras.layers.Dense(x_train.shape[1], activation='relu'),
#      tf.keras.layers.Dense(1)]
# ]
# # Define the function to create and train ANN model
# def create_and_train_model(architecture, x_train, y_train, x_test, y_test):
#     model = tf.keras.models.Sequential(architecture)
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)
#     y_pred = model.predict(x_test)
#     r_squared = r2_score(y_test, y_pred)
#     loss = model.evaluate(x_test, y_test)
#     return r_squared, loss
# # results = []
# # for arch in archs:
# #     r_squared, loss = create_and_train_model(arch, x_train, y_train, x_test, y_test)
# #     results.append((arch, r_squared, loss))
# # for arch, r_squared, loss in results:
# #     # print("Arch: ", arch)
# #     print("R2: ", r_squared)
# #     # print('mean standard error: ', loss)



# #find the best hyperparameters (do after finding best arch)
# #hyperparameter tuning
# def create_model(optimizer='adam'):
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Dense(64, activation='relu', input_dim=x_train.shape[1]),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dense(1)
#     ])
#     model.compile(optimizer=optimizer, loss='mean_squared_error')
#     return model
# # #need a keras wrapper to use gridsearch cv 
# # keras_model = KerasRegressor(build_fn=create_model, verbose=0)
# #set hyperparameter grid
# param_grid = {
#     'batch_size': [32, 64, 128],
#     'epochs': [50, 100, 200],
#     'optimizer': ['adam', 'rmsprop']
# }
# # #grid search
# # grid_search = GridSearchCV(estimator=keras_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
# # grid_search.fit(x_train, y_train)
# # # print the best hyperparameters
# # best_params = grid_search.best_params_
# # print("Best Hyperparameters:", best_params)
# # #Best Hyperparameters: {'batch_size': 64, 'epochs': 100, 'optimizer': 'rmsprop'}

