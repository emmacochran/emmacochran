#predawn integral analysis for MOFLUX site
#creates csv with continuous community averaged predawn leaf water potentials for growing season monitoring periods
#non-growing season is excluded

import pandas as pd
import numpy as num
import matplotlib.pyplot as plt
from scipy import integrate as intg
import time
from datetime import datetime

start_time=time.perf_counter()
root='PLWP'
#load PDWLP as data file
df=pd.read_csv(root+"/PDWLP.csv")
df.drop(0,axis=0, inplace=True)

#average PLWP values for each day
df[" PLWP"]=pd.to_numeric(df[" PLWP"])
df=df.groupby(['Year', ' DOY'], as_index=False)[' PLWP'].mean()

#list of years
yearList=df['Year'].unique()
yearList=yearList.tolist()
#get rid of 2004 due to insufficient data
if yearList[0]=='2004':
    yearList.pop(0)

#create separate csv files containing all the data for each year
for i,value in enumerate(yearList):
    df[df['Year']==value].to_csv(root+ '/' + r'Year_'+str(value)+r'_Clean.csv',index=False, na_rep='N/A')

#function for inserting row into middle of df 
def insertRow(row_number,df,row_value):
    startUpper=0
    endUpper=row_number
    startLower=row_number
    endLower=df.shape[0]
    upperHalf=[*range(startUpper, endUpper,1)]
    lowerHalf=[*range(startLower, endLower,1)]
    lowerHalf=[x.__add__(1) for x in lowerHalf]
    index_=upperHalf+lowerHalf
    df.index=index_
    df.loc[row_number]=row_value
    df=df.sort_index()
    return df

#final df with all growing seasons
growingseasonsdf = []

#create variable to hold values within for loop
integralList= []
minPLWPList= []
allYearData= []
histogramdf=pd.DataFrame()
fullPLWPgrowingseason = []
#create blank rows for every day not accounted for
yearListStrs= [str(x) for x in yearList]
for year in yearListStrs:
    df1=pd.read_csv(root+ '/' + "Year_" + year + "_Clean.csv")
    firstDay=df1[' DOY'][0]
    lastDay=df1[' DOY'].iat[-1]
    counter=1
    dayCounter=firstDay
    for i in range(firstDay, lastDay):
        #if the next row isnt a consecutive day, add in a row, else, skip
        if df1[' DOY'][counter] != dayCounter+1:
            row_number = counter
            row_value = [year, dayCounter+1, num.NaN]
            if row_number>df.index.max()+1:
                print("something went wrong")
            else:
                df1= insertRow(row_number, df1, row_value)            
            counter+=1
            dayCounter+=1
        else:
            counter+=1
            dayCounter+=1
    #interpolate the values for the inbetween days 
    df1=df1.interpolate()
    growingseasonsdf.append(df1)
    
growingseasonsdf=pd.concat(growingseasonsdf)
growingseasonsdf=growingseasonsdf.reset_index()
growingseasonsdf['TIMESTAMP']=num.nan

for ind in growingseasonsdf.index:
    year= growingseasonsdf.iloc[ind]['Year']
    day= growingseasonsdf.iloc[ind][' DOY']
    year = str(year)
    #day+=1
    day = str(day)
    day.rjust(3+len(day), '0')
    timedate= datetime.strptime(year + "-" + day, "%Y-%j").strftime("%Y%m%d")
    growingseasonsdf.at[ind, 'TIMESTAMP']=timedate

# growingseasonsdf=growingseasonsdf.drop(['Year', ' DOY'], axis=1)
growingseasonsdf=growingseasonsdf.drop(['Year'], axis=1)
growingseasonsdf=growingseasonsdf.to_csv('Chapter1_three_bin' +'/'+ r'growing_seasons_all.csv', index=False)

#     #find min value
#     minPLWP= df1[' PLWP'].min()
#     minPLWPList.append(minPLWP)

#     #add in other days
#     counter=1
#     row_value = [year, 1, 0]
#     df1= insertRow(0, df1, row_value)
#     intYear=int(year)
#     if intYear%4==0:
#         isaLeapYear=366
#     else:
#         isaLeapYear=365
#     row_value= [year, isaLeapYear+1, 0]
#     df1= insertRow(df1.shape[0],df1, row_value)
#     for i in range(0, isaLeapYear):
#         if df1[' DOY'][counter] != counter+1:
#             row_value= [year, counter+1, 0]
#             df1= insertRow(counter, df1, row_value)
#             counter +=1
#         else:
#             counter+=1

#     #integral
#     xdata= df1[' DOY']
#     ydata= df1[' PLWP']
#     totalInt = intg.trapz(ydata, xdata)
#     totalInt=round(totalInt,2)
#     integralList.append(totalInt)

#     #plot the PLWP against DOY
#     plt.rcParams["figure.figsize"] = [7.50, 3.50]
#     plt.rcParams["figure.autolayout"] = True
#     x= df1[' DOY']
#     y= df1[' PLWP']
#     ax=plt.subplot()
#     plt.text(0.1,0.1, 'Min: {:+.4f}'.format(minPLWP), ha='center', va='center', transform=ax.transAxes, fontsize = 12, bbox=dict(facecolor = 'green', alpha=0.5))
#     plt.text(0.9,0.1, 'Int: {:+.3f}'.format(totalInt), ha='center', va='center', transform=ax.transAxes, fontsize = 12, bbox=dict(facecolor = 'cyan'), alpha=0.5)
#     ax.plot(x,y,c='g')
#     plt.xlim([125,300])
#     plt.ylim([-4,0])
#     plt.title(year+ ' Predawn Data')
#     plt.xlabel('Day of the Year')
#     plt.ylabel('PLWP')
#     plt.savefig(root+'/'+year+r'PredawnDataClean.png')
#     plt.close()

#     df1=df1.to_csv(root+'/'+r'FULL_'+year+r'.csv')

# #integral analysis
# intmin= min(integralList)
# intmax= max(integralList)
# intavg= round(sum(integralList) / len(integralList), 2)
# #print(integralList, intmin, intmax, intavg)
# #min PLWP analysis
# minPLWPmin= min(minPLWPList)
# minPLWPmax= max(minPLWPList)
# minPLWPavg= round(sum(minPLWPList) / len(minPLWPList), 2)
# #print(minPLWPList, minPLWPmin, minPLWPmax, minPLWPavg)

# #graph integral on line graph
# x=yearListStrs
# y=integralList
# ax=plt.subplot()
# ax.plot(x,y)
# plt.xticks(fontsize= 8)
# plt.yticks(fontsize= 8)
# plt.title('Integral Line Graph')
# plt.xlabel('Year')
# plt.ylabel('Integral (mmol*year)')
# plt.savefig(root+'/'+'IntegralLineGraph.png')
# plt.close()

end_time=time.perf_counter()
print(f"Start Time: {start_time}")
print(f"End Time: {end_time}")
print(f"Execution Time: {end_time - start_time: 0.4f}")
