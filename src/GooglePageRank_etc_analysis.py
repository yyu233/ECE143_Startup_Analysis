#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd                                                             
import numpy as np                                                              
import matplotlib.pyplot as plt                                                 
import seaborn as sns                                                           
import warnings                                                                 
warnings.filterwarnings('ignore')


# In[ ]:


# Slicing Columns


# In[2]:


data = pd.read_csv("CAX_Startup_Data.csv")


# In[4]:


data.shape


# In[5]:


data.info


# In[8]:


companys = data['Company_Name']
companys


# In[186]:


companyStatus = data['Dependent-Company Status']
companyStatus


# In[122]:


expoGlobe = data['Exposure across the globe']
expoGlobe


# In[100]:


numResearchPub = data['Number of  of Research publications']
numResearchPub


# In[20]:


timeMarketSrvOrProd = data['Time to market service or product']
timeMarketSrvOrProd


# In[11]:


googlePageRank = data['google page rank of company website']
googlePageRank


# In[12]:


companyAwards = data['Company awards']
companyAwards


# In[13]:


numDirCompetitors = data['Number of Direct competitors']
numDirCompetitors


# In[14]:


employeePerYear = data['Employees per year of company existence']
employeePerYear


# In[16]:


lastRndFndingAmtMilUSD = data['Last round of funding received (in milionUSD)']
lastRndFndingAmtMilUSD


# In[17]:


survivalRecession = data['Survival through recession, based on existence of the company through recession times']
survivalRecession


# In[18]:


avgTimeInvestment = data['Avg time to investment - average across all rounds, measured from previous investment']
avgTimeInvestment


# In[44]:





# In[59]:


companys.unique()


# In[187]:


companyStatus.unique()


# In[123]:


expoGlobe.unique()


# In[101]:


numResearchPub.unique()


# In[60]:


timeMarketSrvOrProd.unique()


# In[61]:


companyAwards.unique()


# In[62]:


survivalRecession.unique()


# In[63]:


googlePageRank.unique()


# In[64]:


numDirCompetitors.unique()


# In[65]:


employeePerYear.unique()


# In[66]:


lastRndFndingAmtMilUSD.unique()


# In[67]:


avgTimeInvestment.unique()


# In[ ]:


# Data Cleaning


# In[107]:


expoGlobe.replace({"No Info": np.nan}, inplace=True)
expoGlobe.unique()


# In[102]:


numResearchPub.replace({"No Info": np.nan}, inplace=True)
numResearchPub.unique()


# In[97]:


timeMarketSrvOrProd.replace({"No Info": np.nan}, inplace=True)
timeMarketSrvOrProd.unique()


# In[98]:


companyAwards.replace({"No Info": np.nan}, inplace=True)
companyAwards.unique()


# In[99]:


survivalRecession.replace({"No Info": np.nan}, inplace=True)
survivalRecession.unique()


# In[ ]:





# In[75]:


googlePageRank.replace({"No Info": np.nan}, inplace=True)
googlePageRank = googlePageRank.astype(float)
googlePageRank.unique()


# In[77]:


numDirCompetitors.replace({"No Info": np.nan}, inplace=True)
numDirCompetitors = numDirCompetitors.astype(float)
numDirCompetitors.unique()


# In[78]:


employeePerYear.replace({"No Info":np.nan}, inplace=True)
employeePerYear = employeePerYear.astype(float)
employeePerYear.unique()


# In[72]:


lastRndFndingAmtMilUSD.replace({"unknown amount":np.nan, "No Info":np.nan}, inplace=True)
lastRndFndingAmtMilUSD = lastRndFndingAmtMilUSD.astype(float)
lastRndFndingAmtMilUSD.unique()


# In[80]:


avgTimeInvestment.replace({"No Info": np.nan}, inplace=True)
avgTimeInvestment = avgTimeInvestment.astype(float)
avgTimeInvestment.unique()


# In[ ]:


# Get missing value ratio


# In[103]:


catCols = [expoGlobe, numResearchPub,timeMarketSrvOrProd, companyAwards, survivalRecession]
len(catCols)


# In[104]:


numCols = [googlePageRank, numDirCompetitors, employeePerYear, lastRndFndingAmtMilUSD, avgTimeInvestment]
len(numCols)


# In[105]:


catColsPD = pd.concat(catCols, axis=1)
catColsPD


# In[106]:


catColsNull=pd.DataFrame(catColsPD.isnull().sum(),columns=["Null Values"])
catColsNull["% Missing Values"]=(catColsPD.isna().sum()/len(catColsPD)*100)
catColsNull.style.background_gradient(cmap='viridis',low =0.2,high=0.1)


# In[93]:


numColsPD = pd.concat(numCols, axis=1)
numColsPD


# In[95]:


numColsNull=pd.DataFrame(numColsPD.isnull().sum(),columns=["Null Values"])
numColsNull["% Missing Values"]=(numColsPD.isna().sum()/len(numColsPD)*100)
numColsNull.style.background_gradient(cmap='viridis',low =0.2,high=0.1)


# In[ ]:





# In[108]:


# replaced NaN with mode for categorical columns, with mean for numerical columns


# In[128]:


expoGlobeMode = expoGlobe.mode()[0]
expoGlobeMode


# In[130]:


expoGlobe = expoGlobe.fillna(expoGlobeMode)
expoGlobe.isnull().sum()


# In[131]:


numResearchPubMode = numResearchPub.mode()[0]
numResearchPubMode


# In[134]:


numResearchPub = numResearchPub.fillna(numResearchPubMode)
numResearchPub.isnull().sum()


# In[135]:


timeMarketSrvOrProdMode = timeMarketSrvOrProd.mode()[0]
timeMarketSrvOrProdMode


# In[136]:


timeMarketSrvOrProd = timeMarketSrvOrProd.fillna(timeMarketSrvOrProdMode)
timeMarketSrvOrProd.isnull().sum()


# In[141]:


companyAwardsMode = companyAwards.mode()[0]
companyAwardsMode


# In[142]:


companyAwards = companyAwards.fillna(companyAwardsMode)
companyAwards.isnull().sum()


# In[200]:


survivalRecessionMode = survivalRecession.mode()[0]
survivalRecessionMode


# In[199]:


survivalRecession = survivalRecession.fillna(survivalRecessionMode)
survivalRecession.isnull().sum()


# In[ ]:





# In[146]:


googlePageRankMean = googlePageRank.mean()
googlePageRankMean


# In[147]:


googlePageRank = googlePageRank.fillna(googlePageRankMean)
googlePageRank.isnull().sum()


# In[148]:


numDirCompetitorsMean = numDirCompetitors.mean()
numDirCompetitorsMean


# In[149]:


numDirCompetitors = numDirCompetitors.fillna(numDirCompetitorsMean)
numDirCompetitors.isnull().sum()


# In[150]:


employeePerYearMean = employeePerYear.mean()
employeePerYearMean


# In[151]:


employeePerYear = employeePerYear.fillna(employeePerYearMean)
employeePerYear.isnull().sum()


# In[152]:


lastRndFndingAmtMilUSDMean = lastRndFndingAmtMilUSD.mean()
lastRndFndingAmtMilUSDMean


# In[153]:


lastRndFndingAmtMilUSD = lastRndFndingAmtMilUSD.fillna(lastRndFndingAmtMilUSDMean)
lastRndFndingAmtMilUSD.isnull().sum()


# In[154]:


avgTimeInvestmentMean = avgTimeInvestment.mean()
avgTimeInvestmentMean


# In[155]:


avgTimeInvestment = avgTimeInvestment.fillna(avgTimeInvestmentMean)
avgTimeInvestment.isnull().sum()


# In[185]:


# Get the distribution stats of columns value


# In[157]:


catCols = [expoGlobe, numResearchPub,timeMarketSrvOrProd, companyAwards, survivalRecession]
len(catCols)


# In[158]:


numCols = [googlePageRank, numDirCompetitors, employeePerYear, lastRndFndingAmtMilUSD, avgTimeInvestment]
len(numCols)


# In[171]:


catColsPD = pd.concat(catCols, axis=1)
catColsPD


# In[172]:


numColsPD = pd.concat(numCols, axis=1)
numColsPD


# In[182]:


catColsDesc = catColsPD.describe()
catColsDesc.T.style.background_gradient(cmap='viridis',low =0.2,high=0.1)


# In[363]:


numColsDesc = numColsPD.describe()
numColsDesc.T.style.background_gradient(cmap='viridis',low =0.2,high=0.1)


# In[ ]:





# In[184]:


#EDA


# In[188]:


font={'weight' : 'bold',
        'size': 16}


# In[189]:


plt.figure(figsize=(12,8))

ax=sns.countplot(expoGlobe,hue=companyStatus)
plt.title("Exposure across the globe",**font)
plt.ylabel("Number of companies")
plt.xticks(rotation=45)
for i in ax.patches:
    ax.text(i.get_x(), i.get_height()+2,             str(round((i.get_height()), 2)), fontsize=14, color='dimgrey')
plt.show()


# In[190]:


plt.figure(figsize=(12,8))

ax=sns.countplot(numResearchPub,hue=companyStatus)
plt.title("Number of of Research publications",**font)
plt.ylabel("Number of companies")
plt.xticks(rotation=45)
for i in ax.patches:
    ax.text(i.get_x(), i.get_height()+2,             str(round((i.get_height()), 2)), fontsize=14, color='dimgrey')
plt.show()


# In[191]:


plt.figure(figsize=(12,8))

ax=sns.countplot(timeMarketSrvOrProd,hue=companyStatus)
plt.title("Time to market service or product",**font)
plt.ylabel("Number of companies")
plt.xticks(rotation=45)
for i in ax.patches:
    ax.text(i.get_x(), i.get_height()+2,             str(round((i.get_height()), 2)), fontsize=14, color='dimgrey')
plt.show()


# In[192]:


plt.figure(figsize=(12,8))

ax=sns.countplot(companyAwards,hue=companyStatus)
plt.title("Company awards",**font)
plt.ylabel("Number of companies")
plt.xticks(rotation=45)
for i in ax.patches:
    ax.text(i.get_x(), i.get_height()+2,             str(round((i.get_height()), 2)), fontsize=14, color='dimgrey')
plt.show()


# In[205]:


plt.figure(figsize=(12,8))

ax=sns.countplot(survivalRecession,hue=companyStatus)
plt.title("Survival through recession",**font)
plt.ylabel("Number of companies")
plt.xticks(rotation=45)
for i in ax.patches:
    ax.text(i.get_x(), i.get_height()+2,             str(round((i.get_height()), 2)), fontsize=14, color='dimgrey')
plt.show()


# In[ ]:





# In[220]:


plt.figure(figsize=(12,8))

df = pd.concat([numColsPD, companyStatus], axis=1)
sns.FacetGrid(df,hue='Dependent-Company Status',height=10).map(sns.distplot,"google page rank of company website").add_legend()

plt.title("google page rank of company website distribution density plot",**font)
plt.show()


# In[218]:


plt.figure(figsize=(12,8))

df = pd.concat([numColsPD, companyStatus], axis=1)
sns.FacetGrid(df,hue='Dependent-Company Status',height=8).map(sns.distplot,"Number of Direct competitors").add_legend()

plt.title(" Number of Direct competitors distribution density plot",**font)
plt.show()


# In[225]:


plt.figure(figsize=(12,8))

df = pd.concat([numColsPD, companyStatus], axis=1)
sns.FacetGrid(df,hue='Dependent-Company Status',height=8).map(sns.distplot,"Employees per year of company existence").add_legend()

plt.title("Employees per year of company existence distribution density plot",**font)
plt.show()


# In[226]:


plt.figure(figsize=(12,8))

df = pd.concat([numColsPD, companyStatus], axis=1)
sns.FacetGrid(df,hue='Dependent-Company Status',height=10).map(sns.distplot,"Last round of funding received (in milionUSD)").add_legend()

plt.title("Last round of funding received (in milionUSD) distribution density plot",**font)
plt.show()


# In[230]:



plt.figure(figsize=(12,8))

df = pd.concat([numColsPD, companyStatus], axis=1)
sns.FacetGrid(df,hue='Dependent-Company Status',height=10).map(sns.distplot,"Avg time to investment - average across all rounds, measured from previous investment").add_legend()

plt.title("Avg time to investment distribution density plot",**font)
plt.show()


# In[ ]:





# In[241]:


plt.figure(figsize = (15, 4))
plt.subplot(1, 2, 1)
googlePageRank.hist(grid=False)
plt.ylabel('count')
plt.subplot(1, 2, 2)
sns.boxplot(x=googlePageRank)
plt.show()

print(f"google page rank data skew:{round(googlePageRank.skew(),2)}")


# In[244]:


plt.figure(figsize = (15, 4))
plt.subplot(1, 2, 1)
numDirCompetitors.hist(grid=False)
plt.ylabel('count')
plt.subplot(1, 2, 2)
sns.boxplot(x=numDirCompetitors)
plt.show()

print(f"Number of Direct competitors data skew: {round(numDirCompetitors.skew(), 2)}")


# In[245]:


plt.figure(figsize = (15, 4))
plt.subplot(1, 2, 1)
employeePerYear.hist(grid=False)
plt.ylabel('count')
plt.subplot(1, 2, 2)
sns.boxplot(x=employeePerYear)
plt.show()

print(f"Employees per year of company existence data skew: {round(employeePerYear.skew(), 2)}")


# In[246]:


plt.figure(figsize = (15, 4))
plt.subplot(1, 2, 1)
lastRndFndingAmtMilUSD.hist(grid=False)
plt.ylabel('count')
plt.subplot(1, 2, 2)
sns.boxplot(x=lastRndFndingAmtMilUSD)
plt.show()

print(f"Last round of funding received (in milionUSD) data skew: {round(lastRndFndingAmtMilUSD.skew(), 2)}")


# In[247]:


plt.figure(figsize = (15, 4))
plt.subplot(1, 2, 1)
avgTimeInvestment.hist(grid=False)
plt.ylabel('count')
plt.subplot(1, 2, 2)
sns.boxplot(x=avgTimeInvestment)
plt.show()

print(f"Avg time to investment - average across all rounds, measured from previous investment data skew: {round(avgTimeInvestment.skew(), 2)}")


# In[ ]:





# In[248]:


# Apply Turkey's rule to detect outliers
# IQR = Q3 - Q1
# Lower Boundary: Q1 - 1.5 * IQR
# Upper Boundary: Q3 + 1.5 * IQR


# In[249]:


numColsDesc = numColsPD.describe()
numColsDesc.T.style.background_gradient(cmap='viridis',low =0.2,high=0.1)


# In[258]:


q1Google = googlePageRank.quantile(0.25)
q1Google


# In[259]:


q3Google = googlePageRank.quantile(0.75)
q3Google


# In[260]:


IQR_google = q3Google - q1Google
IQR_google


# In[262]:


lowerBndGoogle = q1Google - 1.5 * IQR_google
lowerBndGoogle


# In[265]:


upperBndGoogle = q3Google + 1.5 * IQR_google
upperBndGoogle


# In[269]:


outliersLowerGoogle = googlePageRank[googlePageRank < lowerBndGoogle]
outliersLowerGoogle


# In[270]:


outliersUpperGoogle = googlePageRank[googlePageRank > upperBndGoogle]
outliersUpperGoogle


# In[271]:


len(googlePageRank)


# In[272]:


len(googlePageRank) - (len(outliersLowerGoogle) + len(outliersUpperGoogle))


# In[341]:


googlePageRankNoOutliers = googlePageRank[~((googlePageRank < lowerBndGoogle) | (googlePageRank > upperBndGoogle))]
googlePageRankNoOutliers


# In[277]:


len(googlePageRankNoOutliers)


# In[342]:


googNoOutPD = pd.concat([googlePageRank, companyStatus], axis = 1)
googNoOutPD = googNoOutPD[~((googlePageRank < lowerBndGoogle) | (googlePageRank > upperBndGoogle))]
googNoOutPD


# In[278]:


plt.figure(figsize = (15, 4))
plt.subplot(1, 2, 1)
googlePageRankNoOutliers.hist(grid=False)
plt.ylabel('count')
plt.subplot(1, 2, 2)
sns.boxplot(x=googlePageRankNoOutliers)
plt.show()

print(f"google page rank without outliers data skew:{round(googlePageRankNoOutliers.skew(),2)}")


# In[ ]:





# In[279]:


q1DirCompetitors = numDirCompetitors.quantile(0.25)
q1DirCompetitors


# In[280]:


q3DirCompetitors = numDirCompetitors.quantile(0.75)
q3DirCompetitors


# In[281]:


IQR_DirCompetitors = q3DirCompetitors - q1DirCompetitors
IQR_DirCompetitors


# In[282]:


lowerBndDirComp = q1DirCompetitors - 1.5 * IQR_DirCompetitors
lowerBndDirComp


# In[283]:


upperBndDirComp = q3DirCompetitors + 1.5 * IQR_DirCompetitors
upperBndDirComp


# In[284]:


outliersLowerDirComp = numDirCompetitors[numDirCompetitors < lowerBndDirComp]
outliersLowerDirComp


# In[285]:


outliersUpperDirComp = numDirCompetitors[numDirCompetitors > upperBndDirComp]
outliersUpperDirComp


# In[286]:


len(outliersUpperDirComp)


# In[303]:


len(numDirCompetitors) - (len(outliersLowerDirComp) + len(outliersUpperDirComp))


# In[291]:


numDirCompetitorsNoOutliers = numDirCompetitors[~((numDirCompetitors < lowerBndDirComp) | (numDirCompetitors > upperBndDirComp))]
numDirCompetitorsNoOutliers


# In[292]:


len(numDirCompetitorsNoOutliers)


# In[343]:


numDirCompNoOutPD = pd.concat([numDirCompetitors, companyStatus], axis=1)
numDirCompNoOutPD = numDirCompNoOutPD[~((numDirCompetitors < lowerBndDirComp) | (numDirCompetitors > upperBndDirComp))]
numDirCompNoOutPD


# In[304]:


plt.figure(figsize = (15, 4))
plt.subplot(1, 2, 1)
numDirCompetitorsNoOutliers.hist(grid=False)
plt.ylabel('count')
plt.subplot(1, 2, 2)
sns.boxplot(x=numDirCompetitorsNoOutliers)
plt.show()

print(f"Number of Direct competitors without outliers data skew: {round(numDirCompetitorsNoOutliers.skew(), 2)}")


# In[ ]:





# In[305]:


q1EmpPerYear = employeePerYear.quantile(0.25)
q1EmpPerYear


# In[306]:


q3EmpPerYear = employeePerYear.quantile(0.75)
q3EmpPerYear


# In[308]:


IQR_EmpPerYear = q3EmpPerYear - q1EmpPerYear
IQR_EmpPerYear


# In[309]:


lowerBndEmpPerYear = q1EmpPerYear - 1.5 * IQR_EmpPerYear
lowerBndEmpPerYear


# In[310]:


upperBndEmpPerYear = q3EmpPerYear + 1.5 * IQR_EmpPerYear
upperBndEmpPerYear


# In[311]:


outliersLowerBndEmpPerYear = employeePerYear[employeePerYear < lowerBndEmpPerYear]
outliersLowerBndEmpPerYear


# In[312]:


outlierUpperBndEmpPerYear = employeePerYear[employeePerYear > upperBndEmpPerYear]
outlierUpperBndEmpPerYear


# In[313]:


len(employeePerYear) - (len(outliersLowerBndEmpPerYear) + len(outlierUpperBndEmpPerYear))


# In[314]:


employeePerYearNoOutliers = employeePerYear[~((employeePerYear < lowerBndEmpPerYear) | (employeePerYear > upperBndEmpPerYear))]
employeePerYearNoOutliers


# In[315]:


len(employeePerYearNoOutliers)


# In[344]:


empPerYearNoOutPD = pd.concat([employeePerYear, companyStatus], axis=1)
empPerYearNoOutPD = empPerYearNoOutPD[~((employeePerYear < lowerBndEmpPerYear) | (employeePerYear > upperBndEmpPerYear))]
empPerYearNoOutPD


# In[316]:


plt.figure(figsize = (15, 4))
plt.subplot(1, 2, 1)
employeePerYearNoOutliers.hist(grid=False)
plt.ylabel('count')
plt.subplot(1, 2, 2)
sns.boxplot(x=employeePerYearNoOutliers)
plt.show()

print(f"Employees per year of company existence data skew: {round(employeePerYearNoOutliers.skew(), 2)}")


# In[ ]:





# In[317]:


q1LastRndFnding = lastRndFndingAmtMilUSD.quantile(0.25)
q1LastRndFnding


# In[318]:


q3LastRndFnding = lastRndFndingAmtMilUSD.quantile(0.75)
q3LastRndFnding


# In[319]:


IQR_LastRndFnding = q3LastRndFnding - q1LastRndFnding
IQR_LastRndFnding


# In[320]:


lowerBndLastRndFunding = q1LastRndFnding - 1.5 * IQR_LastRndFnding
lowerBndLastRndFunding


# In[321]:


upperBndLastRndFnding = q3LastRndFnding + 1.5 * IQR_LastRndFnding
upperBndLastRndFnding


# In[322]:


outliersLowerBndLastRndFnding = lastRndFndingAmtMilUSD[lastRndFndingAmtMilUSD < lowerBndLastRndFunding]
outliersLowerBndLastRndFnding


# In[323]:


outliersUpperBndLastRndFnding = lastRndFndingAmtMilUSD[lastRndFndingAmtMilUSD > upperBndLastRndFnding]
outliersUpperBndLastRndFnding


# In[326]:


len(lastRndFndingAmtMilUSD) - (len(outliersLowerBndLastRndFnding) + len(outliersUpperBndLastRndFnding))


# In[359]:


lastRndFndingAmtMilUSDNoOutliers = lastRndFndingAmtMilUSD[~((lastRndFndingAmtMilUSD < lowerBndLastRndFunding) | (lastRndFndingAmtMilUSD > upperBndLastRndFnding))]
lastRndFndingAmtMilUSDNoOutliers


# In[328]:


len(lastRndFndingAmtMilUSDNoOutliers)


# In[345]:


lastRndFndNoOutPD = pd.concat([lastRndFndingAmtMilUSD, companyStatus], axis = 1)
lastRndFndNoOutPD = lastRndFndNoOutPD[~((lastRndFndingAmtMilUSD < lowerBndLastRndFunding) | (lastRndFndingAmtMilUSD > upperBndLastRndFnding))]
lastRndFndNoOutPD


# In[329]:


plt.figure(figsize = (15, 4))
plt.subplot(1, 2, 1)
lastRndFndingAmtMilUSDNoOutliers.hist(grid=False)
plt.ylabel('count')
plt.subplot(1, 2, 2)
sns.boxplot(x=lastRndFndingAmtMilUSDNoOutliers)
plt.show()

print(f"Last round of funding received (in milionUSD) without outliers data skew: {round(lastRndFndingAmtMilUSDNoOutliers.skew(), 2)}")


# In[ ]:





# In[330]:


q1AvgTimeInvest = avgTimeInvestment.quantile(0.25)
q1AvgTimeInvest


# In[331]:


q3AvgTimeInvest = avgTimeInvestment.quantile(0.75)
q3AvgTimeInvest


# In[332]:


IQR_AvgTimeInvest = q3AvgTimeInvest - q1AvgTimeInvest
IQR_AvgTimeInvest


# In[333]:


lowerBndAvgTimeInvest = q1AvgTimeInvest - 1.5 * IQR_AvgTimeInvest
lowerBndAvgTimeInvest


# In[334]:


upperBndAvgTimeInvest = q3AvgTimeInvest + 1.5 * IQR_AvgTimeInvest
upperBndAvgTimeInvest


# In[335]:


outliersLowerBndAvgTimeInvest = avgTimeInvestment[avgTimeInvestment < lowerBndAvgTimeInvest]
outliersLowerBndAvgTimeInvest


# In[336]:


outliersUpperBndAvgTimeInvest = avgTimeInvestment[avgTimeInvestment > upperBndAvgTimeInvest]
outliersUpperBndAvgTimeInvest


# In[337]:


len(avgTimeInvestment) - (len(outliersLowerBndAvgTimeInvest) + len(outliersUpperBndAvgTimeInvest))


# In[338]:


avgTimeInvestmentNoOutliers = avgTimeInvestment[~((avgTimeInvestment < lowerBndAvgTimeInvest) | (avgTimeInvestment > upperBndAvgTimeInvest))]
avgTimeInvestmentNoOutliers


# In[339]:


len(avgTimeInvestmentNoOutliers)


# In[346]:


avgTimeInvNoOutPD = pd.concat([avgTimeInvestment, companyStatus], axis=1)
avgTimeInvNoOutPD = avgTimeInvNoOutPD[~((avgTimeInvestment < lowerBndAvgTimeInvest) | (avgTimeInvestment > upperBndAvgTimeInvest))]
avgTimeInvNoOutPD


# In[340]:


plt.figure(figsize = (15, 4))
plt.subplot(1, 2, 1)
avgTimeInvestmentNoOutliers.hist(grid=False)
plt.ylabel('count')
plt.subplot(1, 2, 2)
sns.boxplot(x=avgTimeInvestmentNoOutliers)
plt.show()

print(f"Avg time to investment - average across all rounds, measured from previous investment without outliers data skew: {round(avgTimeInvestmentNoOutliers.skew(), 2)}")


# In[ ]:





# In[ ]:





# In[347]:


plt.figure(figsize=(12,8))

sns.FacetGrid(googNoOutPD,hue='Dependent-Company Status',height=10).map(sns.distplot,"google page rank of company website").add_legend()

plt.title("google page rank of company no outliers website distribution density plot",**font)
plt.show()


# In[349]:


plt.figure(figsize=(12,8))

sns.FacetGrid(numDirCompNoOutPD,hue='Dependent-Company Status',height=8).map(sns.distplot,"Number of Direct competitors").add_legend()

plt.title(" Number of Direct competitors no outliers distribution density plot",**font)
plt.show()


# In[353]:


plt.figure(figsize=(12,8))

sns.FacetGrid(empPerYearNoOutPD,hue='Dependent-Company Status',height=8).map(sns.distplot,"Employees per year of company existence").add_legend()

plt.title("Employees per year of company existence no outliers distribution density plot",**font)
plt.show()


# In[350]:


plt.figure(figsize=(12,8))

sns.FacetGrid(lastRndFndNoOutPD,hue='Dependent-Company Status',height=10).map(sns.distplot,"Last round of funding received (in milionUSD)").add_legend()

plt.title("Last round of funding received (in milionUSD) no outliers distribution density plot",**font)
plt.show()


# In[352]:



plt.figure(figsize=(12,8))

sns.FacetGrid(avgTimeInvNoOutPD,hue='Dependent-Company Status',height=10).map(sns.distplot,"Avg time to investment - average across all rounds, measured from previous investment").add_legend()

plt.title("Avg time to investment distribution no outliers density plot",**font)
plt.show()


# In[361]:





# In[362]:


plt.figure(figsize=(12, 7))
df = pd.concat([googlePageRankNoOutliers, numDirCompetitorsNoOutliers, employeePerYearNoOutliers,                 lastRndFndingAmtMilUSDNoOutliers, avgTimeInvestmentNoOutliers, companyStatus], axis=1)
sns.heatmap(df.corr(), annot = True, vmin = -1, vmax = 1)
plt.show()


# In[ ]:




