#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("CAX_Startup_Data.csv")

data.info

lastfundingamount = data['Last Funding Amount']
lastfundingamount

employeecount = data['Employee Count']
employeecount

employeecountMOM = data['Employees count MoM change']
employeecountMOM

cofoundersnum = data['Number of Co-founders']
cofoundersnum

advisorsnum = data['Number of of advisors']
advisorsnum

companys = data['Company_Name']
companys.unique()

companyStatus = data['Dependent-Company Status']
companyStatus

companyStatus.unique()

lastfundingamount.unique()

employeecount.unique()

employeecountMOM.unique()

cofoundersnum.unique()

advisorsnum.unique()

# now we need to clean the data 

companyStatus.replace({"No Info": np.nan}, inplace=True)
companyStatus.unique()

lastfundingamount.replace({"No Info": np.nan}, inplace=True)
lastfundingamount =lastfundingamount.astype(float)
lastfundingamount.unique()

employeecount.replace({"No Info": np.nan}, inplace=True)
employeecount =employeecount.astype(float)
employeecount.unique()

employeecountMOM.replace({"No Info": np.nan}, inplace=True)
employeecountMOM =employeecountMOM.astype(float)
employeecountMOM.unique()

cofoundersnum.replace({"No Info": np.nan}, inplace=True)
cofoundersnum = cofoundersnum.astype(float)
cofoundersnum.unique()

advisorsnum.replace({"No Info": np.nan}, inplace=True)
advisorsnum = advisorsnum.astype(float)
advisorsnum.unique()

nonnumericalcolumns = [companys, companyStatus]
len(nonnumericalcolumns)

numericalcolumns = [lastfundingamount, employeecount, employeecountMOM, cofoundersnum, advisorsnum]
len(numericalcolumns)

nonnumericalcolumnsplot = pd.concat(nonnumericalcolumns, axis=1)
nonnumericalcolumnsplot

nonnumericalcolumnsNull=pd.DataFrame(nonnumericalcolumnsplot.isnull().sum(),columns=["NaN Values"])
nonnumericalcolumnsNull["Missing Values (in%)"]=(nonnumericalcolumnsplot.isna().sum()/len(nonnumericalcolumnsplot)*100)
nonnumericalcolumnsNull.style.background_gradient(cmap='viridis',low =0.2,high=0.1)

numericalcolumnsplot = pd.concat(numericalcolumns, axis=1)
numericalcolumnsplot

numericalcolumnsNull=pd.DataFrame(numericalcolumnsplot.isnull().sum(),columns=["NaN Values"])
numericalcolumnsNull["Missing Values (in%)"]=(numericalcolumnsplot.isna().sum()/len(numericalcolumnsplot)*100)
numericalcolumnsNull.style.background_gradient(cmap='viridis',low =0.2,high=0.1)


#replace all the NaN values for the numerical categories

lastfundingamountMean = lastfundingamount.mean()
lastfundingamountMean


lastfundingamount = lastfundingamount.fillna(lastfundingamountMean)
lastfundingamount.isnull().sum()

employeecountMean = employeecount.mean()
employeecountMean

employeecount = employeecount.fillna(employeecountMean)
employeecount.isnull().sum()

employeecountMOMMean = employeecountMOM.mean()
employeecountMOMMean

employeecountMOM = employeecountMOM.fillna(employeecountMOMMean)
employeecountMOM.isnull().sum()

cofoundersnumMean = cofoundersnum.mean()
cofoundersnumMean

cofoundersnum = cofoundersnum.fillna(cofoundersnumMean)
cofoundersnum.isnull().sum()

advisorsnumMean = advisorsnum.mean()
advisorsnumMean

advisorsnum = advisorsnum.fillna(advisorsnumMean)
advisorsnum.isnull().sum()

nonnumericalcolumnsplot = pd.concat(nonnumericalcolumns, axis=1)
nonnumericalcolumnsplot

numericalcolumns = [lastfundingamount, employeecount, employeecountMOM, cofoundersnum, advisorsnum]
len(numericalcolumns)

numericalcolumnsplot = pd.concat(numericalcolumns, axis=1)
numericalcolumnsplot

numericalcolumnsinfo = numericalcolumnsplot.describe()
numericalcolumnsinfo.T.style.background_gradient(cmap='viridis',low =0.2,high=0.1)

font={'weight' : 'bold',
 'size': 18}

plt.figure(figsize=(18,8))
ax = sns.countplot(lastfundingamount,hue=companyStatus)
plt.title("Last Funding Amount",**font)
plt.ylabel("Amount (millions $)")
ax.axes.get_xaxis().set_visible(False)
plt.xlabel("Companys")
plt.show()

font={'weight' : 'bold',
 'size': 18}

plt.figure(figsize=(18,8))
ax = sns.countplot(employeecount,hue=companyStatus)
plt.title("Number of employees",**font)
plt.ylabel("Employee Count")
ax.axes.get_xaxis().set_visible(False)
plt.xlabel("Companys")
plt.show()

font={'weight' : 'bold',
 'size': 18}

plt.figure(figsize=(18,8))
ax = sns.countplot(employeecountMOM,hue=companyStatus)
plt.title("Number of MOM employees",**font)
plt.ylabel("MOM Employee Count")
ax.axes.get_xaxis().set_visible(False)
plt.xlabel("Companys")
plt.show()

font={'weight' : 'bold',
 'size': 18}

plt.figure(figsize=(18,8))
ax = sns.countplot(cofoundersnum,hue=companyStatus)
plt.title("Number of Cofounders",**font)
plt.ylabel("Employee Count")
ax.axes.get_xaxis().set_visible(False)
plt.xlabel("Companys")
plt.show()

font={'weight' : 'bold',
 'size': 18}

plt.figure(figsize=(18,8))
ax = sns.countplot(advisorsnum,hue=companyStatus)
plt.title("Number of Advisors",**font)
plt.ylabel("Employee Count")
ax.axes.get_xaxis().set_visible(False)
plt.xlabel("Companys")
plt.show()

df = pd.concat([numericalcolumnsplot, companyStatus], axis=1)
sns.FacetGrid(df,hue='Dependent-Company Status',height=10).map(sns.distplot,"Employee Count").add_legend()

plt.show()

df = pd.concat([numericalcolumnsplot, companyStatus], axis=1)
sns.FacetGrid(df,hue='Dependent-Company Status',height=10).map(sns.distplot,"Employees count MoM change").add_legend()

plt.show()

df = pd.concat([numericalcolumnsplot, companyStatus], axis=1)
sns.FacetGrid(df,hue='Dependent-Company Status',height=10).map(sns.distplot,"Number of of advisors").add_legend()

plt.show()


df = pd.concat([numericalcolumnsplot, companyStatus], axis=1)
sns.FacetGrid(df,hue='Dependent-Company Status',height=10).map(sns.distplot,"Number of Co-founders").add_legend()

plt.show()
