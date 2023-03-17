# ECE 142 Final Project (Team #11)

## Problem Statement

Analyzing data relating to a startup allows for us to be able to better understand which factors
cause a startup to fail or succeed.         

The analysis of a startup involves evaluating various aspects of the company to determine its
potential for success and investment value. There are several factors that influence the
company’s success such as business model, market opportunity, management team,
implementation of startup principles, leadership, etc.      

The goal of this analysis is to help investors, entrepreneurs and stakeholders to make informed decisions about potential for success
and return on investment for a startup company.

## Project File Structure

* **data**: contains the raw dataset      
* **doc**: contains the group presentation and project dependencies requirement.      
* **notebook**: contains the Jupyter Notebook from each team memeber's contribution
* **src**: contains the Python file with respect to the Jupyter Notebook

## Dataset

Startup Analysis Dataset (https://www.kaggle.com/datasets/ajaygorkar/startup-analysis).     

The dataset consists of a .csv file, which contains various information about 472 startups like
Success, Industry of company, Country of Origin, Experience of founders etc.

## Data Extraction

This work is accomplished by doing data analysis on a dataset of 472 different companies.
With the 115 different attributes that are listed for these companies, useful information can be
discovered which will draw conclusions on what makes a startup successful.  

In the presentiation, we illustrate analysis of startup success based on:
* Geography
* Business Strategy and Product Management
* Founders Previous Experience
* Exposure across Globe
* Time to Market Service or Product
* Google Page Rank of Company Website
* Number of Direct Competitors 
* Last Round of Funding Received

Due to the limited time of presentation, not all attribute analysis is included in the presentation slides. The full attribute analysis could be found in the Jupyter Notebooks.

## Data Cleaning

We count the number of missing values for each attribute. If the number of missing values is greater than 30% of the total number of values for that attribute, we drop this attribute. If the number of missing values is fewer than 30%, we replace the missing values with the mode for categorical atrribute and mean for numerical atrribute.

We use boxplot to visualize data skew and calculate the skew magnitude. After identifying which attribute data has high data skew, we utilize the quartile info for handling outliers. We set the lower and up bound of valid data by using Inter Quartile Range between Q3 and Q1 as a filter.

## Data Analysis & Visualization

We use Pie Chart, Bar Plot, Density Distribution Plot, and Multi-variate Heatmap. 










