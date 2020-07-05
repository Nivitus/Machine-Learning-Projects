# LOAN PRICE PREDICTION USING MACHINE LEARNING #
## Table of Content ##

[![Alt Text](Images/1.png)](https://medium.com/@Nivitus./loan-price-prediction-using-machine-learning-b585aafa3e7)

- [Overview](#overview)
- [Motivation](#motivation)
- [About the Dataset](#about-the-dataset)
- [About the Algorithms used in this Project](#about-the-algorithms-used-in-this-project)
- [Directory Tree Structure](#directory-tree-structure) 
- [Technologies Used](#technologies-used) 
- [Packages Used](#packages-used)
- [Data Collections](#data-collections)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Observation](#feature-observation)
- [Model Building](#model-building)
- [Model Performances](#model-performances)
- [Prediction and Final Score](#prediction-and-final-score)
- [Deployment](#deployment)
- [Team](#team)
- [Conclusion](#conclusion)


## Overview

In **finance**, a loan is the lending of money by one or more individuals, organizations, or other entities to other individuals, organizations, etc. The recipient (i.e. the borrower) incurs a debt, and is usually liable to pay interest on that debt until it is repaid, and also to repay the principal amount borrowed. To read more check out Wikipedia. The whole process of ascertaining if a burrower would pay back loans might be tedious hence the need to automate the procedure.

In this Project we are going to do implementing a scalable model for predicting a person got the eligible for getting loan or not. It’s a **Binary Classification** problem for predicting Yes or No results. There are some of the processing techniques for creating a model. We will see about it in upcoming parts

## Motivation

What could be a perfect way to utilize unfortunate lockdown period? Like most of, I spend my time Games, Movies, Coding and Writing Blogs and read about some upcoming AI articles. Now a day’s most of the people would like to apply the loan and could wish to get money from the bank for various purposes or any other personal uses. That’s what I came up with an idea about **Loan Predicting** system model.


## About the Dataset

In this Dataset made for predicting **binary classification** problem. Here all of the features represent unique client information like Loan **ID, Name, Income, and Loan Amount and so on.** We will see about it.

![](Images/12.JPG)

### Code for Checking name of the Features / Columns in the Dataset

``` python
TODO: Checking the name of the columns / Features

df.columns
```
### Output for Name of the Features /Columns in the Dataset

``` python

Index(['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status'],
      dtype='object')
```
### Let’s know more about the data features

``` python
TODO: View the data Information into the dataset
df.info()
```

``` python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 614 entries, 0 to 613
Data columns (total 13 columns):
Loan_ID              614 non-null object
Gender               601 non-null object
Married              611 non-null object
Dependents           599 non-null object
Education            614 non-null object
Self_Employed        582 non-null object
ApplicantIncome      614 non-null int64
CoapplicantIncome    614 non-null float64
LoanAmount           592 non-null float64
Loan_Amount_Term     600 non-null float64
Credit_History       564 non-null float64
Property_Area        614 non-null object
Loan_Status          614 non-null object
dtypes: float64(4), int64(1), object(8)
memory usage: 62.4+ KB
```
## About the Algorithms used in this Project

The major aim of this project is to predict which of the customers will have their loan paid or not. Therefore, this is a **supervised classification** problem to be trained with algorithms like:

**1. Logistic Regression
**2. Decision Tree
**3. Random Forest


## Directory Tree Structure


## Technologies Used

[![Alt Text](Images/19.JPG)](https://www.python.org/)

## Packages Used

[![Alt Text](Images/14.png)](https://numpy.org/doc/)  [![Alt Text](Images/15.png)](https://pandas.pydata.org/)    

[![Alt Text](Images/16.png)](https://seaborn.pydata.org/)  [![Alt Text](Images/17.jpg)](https://matplotlib.org/)
  
[![Alt Text](Images/00.JPG)](https://scikit-learn.org/stable/)


## Data Collections

I got the Dataset from Kaggle.If you wanna get this dataset just [Click here](https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset) This Dataset consist several features such as Name, Loan ID, and Loan Amount and so on. Let’s know about how to read the dataset into the Jupyter Notebook. You can download the dataset from Kaggle in csv file format.

#### Read the dataset for sample viewing

``` python
TODO: Read the Dataset using Pandas CSV format

import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv("train.csv") 
df.head()
```

#### Display the dataset in Dataframe

![](Images/13.JPG)

## Data Preprocessing

When I download the dataset from **[Kaggle](https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset)** I just looked up the dataset after I understand i need to do some of the data preprocessing techniques I must to do. Here I mentioned the Techniques which are really help full for you know about the Art of Data Preprocessing and [Data Cleaning.](https://towardsdatascience.com/the-art-of-cleaning-your-data-b713dbd49726)

- [1.Handling Null / Missing Values.](https://analyticsindiamag.com/5-ways-handle-missing-values-machine-learning-datasets/)
- [2.Handling the categorical Values.](https://www.pluralsight.com/guides/handling-categorical-data-in-machine-learning-models)
- [3.Using get_dummies Method (Feature Engineering).](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html)

### Handling Null / Missing Values

Handling the missing values is one of the greatest challenges faced by analysts, because making the right decision on how to handle it.
Before that we should see on what are the features has contains the null or missing values into the dataset.

``` python
TODO: # Count the sum of the missing values in each of the features

df.isnull().sum()

Loan_ID               0
Gender               13
Married               3
Dependents           15
Education             0
Self_Employed        32
ApplicantIncome       0
CoapplicantIncome     0
LoanAmount           22
Loan_Amount_Term     14
Credit_History       50
Property_Area         0
Loan_Status           0
dtype: int64
```

### Here we can able to see the null or missing values in visual format using heatmap

![](Images/2.JPG)

#### Here I think almost six features have missing values. That’s ok, don’t panic let’s we’ll see about how to fill up the null or missing values in this dataset.

### Handling the Null / Missing Values in Numerical Values

``` python
TODO: # Using fillna methods to fillup the maean of the features

df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())
```

``` python
TODO: Count the sum of the missing values in each of the features
df.isnull().sum()

Loan_ID               0
Gender               13
Married               3
Dependents           15
Education             0
Self_Employed        32
ApplicantIncome       0
CoapplicantIncome     0
LoanAmount            0
Loan_Amount_Term      0
Credit_History        0
Property_Area         0
Loan_Status           0
dtype: int64
```



## Exploratory Data Analysis



## Feature Observation


## Model Building


## Model Performances


## Prediction and Final Score


## Deployment


## Team


## Conclusion
