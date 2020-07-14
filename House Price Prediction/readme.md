<p align = "center"><h1><b> HOUSE PRICE PREDCTION USING MACHINE LEARNING ALGORITHM </b></h1> </p> </br>

[![Alt Text](Images/12.jpg)](https://medium.com/@Nivitus./boston-house-price-prediction-using-machine-learning-ad3750a866cd)

## Table of Content ##

- [Overview](#overview)
- [Motivation](#motivation)
- [Understand the Problem Statement](#understand-the-problem-statement)
- [About the Dataset](#about-the-dataset)
- [About the Algorithms used in this Project](#about-the-algorithms-used-in-this-project)
- [Directory Tree Structure](#directory-tree-structure) 
- [Technologies Used](#technologies-used) 
- [Packages Used](#packages-used)
- [Data Collections](#data-collections)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Observation](#feature-observation)
- [Feature Selection](#feature-selection)
- [Model Building](#model-building)
- [Model Performances](#model-performances)
- [Prediction and Final Score](#prediction-and-final-score)
- [Deployment](#deployment)
- [Team](#team)
- [Conclusion](#conclusion)

## Overview

So far so good, today we are going to work on a dataset which consists information about the location of the house, price and other aspects such as square feet etc. When we work on these sorts of data, we need to see which column is important for us and which is not. Our main aim today is to make a model which can give us a good prediction on the price of the house based on other variables. We are going to use Linear Regression for this dataset and see if it gives us a good accuracy or not.

In this Blog we are going to do implementing a salable model for predicting the house price prediction using some of the regression techniques based of some of features in the dataset which is called Boston House Price Prediction. There are some of the processing techniques for creating a model. We will see about it in upcoming parts …

## Motivation

The Motivation behind it I just wanna know about the house prices in California as well as I’ve an idea about to do some of the useful things do in the lock down period. I think this is a limited motivation for doing this blog well.

## Understand the Problem Statement

![](Images/13.png)

Housing prices are an important reflection of the economy, and housing price ranges are of great interest for both buyers and sellers. Ask a home buyer to describe their dream house, and they probably won’t begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition’s data-set proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

## About the Dataset

Housing prices are an important reflection of the economy, and housing price ranges are of great interest for both buyers and sellers. In this project, house prices will be predicted given explanatory variables that cover many aspects of residential houses. The goal of this project is to create a regression model that is able to accurately estimate the price of the house given the features.
In this dataset made for predicting the Boston House Price Prediction. Here I just show the all of the feature for each house separately. Such as Number of Rooms, Crime rate of the House’s Area and so on. We’ll show in the upcoming part.

### Data Overview

![](Images/20.JPG)

``` python
1. CRIM per capital crime rate by town
2. ZN proportion of residential land zoned for lots over 25,000 sq.ft.
3. INDUS proportion of non-retail business acres per town
4. CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
5. NOX nitric oxides concentration (parts per 10 million)
6. RM average number of rooms per dwelling
7. AGE proportion of owner-occupied units built prior to 1940
8. DIS weighted distances to five Boston employment centers
9. RAD index of accessibility to radial highways
10.TAX full-value property-tax rate per 10,000 USD
11.PTRATIO pupil-teacher ratio by town
12.Black 1000(Bk — 0.63)² where Bk is the proportion of blacks by town
13.LSTAT % lower status of the population
```
## About the Algorithms used in this Project

The major aim of in this project is to predict the house prices based on the features using some of the regression techniques and algorithms.

### 1. Linear Regression
### 2. Random Forest Regressor

## Directory Tree Structure

## Technologies Used

[![Alt Text](Images/19.JPG)](https://www.python.org/)

## Packages Used

[![Alt Text](Images/14.png)](https://numpy.org/doc/)  [![Alt Text](Images/15.png)](https://pandas.pydata.org/)    

[![Alt Text](Images/16.png)](https://seaborn.pydata.org/)  [![Alt Text](Images/17.jpg)](https://matplotlib.org/)
  
[![Alt Text](Images/00.JPG)](https://scikit-learn.org/stable/)

## Data Collections

I got the Dataset from [Kaggle](https://www.kaggle.com/prasadperera/the-boston-housing-dataset). This Dataset consist several features such as Number of Rooms, Crime Rate, and Tax and so on. Let’s know about how to read the dataset into the Jupyter Notebook. You can download the dataset from Kaggle in csv file format.Yup! you wanna get dataset from kaggle just [click here](https://www.kaggle.com/prasadperera/the-boston-housing-dataset)

As well we can also able to get the dataset from the sklearn datasets. Yup! It’s available into the sklearn Dataset just [click here](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html) and get it

### Let’s we see how can we retrieve the dataset from the sklearn dataset.

``` python
from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)
```

### Code for collecting data from CSV file into Jupyter Notebook!

``` python
# Import libraries
import numpy as np
import pandas as pd
# Import the dataset
df = pd.read_csv(“train.csv”)
df.head()
```
![](Images/20.JPG)



## Data Preprocessing

## Exploratory Data Analysis

## Feature Observation

## Feature Selection

## Model Building

## Model Performances

## Prediction and Final Score

## Deployment

## Team

## Conclusion
