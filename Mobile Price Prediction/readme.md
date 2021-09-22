# Mobile Price Prediction Using Machine Learning #
## Table of Content ##
- [Overview](#overview) 
- [Motivation](#motivation) 
- [Installation](#installation)
- [Understand the Problem Statement](#understand-the-problem-statement) 
- [About the Data](#about-the-data) 
- [About the Web Scrapping](#about-the-web-scrapping) 
- [Algorithms and Technologies Used](#algorithms-and-technologies-used) 
- [Packages Used](#packages-used)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing) 
- [Exploratory Data Analysis](#exploratory-data-analysis) 
- [About Data Features](#about-data-features)
- [Model Fitting](#model-fitting) 
- [Accuracy and Prediction Score](#accuracy-and-prediction-score)
- [Team](#team) 

## Overview 
We are going to do implementing a salable model for predicting the mobile price prediction using some of the regression techniques based of some of features in the dataset which is called mobile Price Prediction. There are some of the processing techniques for creating a model. In this project i used web scrapping techniques for collecting the mobile data from E-Commerce website. We will see about it in upcoming parts.

## Motivation 
The Motivation behind it I just wanna know about the various kinds of mobile prices during the lock down period. Because now a days most of the E-Commerce website are focusing to sell the mobile for consumers.

Because now a days many of the students including me also having the online class rooms for continue our education systems. So I got the idea about to do some of the useful things do in the lock down period. That’s why I decided to doing in this project. As well as one of my brother asked to me “Bro why shouldn’t we do this mobile price prediction from end to end? Like we are not going to do get the data from Kaggle for this project” .So I decided to make in this way.

## Installation 
The Code is written in Python 3.7. If you don't have Python installed just [clik here](https://www.python.org/downloads/) and install python on your system. 
If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, such as numpy, pandas, matplotlib and sklearn.

## Understand the Problem Statement
Mobile prices are an important reflection of the Humans and some ranges are of great interest for both buyers and sellers. Ask a mobile buyer to describe their dream Mobile or Branded Mobile Phones. So in this blog we are going to see about how the prices are segregated based on the some of the features. As well as the target feature prediction based on the same features.

## About the Data 
In this dataset I wasn’t downloading from Kaggle or any other data collecting websites. I just make or create the dataset using one of the web scrapping tools. I’ll tell about next upcoming part. So a little bit of overview we understand about the data and its features.

``` bash
# lets understand features of this dataset
df.columns
Index([‘Brand me’, ‘Ratings’, ‘RAM’, ‘ROM’, ‘Mobile_Size’, ‘Primary_Cam’,
‘Selfi_Cam’, ‘Battery_Power’, ‘Price’],
dtype=’object’)

```

## Data Overview




## Technologies Used 

[![Alt Text](Images/10.JPG)](https://www.python.org/)

## Packages Used 

  [![Alt Text](Images/12.png)](https://numpy.org/doc/)  [![Alt Text](Images/11.png)](https://pandas.pydata.org/)    

  [![Alt Text](Images/13.png)](https://seaborn.pydata.org/)  [![Alt Text](Images/15.jpg)](https://matplotlib.org/)
  
  [![Alt Text](Images/sk.JPG)](https://scikit-learn.org/stable/)

## Data Collection 

* I got the dataset from Kaggle if you wanaa get it [click here](https://www.kaggle.com/arshid/iris-flower-dataset).
* This dataset consists of 3 categories of species which is setosa, versicolor and virginica.
* We can find two kind of data from kaggle which is CSV data and SQLITE database.
* Each iris species consists of 50 samples.
* The features of iris flower are Sepal Length in cm, Sepal Width in cm, Petal Length in cm and Petal Width in cm.
* In this Iris dataset in avilable on **[sklearn](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)** library.
### The following code can help to get the Iris dataset from sklearn 

```python
# TODO : Load Iris Dataset

# Load iris Dataset from sklearn
from sklearn import datasets
iris = datasets.load_iris()

# Load Iris csv dataset
iris_csv = pd.read_csv('../data/Iris.csv')

# Load Iris sqlite data
data = sqlite3.connect('../data/database.sqlite')
query = data.execute('SELECT * FROM Iris')
columns = [col[0] for col in query.description]
iris_db = pd.DataFrame.from_records(data = query.fetchall(), columns = columns)
```

## Data Cleaning 

We need not to cleaning the data for making the machine learning model. Because we were retrive the data from kaggle and sklearn, is already have the format of csv(Comma Separated File) It's already getting clean data. So We don't put the stuffs for data cleaning. If you wanna more about the Art of Data Cleaning Process in Machine Learning 
just [Click Here](https://towardsdatascience.com/the-art-of-cleaning-your-data-b713dbd49726).

## Exploratory Data Analysis 

In statistics, exploratory data analysis (EDA) is an approach to analyzing data sets to summarize their main characteristics, often with visual methods.A statistical model can be used or not, but primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task. Here we performed some of the EDA Process to help the audience for visualize the Iris Flower data.

#### Letz Understand the correlation concepts between the data features using heatmap
```python
# TODO : Correlation between the data features
cor = df.drop("Species", axis=1).corr()
sns.heatmap(data=cor,annot = True,cmap="YlGnBu")
```
![](Images/22.JPG)

#### Next we would Understand the data features in Species Column using pairplot
```python
# TODO : Species data features
sns.pairplot(df,hue='Species')
```
![](Images/6.JPG)

#### Let's understand how the Petal length and Petal width contributes together to classify iris species.
```python
# TODO : Sepal and Petal data features
sns.relplot(x = 'PetalLengthCm', y = 'PetalWidthCm', data = df, hue = 'Species', aspect =1.5, height = 4)
```

![](Images/00.JPG)

#### Let's understand how the null values are already cleared and also visualize using the heatmap
```python
# TODO : Checking the null values into the data features
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
```
![](Images/7.JPG)

**There is no null values into the dataset 

#### Here we count the each speices values using Boxplot
```python
# TODO : Checking the number of the each species values into the Iris dataset
sns.set_style('whitegrid')
sns.countplot(x='Species',data=df)
```
![](Images/3.JPG)

## Model Fitting 
``` python
TODO: Assigning the values for model fitting
X = df.iloc[:,[1,2,3,4]]
y = df.iloc[:,[-1]]

```
```python
# TODO: Train Test Split and Build and Train the model

Since our process involves training and testing, we should split our dataset. It can be executed by the following code.

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
```
``` python
TODO: Using SVM Classifier

from sklearn.svm import SVC

svm = SVC()

svm.fit(X_train,y_train)

```
``` python
TODO: Prediciting
pred = svm.predict(X_test)
```

## Accuracy and Prediction Score 

#### Using Confusion Matrix for Classification prediction

``` python
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
```
#### Confusion Matrix Result
``` python 
[[16  0  0]
 [ 0 17  1]
 [ 0  0 11]]
 
 ```
 #### Using Classification Reports for prediction 
 
 ``` python
 print(classification_report(y_test,pred))
 ```
 
 #### Classification Report Results
 ``` python
                  precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        16
Iris-versicolor       1.00      0.94      0.97        18
 Iris-virginica       0.92      1.00      0.96        11

       accuracy                           0.98        45
      macro avg       0.97      0.98      0.98        45
   weighted avg       0.98      0.98      0.98        45

```
#### Making Sample Prediction

``` python
svm.predict([[5.3,3.4,2.7,3.9]])
```

#### Output for Sample Prediction

``` python
array(['Iris-virginica'], dtype=object)
```

### Prediction Score

``` python
print("Accuracy Score:",svm.score(X_test,y_test) * 100)
```
### Accuracy Score: 97.77777777777777

## Deployment 


## Team

![](Images/Niv.JPG)





