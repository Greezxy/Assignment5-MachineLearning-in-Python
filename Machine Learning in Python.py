#!/usr/bin/env python
# coding: utf-8

# # Assignment 5 – Machine Learning in Python
# 
# **Author**: David  
# **Date**: July 24, 2025  
# **Environment**: See `environment.yml`  
# **Description**:  
# This notebook explores basic machine learning using `scikit-learn`, including data loading, exploratory data analysis, model training, evaluation, and prediction using the Iris dataset.  
# Adapted from Jason Brownlee’s tutorial: [Machine Learning in Python Step-by-Step](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/)

# ## 1. Check the Versions of Libraries  
# It is a good idea to make sure your Python environment was installed successfully and is working as expected.  
# ### Python version

# In[1]:


import sys
print('Python: {}'.format(sys.version))


# ### scipy

# In[4]:


import scipy
print('scipy: {}'.format(scipy.__version__))


# ### numpy

# In[5]:


import numpy
print('numpy: {}'.format(numpy.__version__))


# ### matplotlib

# In[6]:


import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))


# ### pandas

# In[7]:


import pandas
print('pandas: {}'.format(pandas.__version__))


# ### scikit-learn

# In[8]:


import sklearn
print('sklearn: {}'.format(sklearn.__version__))


# ## 2. Load the Data
# We are going to use the iris flowers dataset. This dataset is famous because it is used as the “hello world” dataset in machine learning and statistics by pretty much everyone.  
# 
# The dataset contains 150 observations of iris flowers. There are four columns of measurements of the flowers in centimeters. The fifth column is the species of the flower observed. All observed flowers belong to one of three species.

# ### 2.1 Import Libraries
# First, let’s import all of the modules, functions and objects we are going to use in this tutorial.

# In[24]:


# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# ### 2.2 Load Dataset
# We are using pandas to load the data. We will also use pandas next to explore the data both with descriptive statistics and data visualization.
# 
# Note that we are specifying the names of each column when loading the data. This will help later when we explore the data.

# In[23]:


# Load dataset
url = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)


# ## 3. Summarize the Dataset
# In this step we are going to take a look at the data a few different ways:  
# 1. Dimensions of the dataset.
# 2. Peek at the data itself.
# 3. Statistical summary of all attributes.
# 4. Breakdown of the data by the class variable.

# ### 3.1 Dimensions of Dataset
# We can get a quick idea of how many instances (rows) and how many attributes (columns) the data contains with the shape property.

# In[22]:


# shape
print(dataset.shape)


# ### 3.2 Peek at the Data
# It is also always a good idea to actually eyeball your data.

# In[21]:


# head
print(dataset.head(20))


# ### 3.3 Statistical Summary
# Now we can take a look at a summary of each attribute.  
# This includes the count, mean, the min and max values as well as some percentiles.

# In[20]:


# descriptions
print(dataset.describe())


# We can see that all of the numerical values have the same scale (centimeters) and similar ranges between 0 and 8 centimeters.

# ### 3.4 Class Distribution
# Let’s now take a look at the number of instances (rows) that belong to each class. We can view this as an absolute count.

# In[19]:


# class distribution
print(dataset.groupby('class').size())


# We can see that each class has the same number of instances (50 or 33% of the dataset).

# ### 3.5 Complete Example
# For reference, we can tie all of the previous elements together into a single cell.
# 
# The complete example is listed below.

# In[17]:


# summarize the data
from pandas import read_csv
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
# shape
print(dataset.shape)
# head
print(dataset.head(20))
# descriptions
print(dataset.describe())
# class distribution
print(dataset.groupby('class').size())


# ## 4. Data Visualization
# We now have a basic idea about the data. We need to extend that with some visualizations.
# 
# We are going to look at two types of plots:  
# 1. Univariate plots to better understand each attribute.
# 2. Multivariate plots to better understand the relationships between attributes.

# ### 4.1 Univariate Plots
# We start with some univariate plots, that is, plots of each individual variable.
# 
# Given that the input variables are numeric, we can create box and whisker plots of each.
# 
# This gives us a much clearer idea of the distribution of the input attributes:

# In[18]:


# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()


# We can also create a histogram of each input variable to get an idea of the distribution.

# In[25]:


# histograms
dataset.hist()
pyplot.show()


# It looks like perhaps two of the input variables have a Gaussian distribution. This is useful to note as we can use algorithms that can exploit this assumption.

# ### 4.2 Multivariate Plots
# Now we can look at the interactions between the variables.
# 
# First, let’s look at scatterplots of all pairs of attributes. This can be helpful to spot structured relationships between input variables.

# In[26]:


# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()


# Note the diagonal grouping of some pairs of attributes. This suggests a high correlation and a predictable relationship.

# ### 4.3 Complete Example
# For reference, we can tie all of the previous elements together into a single script.
# 
# The complete example is listed below.

# In[27]:


# visualize the data
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# histograms
dataset.hist()
pyplot.show()

# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()


# ## 5. Evaluate Some Algorithms
# Now it is time to create some models of the data and estimate their accuracy on unseen data.
# 
# Here is what we are going to cover in this step:  
# 1. Separate out a validation dataset.
# 2. Set-up the test harness to use 10-fold cross validation.
# 3. Build multiple different models to predict species from flower measurements
# 4. Select the best model.

# ### 5.1 Create a Validation Dataset
# We need to know that the model we created is good.
# 
# Later, we will use statistical methods to estimate the accuracy of the models that we create on unseen data. We also want a more concrete estimate of the accuracy of the best model on unseen data by evaluating it on actual unseen data.
# 
# That is, we are going to hold back some data that the algorithms will not get to see and we will use this data to get a second and independent idea of how accurate the best model might actually be.
# 
# We will split the loaded dataset into two, 80% of which we will use to train, evaluate and select among our models, and 20% that we will hold back as a validation dataset.

# In[28]:


# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


# You now have training data in the X_train and Y_train for preparing models and a X_validation and Y_validation sets that we can use later.

# ### 5.2 Test Harness
# We will use stratified 10-fold cross validation to estimate model accuracy.
# 
# This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits.
# 
# Stratified means that each fold or split of the dataset will aim to have the same distribution of example by class as exist in the whole training dataset.
# 
# We set the random seed via the random_state argument to a fixed number to ensure that each algorithm is evaluated on the same splits of the training dataset.
# 
# We are using the metric of ‘accuracy‘ to evaluate models.
# 
# This is a ratio of the number of correctly predicted instances divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate). We will be using the scoring variable when we run build and evaluate each model next.

# ### 5.3 Build Models
# We don’t know which algorithms would be good on this problem or what configurations to use.
# 
# We get an idea from the plots that some of the classes are partially linearly separable in some dimensions, so we are expecting generally good results.
# 
# Let’s test 6 different algorithms:  
# - Logistic Regression (LR)
# - Linear Discriminant Analysis (LDA)
# - K-Nearest Neighbors (KNN)
# - Classification and Regression Trees (CART)
# - Gaussian Naive Bayes (NB)
# - Support Vector Machines (SVM)
# 
# This is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms.
# 
# Let’s build and evaluate our models:

# In[33]:


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='lbfgs', max_iter=200)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# ### 5.4 Select Best Model
# We now have 6 models and accuracy estimations for each. We need to compare the models to each other and select the most accurate.
# 
# Running the example above, we get the following raw results:  
# 1. LR: 0.966667 (0.040825)
# 2. LDA: 0.975000 (0.038188)
# 3. KNN: 0.958333 (0.041667)
# 4. CART: 0.950000 (0.040825)
# 5. NB: 0.950000 (0.055277)
# 6. SVM: 0.983333 (0.033333)
# 
# #### What scores did you get?
# In this case, we can see that it looks like Support Vector Machines (SVM) has the largest estimated accuracy score at about 0.9833 or 98.33%.
# 
# We can also create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. There is a population of accuracy measures for each algorithm because each algorithm was evaluated 10 times (via 10 fold-cross validation).
# 
# A useful way to compare the samples of results for each algorithm is to create a box and whisker plot for each distribution and compare the distributions.

# In[35]:


# Compare Algorithms
pyplot.boxplot(results, tick_labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()


# ### 5.5 Complete Example
# For reference, we can tie all of the previous elements together into a single script.
# 
# The complete example is listed below.

# In[36]:


# compare algorithms
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='lbfgs', max_iter=200)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
pyplot.boxplot(results, tick_labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()


# ## 6. Make Predictions
# We must choose an algorithm to use to make predictions.
# 
# The results in the previous section suggest that the SVM was perhaps the most accurate model. We will use this model as our final model.
# 
# Now we want to get an idea of the accuracy of the model on our validation set.
# 
# This will give us an independent final check on the accuracy of the best model. It is valuable to keep a validation set just in case you made a slip during training, such as overfitting to the training set or a data leak. Both of these issues will result in an overly optimistic result.

# ### 6.1 Make Predictions
# We can fit the model on the entire training dataset and make predictions on the validation dataset.

# # Make predictions on validation dataset
# model = SVC(gamma='auto')
# model.fit(X_train, Y_train)
# predictions = model.predict(X_validation)

# ### 6.2 Evaluate Predictions
# We can evaluate the predictions by comparing them to the expected results in the validation set, then calculate classification accuracy, as well as a confusion matrix and a classification report.

# In[38]:


# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# We can see that the accuracy is 0.966 or about 96% on the hold out dataset.
# 
# The confusion matrix provides an indication of the errors made.
# 
# Finally, the classification report provides a breakdown of each class by precision, recall, f1-score and support showing excellent results (granted the validation dataset was small).

# ### 6.3 Complete Example
# For reference, we can tie all of the previous elements together into a single script.
# 
# The complete example is listed below.

# In[39]:


# make predictions
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# ## Summary
# In this post, we discovered step-by-step how to complete your first machine learning project in Python.
# 
# We discovered that completing a small end-to-end project from loading the data to making predictions is the best way to get familiar with a new platform.
