### 1. Importing Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import datasets
```
- **`import`:** Keyword to bring in external functionality.
- **`pandas`:** Library for data manipulation and analysis.
- **`numpy`:** Library for numerical operations on arrays and matrices.
- **`matplotlib.pyplot`:** Library for creating static, animated, and interactive visualizations.
- **`%matplotlib inline`:** Magic command for inline plotting in Jupyter Notebook.
- **`sklearn`:** Scikit-learn library, a machine learning library in Python.
- **`datasets`:** Module in scikit-learn for loading datasets.

### 2. Inserting Datasets
```python
hi = pd.read_csv("iris.csv")
```
- **`pd.read_csv`:** Function from Pandas to read a CSV (Comma-Separated Values) file and create a DataFrame.
- **`"iris.csv"`:** File path or URL of the dataset to be read.
- **`hi`:** Variable assigned to the DataFrame created from the Iris dataset.

### 3. Head of Data Set
```python
hi.head()
```
- **`hi.head()`:** Method to display the first few rows of the DataFrame 'hi'.

### 4. Shape of Dataset
```python
hi.shape
```
- **`hi.shape`:** Attribute of the DataFrame to get the dimensions (number of rows and columns) of the dataset.

### 5. Split the Datasets
```python
from sklearn.model_selection import train_test_split
x = hi.drop("Species", axis=1)
x = x.drop("Id", axis=1)
y = hi['Species']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
```
- **`from sklearn.model_selection import train_test_split`:** Importing the `train_test_split` function from scikit-learn to split the dataset.
- **`hi.drop("Species", axis=1)`:** Removing the 'Species' column from the features (input variables).
- **`hi['Species']`:** Selecting the 'Species' column as the target variable (output).
- **`train_test_split(x, y, test_size=0.30)`:** Splitting the dataset into training and testing sets. `test_size=0.30` means 30% of the data will be used for testing.

### 6. Import Library for KNN Algorithm
```python
from sklearn.neighbors import KNeighborsClassifier
```
- **`from sklearn.neighbors import KNeighborsClassifier`:** Importing the `KNeighborsClassifier` class from scikit-learn, which implements the K-Nearest Neighbors algorithm.

### 7. Selection of Hyperparameter k
```python
knn = KNeighborsClassifier(6)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)
```
- **`KNeighborsClassifier(6)`:** Creating a KNN classifier with 'k' (number of neighbors) set to 6.
- **`knn.fit(x_train, y_train)`:** Training the KNN model on the training data.
- **`knn.predict(x_test)`:** Making predictions on the test set.

### 8. Error Rate Calculation
```python
error_rate = []
for i in range(2, 50):
    knn = KNeighborsClassifier(i)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    error_rate.append(np.mean(y_test != pred))
plt.plot(range(2, 50), error_rate)
```
- **`error_rate = []`:** Initializing an empty list to store error rates.
- **`for i in range(2, 50):`:** Looping through values of 'k' from 2 to 49.
- **`KNeighborsClassifier(i)`:** Creating KNN models with varying 'k'.
- **`np.mean(y_test != pred)`:** Calculating the error rate (percentage of incorrect predictions).
- **`plt.plot(range(2, 50), error_rate)`:** Plotting the error rates for different 'k' values.

### 9. Evaluation Metrics
```python
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))
```
- **`from sklearn.metrics import confusion_matrix, classification_report, accuracy_score`:** Importing evaluation metrics from scikit-learn.
- **`confusion_matrix(y_test, pred)`:** Generating a confusion matrix.
- **`classification_report(y_test, pred)`:** Displaying a detailed classification report.
- **`accuracy_score(y_test, pred)`:** Calculating the accuracy score.

This breakdown covers the keywords and steps involved in the Iris Flower Classification project, implementing the K-Nearest Neighbors algorithm.