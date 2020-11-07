# __Machine Learning Algorithms__

<img src='https://github.com/mohd-faizy/__Machine_Learning_Algorithms__/blob/master/Algorithms_png/Head_ML.png'>

## __01-Linear Regression:__

```python

# Import Library
# Import other necessary libraries like panda, numpy...

from sklearn import linear_model

# Load Train and Test datasets
# Identify feature and response variable(s) and 
# values must be numeric and numpy arrays

x_train = input_variables_values_training_datasets
y_train = target_variables_values_training_datasets  
x_test = input_variables_values_test_datasets

# Create linear regression object
linear = linear model.LinearRegression()

#Train the model using the training sets and
#check score 

linear.fit(x train, y_train)
linear.score(x train, y_train)

# Equation coefficient and Intercept

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear. intercept_) 

#Predict Output 
predicted = linear.predict(x_test) 
```



## __02-Logistic Regression:__

```python

# Import Library 
from sklearn.linear model import LogisticRegression

# Assumed you have, X (predictor) and Y (target) 
# for training data set and x_test(predictor) of test dataset 

# Create logistic regression object 
model = LogisticRegression()

# Train the model using the training sets and check score 
model.fit(X, y)
model.score(X, y)

# Equation coefficient and Intercept 
print('Coefficient: \n', model.coef_) 
print('Intercept: \n', model.intercept_)

# Predict Output
predicted = model. predict(x_test) 

```


## __03-Decision Tree:__

```python

# Import Library
# Import other necessary libraries like pandas, numpy...

from sklearn import tree

# Assumed you have, X (predictor) and Y (target) for
# training data set and x_test(predictor) of test dataset 

# Create tree object 
model = tree.DecisionTreeClassifier(criterion='gini') 

# for classification, here you can change the
# algorithm as gini or entropy (information gain) by 
# default it is gini 

model = tree.DecisionTreeRegressor() # for regression

# Train the model using the training sets and check score 
model.fit(X, y)
model.score(X, y) 

# Predict Output 
predicted = model.predict(x_test) 
```


## __04-Support Vector Machine(SVM):__

```python

# Import Library
from sklearn import svm

# Assumed you have, X (predictor) and Y (target) for
# training data set and x_test(predictor) of test_dataset 

# Create SVM classification object
model = svm.svc()

# there are various options associated with it, this is simple for classification.

# Train the model using the training sets & check the score
model.fit(X, y)
model.score(X, y)

# Predict Output 
predicted = model.predict(x_test) 

```

## __05-Naive Bayes:__

```python

# Import Library
from sklearn.naive bayes import GaussianNB

# Assumed you have, X (predictor) and Y (target) for
# training data set and x_test(predictor) of test_dataset 

# Create SVM classification object 
model = GaussianNB()

# there is other distribution for multinomial classes like Bernoulli Naive Bayes

# Train the model using the training sets and check score
model.fit(X, y)

# Predict Output 
predicted = model.predict(x_test) 

```


## __06-K-Nearest Neighbors(kNN):__

```python

# Import Library 
from sklearn.neighbors import KNeighborsClassifier

# Assumed you have, X (predictor) and Y (target) for 
# training data set and x_test(predictor) of test_dataset

# Create KNeighbors classifier object model
KNeighborsClassifier(n_neighbors=6) # default value for n neighbors is 5


# Train the model using the training sets and check score
model.fit(X, y)

# Predict Output
predicted = model.predict(x_test) 

```

## __07-k-Means Clustering:__

```python

# Import Library
from sklearn.cluster import KMeans

# Assumed you have, X (attributes) for training data set 
# and x test(attributes) of test dataset

# Create KNeighbors classifier object model
k means - KMeans(n clusters-3, random state=0)

#Train the model using the training sets and check score
model.fit(X)

#Predict Output 
predicted = model.predict(x_test) 

```
## __08-Random Forest:__

```python

# Import Library
from sklearn.ensemble import RandomForestClassifier

# Assumed you have, X (predictor) and Y (target) for 
# training data set and x_test(predictor) of test_dataset

# Create Random Forest object
model= RandomForestClassifier()

# Train the model using the training sets and check score
model.fit(X, y)

# Predict Output 
predicted = model.predict(x_test) 
```



## __09-Dimensionality Reduction Algorithms(e.g. PCA):__

```python

# Import Library 
from sklearn import decomposition

# Assumed you have training and test data set as train and test

# Create PCA object 
pca= decomposition.PCA(n_components=k) # default value of k -min(n sample, n features)

# For Factor analysis 
fa= decomposition.FactorAnalysis()

# Reduced the dimension of training dataset using PCA 
train_reduced = pca.fit_transform(train)

# Reduced the dimension of test dataset
test_reduced = pca.transform(test) 
```

## __10-Gradient Boosting & AdaBoost(e.g. GBDT):__

```python
 
# Import Library 
from sklearn.ensemble import GradientBoostingClassifier

# Assumed you have, X (predictor) and Y (target) for 
# training data set and x_test(predictor) of test_dataset

# Create Gradient Boosting Classifier object
model= GradientBoostingClassifier(n_estimators=100, \
         learning_rate=1.0, max_depth=1, random_state=0)
         
# Train the model using the training sets and check score 
model.fit(X, y) 

# Predict Output 
predicted = model.predict(x_test) 
```


### Connect with me:


[<img align="left" alt="codeSTACKr | Twitter" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/twitter.svg" />][twitter]
[<img align="left" alt="codeSTACKr | LinkedIn" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/linkedin.svg" />][linkedin]
[<img align="left" alt="codeSTACKr.com" width="22px" src="https://raw.githubusercontent.com/iconic/open-iconic/master/svg/globe.svg" />][StackExchange AI]

[twitter]: https://twitter.com/F4izy
[linkedin]: https://www.linkedin.com/in/faizy-mohd-836573122/
[StackExchange AI]: https://ai.stackexchange.com/users/36737/cypher


---


![Faizy's github stats](https://github-readme-stats.vercel.app/api?username=mohd-faizy&show_icons=true)


[![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=mohd-faizy&layout=compact)](https://github.com/mohd-faizy/github-readme-stats)

