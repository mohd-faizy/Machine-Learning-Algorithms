# __Machine Learning Algorithms__

<img src='https://github.com/mohd-faizy/__Machine_Learning_Algorithms__/blob/master/Algorithms_png/Head_ML.png'>

## __Classification according to the ways of learning:__

:black_circle: Supervised learning

:white_circle: Unsupervised learning

:black_circle: Semi-supervised learning

:white_circle: Reinforcement learning


## __Classification according to the function:__

### :heavy_check_mark: __Regression algorithm__

:white_square_button: Linear regression
 
:white_square_button: Logistic regression  
 
:white_square_button: Multiple Adaptive Regression (MARS)
 
:white_square_button: Local scatter smoothing estimate (LOESS)
 
 ---
 
### :heavy_check_mark: __Instance-based learning algorithm__

:white_square_button: K — proximity algorithm (kNN)
 
:white_square_button: Learning vectorization (LVQ)
 
:white_square_button: Self-Organizing Mapping Algorithm (SOM)
 
:white_square_button: Local Weighted Learning Algorithm (LWL)
 
 ---

### :heavy_check_mark: __Regularization algorithm__

:white_square_button: Ridge Regression

:white_square_button: LASSO（Least Absolute Shrinkage and Selection Operator)

:white_square_button: Elastic Net 

:white_square_button: Minimum Angle Regression (LARS)


--- 


### :heavy_check_mark: __Decision tree algorithm__

:white_square_button: Classification and Regression Tree (CART)

:white_square_button: ID3 algorithm (Iterative Dichotomiser 3)

:white_square_button: C4.5 and C5.0

:white_square_button: CHAID（Chi-squared Automatic Interaction Detection(）

:white_square_button: Random Forest

:white_square_button: Multivariate Adaptive Regression Spline (MARS)|

:white_square_button: Gradient Boosting Machine (GBM)|


---

### :heavy_check_mark: __Bayesian algorithm__

:white_square_button: Naive Bayes

:white_square_button: Gaussian Bayes 

:white_square_button: Polynomial naive Bayes

:white_square_button: AODE（Averaged One-Dependence Estimators）

:white_square_button: Bayesian Belief Network


---

###:heavy_check_mark: __Kernel-based algorithm__

:white_square_button: Support vector machine (SVM)

:white_square_button: Radial Basis Function (RBF)

:white_square_button: Linear Discriminate Analysis (LDA)


---


### :heavy_check_mark: __Clustering Algorithm__

:white_square_button: K — mean

:white_square_button: K — medium number

:white_square_button: EM algorithm

:white_square_button: Hierarchical clustering


---

### :heavy_check_mark: __Association rule learning__

:white_square_button: Apriori algorithm

:white_square_button: Eclat algorithm


---

### :heavy_check_mark: __Neural Networks__

:white_square_button: Sensor

:white_square_button: Backpropagation algorithm (BP)

:white_square_button: Hopfield network

:white_square_button: Radial Basis Function Network (RBFN)


---

### :heavy_check_mark: __Deep learning__

:white_square_button: Deep Boltzmann Machine (DBM)

:white_square_button: Convolutional Neural Network (CNN)

:white_square_button: Recurrent neural network (RNN, LSTM)

:white_square_button: Stacked Auto-Encoder


---


### :heavy_check_mark: __Dimensionality reduction algorithm__

:white_square_button: Principal Component Analysis (PCA)

:white_square_button: Principal component regression (PCR)

:white_square_button: Partial least squares regression (PLSR)

:white_square_button: Salmon map

:white_square_button: Multidimensional scaling analysis (MDS)

:white_square_button: Projection pursuit method (PP)

:white_square_button: Linear Discriminant Analysis (LDA)

:white_square_button: Mixed Discriminant Analysis (MDA)

:white_square_button: Quadratic Discriminant Analysis (QDA)

:white_square_button: Flexible Discriminant Analysis (FDA)


---

### :heavy_check_mark: __Integrated algorithm__

:white_square_button: Boosting

:white_square_button: Bagging

:white_square_button: AdaBoost

:white_square_button: Stack generalization (mixed)

:white_square_button: GBM algorithm

:white_square_button: GBRT algorithm

:white_square_button: Random forest
 
 
---


### :heavy_check_mark: __Other algorithms__

:white_square_button: Feature selection algorithm

:white_square_button: Performance evaluation algorithm

:white_square_button: Natural language processing

:white_square_button: Computer vision

:white_square_button: Recommended system

:white_square_button: Reinforcement learning

:white_square_button: Migration learning

---

## :one:__Linear Regression:__

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



## :two:__Logistic Regression:__

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


## :three:__Decision Tree:__

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


## :four:__Support Vector Machine(SVM):__

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

## :five:__Naive Bayes:__

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


## :six:__K-Nearest Neighbors(kNN):__

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

## :seven:__k-Means Clustering:__

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
## :eight:__Random Forest:__

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



## :nine:__Dimensionality Reduction Algorithms(e.g. PCA):__

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

## :one::zero:__Gradient Boosting & AdaBoost(e.g. GBDT):__

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

