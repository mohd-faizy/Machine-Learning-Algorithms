# __Machine Learning Algorithms__

<img src='https://github.com/mohd-faizy/__Machine_Learning_Algorithms__/blob/master/Algorithms_png/Head_ML.png'>

## __Classification according to the ways of learning:__

:black_circle: Supervised learning

:white_circle: Unsupervised learning

:black_circle: Semi-supervised learning

:white_circle: Reinforcement learning


---
<h2 style="text-align: left;">Classification according to the function</h2>
<table style="height: 496px; width: 629px;">
<tbody>
<tr>
<td style="width: 241px;">
<h3><strong>&nbsp;Regression algorithm</strong></h3>
</td>
<td style="width: 372px;">
<ul>
<li>Linear regression</li>
<li>&nbsp;Logistic regression</li>
<li>Multiple Adaptive Regression (MARS)</li>
<li>&nbsp;Local scatter smoothing estimate (LOESS)</li>
</ul>
</td>
</tr>
<tr>
<td style="width: 241px;">
<h3><strong>&nbsp;Instance-based Learning Algorithm</strong></h3>
</td>
<td style="width: 372px;">
<ul>
<li>K &mdash; proximity algorithm (kNN)</li>
<li>Learning vectorization (LVQ)</li>
<li>Self-Organizing Mapping Algorithm (SOM)</li>
<li>Local Weighted Learning Algorithm (LWL)</li>
</ul>
</td>
</tr>
<tr>
<td style="width: 241px;">
<h3>&nbsp;<strong>Regularization Algorithm</strong></h3>
</td>
<td style="width: 372px;">
<ul>
<li>Ridge Regression</li>
<li>LASSO（Least Absolute Shrinkage and Selection Operator)</li>
<li>Elastic Net</li>
<li>Minimum Angle Regression (LARS)</li>
</ul>
</td>
</tr>
<tr>
<td style="width: 241px;">
<h3><strong>&nbsp;Decision tree Algorithm</strong></h3>
</td>
<td style="width: 372px;">
<ul>
<li>Classification and Regression Tree (CART)</li>
<li>ID3 algorithm (Iterative Dichotomiser 3)</li>
<li>C4.5 and C5.0</li>
<li>CHAID（Chi-squared Automatic Interaction Detection）</li>
<li>Random Forest</li>
<li>Multivariate Adaptive Regression Spline (MARS)</li>
<li>Gradient Boosting Machine (GBM)</li>
</ul>
</td>
</tr>
<tr>
<td style="width: 241px;">
<h3><strong>&nbsp;Bayesian Algorithm</strong></h3>
</td>
<td style="width: 372px;">
<ul>
<li>Naive Bayes</li>
<li>Gaussian Bayes</li>
<li>Polynomial naive Bayes</li>
<li>AODE（Averaged One-Dependence Estimators）</li>
<li>Bayesian Belief Network</li>
</ul>
</td>
</tr>
<tr>
<td style="width: 241px;">
<h3><strong>&nbsp;Kernel-based Algorithm</strong></h3>
</td>
<td style="width: 372px;">
<ul>
<li>Support vector machine (SVM)</li>
<li>Radial Basis Function (RBF)</li>
<li>Linear Discriminate Analysis (LDA)</li>
</ul>
</td>
</tr>
<tr>
<td style="width: 241px;">&nbsp;
<h3><strong>&nbsp;Clustering Algorithm</strong></h3>
</td>
<td style="width: 372px;">
<ul>
<li>K &mdash; mean</li>
<li>K &mdash; medium number</li>
<li>EM algorithm</li>
<li>Hierarchical clustering</li>
</ul>
</td>
</tr>
<tr>
<td style="width: 241px;">&nbsp;
<h3><strong>&nbsp;Association Rule Learning</strong></h3>
</td>
<td style="width: 372px;">
<ul>
<li>&nbsp;Apriori algorithm</li>
<li>&nbsp;Eclat algorithm</li>
</ul>
</td>
</tr>
<tr>
<td style="width: 241px;">&nbsp;
<h3><strong>&nbsp;Neural Networks</strong></h3>
</td>
<td style="width: 372px;">
<ul>
<li>Sensor</li>
<li>Backpropagation algorithm (BP)</li>
<li>Hopfield network</li>
<li>Radial Basis Function Network (RBFN)</li>
</ul>
</td>
</tr>
<tr>
<td style="width: 241px;">
<h3><strong>&nbsp;Deep Learning</strong></h3>
</td>
<td style="width: 372px;">
<ul>
<li>Deep Boltzmann Machine (DBM)</li>
<li>Convolutional Neural Network (CNN)</li>
<li>Recurrent neural network (RNN, LSTM)</li>
<li>Stacked Auto-Encoder</li>
</ul>
</td>
</tr>
<tr>
<td style="width: 241px;">
<h3><strong>&nbsp;Dimensionality Reduction Algorithm</strong></h3>
</td>
<td style="width: 372px;">
<ul>
<li>Principal Component Analysis (PCA)</li>
<li>Principal component regression (PCR)</li>
<li>Partial least squares regression (PLSR)</li>
<li>Salmon map</li>
<li>Multidimensional scaling analysis (MDS)</li>
<li>Projection pursuit method (PP)</li>
<li>Linear Discriminant Analysis (LDA)</li>
<li>Mixed Discriminant Analysis (MDA)</li>
<li>Quadratic Discriminant Analysis (QDA)</li>
<li>Flexible Discriminant Analysis (FDA</li>
</ul>
</td>
</tr>
<tr>
<td style="width: 241px;">
<h3><strong>&nbsp;Integrated Algorithm</strong></h3>
</td>
<td style="width: 372px;">&nbsp;
<ul>
<li>Boosting</li>
<li>Bagging</li>
<li>AdaBoost</li>
<li>Stack generalization (mixed)</li>
<li>GBM algorithm</li>
<li>GBRT algorithm</li>
<li>Random forest</li>
</ul>
</td>
</tr>
<tr>
<td style="width: 241px;">
<h3><strong>&nbsp;Other Algorithms</strong></h3>
</td>
<td style="width: 372px;">&nbsp;
<ul>
<li>Feature selection algorithm</li>
<li>Performance evaluation algorithm</li>
<li>Natural language processing</li>
<li>Computer vision</li>
<li>Recommended system</li>
<li>Reinforcement learning</li>
<li>Migration learning</li>
</ul>
</td>
</tr>
</tbody>
</table>
<p>&nbsp;</p>

---
## __Popular Machine Learning Algorithms__

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

