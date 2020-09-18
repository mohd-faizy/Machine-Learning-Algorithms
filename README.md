# __Machine Learning Algorithms__

## __01-Linear Regression__

Linear regression is a supervised learning algorithm and tries to model the relationship between a continuous target variable and one or more independent variables by fitting a linear equation to the data.
For a linear regression to be a good choice, there needs to be a linear relation between independent variable(s) and target variable. There are many tools to explore the relationship among variables such as scatter plots and correlation matrix. For example, the scatter plot below shows a positive correlation between an independent variable (x-axis) and dependent variable (y-axis). As one increases, the other one also increases.


<img src ='https://miro.medium.com/max/490/0*SJucFv9TduqDWgw7.png'>

A linear regression model tries to fit a regression line to the data points that best represents the relations or correlations. The most common technique to use is ordinary-least squares (OLE). With this method, best regression line is found by minimizing the sum of squares of the distance between data points and the regression line. For the data points above, the regression line obtained using OLE seems like:


<img src='https://miro.medium.com/max/508/0*e2N94sdwIpaNs5iE.jpeg'>

## __02-Logistic Regression__

Logistic regression is a supervised learning algorithm which is mostly used for binary classification problems. Although “regression” contradicts with “classification”, the focus here is on the word “logistic” referring to logistic function which does the classification task in this algorithm. Logistic regression is a simple yet very effective classification algorithm so it is commonly used for many binary classification tasks. Customer churn, spam email, website or ad click predictions are some examples of the areas where logistic regression offers a powerful solution.

The basis of logistic regression is the logistic function, also called the sigmoid function, which takes in any real valued number and maps it to a value between 0 and 1.

<img src='https://miro.medium.com/max/483/0*Xe43fpJ941_xkmud.png'>

Consider we have the following linear equation to solve:

<img src='https://miro.medium.com/max/363/0*-j9l4GxxyNd32ehx.png'>

Logistic regression model takes a linear equation as input and uses logistic function and log odds to perform a binary classification task. Then, we will get the famous s shaped graph of logistic regression:

<img src='https://miro.medium.com/max/693/0*Qpp-M16hdTKQ-Uvb.png'>

We can use the calculated probability ‘as is’. For example, the output can be “the probability that this email is spam is 95%” or “the probability that customer will click on this ad is 70%”. However, in most cases, probabilities are used to classify data points. For instance, if the probability is greater than 50%, the prediction is positive class (1). Otherwise, the prediction is negative class (0).


It is not always desired to choose positive class for all probability values higher than 50%. Regarding the spam email case, we have to be almost sure in order to classify an email as spam. Since emails detected as spam directly go to spam folder, we do not want the user to miss important emails. Emails are not classified as spam unless we are almost sure. On the other hand, when classification in a health-related issue requires us to be much more sensitive. Even if we are a little suspicious that a cell is malignant, we do not want to miss it. So the value that serves as a threshold between positive and negative class is problem-dependent. Good thing is that logistic regression allows us to adjust this threshold value.



## __03-Decision Tree__

A decision tree builds upon iteratively asking questions to partition data. It is easier to conceptualize the partitioning data with a visual representation of a decision tree:

<img src='https://miro.medium.com/max/700/0*k_ug4HTto4BPsHSJ.png'>

## __04-Support Vector Machine(SVM)__
Support Vector Machine (SVM) is a supervised learning algorithm and mostly used for classification tasks but it is also suitable for regression tasks.
SVM distinguishes classes by drawing a decision boundary. How to draw or determine the decision boundary is the most critical part in SVM algorithms. Before creating the decision boundary, each observation (or data point) is plotted in n-dimensional space. “n” is the number of features used. For instance, if we use “length” and “width” to classify different “cells”, observations are plotted in a 2-dimensional space and decision boundary is a line. If we use 3 features, decision boundary is a plane in 3-dimensional space. If we use more than 3 features, decision boundary becomes a hyperplane which is really hard to visualize.

<img src='https://miro.medium.com/max/455/0*JgKQiYT_f74pU85_.png'>

Decision boundary is drawn in a way that the distance to support vectors are maximized. If the decision boundary is too close to a support vector, it will be highly sensitive to noises and not generalize well. Even very small changes in independent variables may cause a misclassification.
The data points are not always linearly separable like in the figure above. In these cases, SVM uses kernel trick which measures the similarity (or closeness) of data points in a higher dimensional space in order to make them linearly separable.

Kernel function is kind of a similarity measure. The inputs are original features and the output is a similarity measure in the new feature space. Similarity here means a degree of closeness. It is a costly operation to actually transform data points to a high-dimensional feature space. The algorithm does not actually transform the data points to a new, high dimensional feature space. Kernelized SVM compute decision boundaries in terms of similarity measures in a high-dimensional feature space without actually doing a transformation. I think this is why it is also called kernel trick.

SVM is especially effective in cases where number of dimensions are more than the number of samples. When finding the decision boundary, SVM uses a subset of training points rather than all points which makes it memory efficient. On the other hand, training time increases for large datasets which negatively effects the performance.

## __05-Naive Bayes__
Naive Bayes is a supervised learning algorithm used for classification tasks. Hence, it is also called Naive Bayes Classifier.

Naive bayes assumes that features are independent of each other and there is no correlation between features. However, this is not the case in real life. This naive assumption of features being uncorrelated is the reason why this algorithm is called “naive”.

The intuition behind naive bayes algorithm is the bayes’ theorem:

<img src='https://miro.medium.com/max/604/0*-Cq1pA2sfPJhyMDQ.png'>


p(A|B): Probability of event A given event B has already occurred
p(B|A): Probability of event B given event A has already occuured
p(A): Probability of event A
p(B): Probability of event B
Naive bayes classifier calculates the probability of a class given a set of feature values (i.e. p(yi | x1, x2 , … , xn)). Input this into Bayes’ theorem:

<img src='https://miro.medium.com/max/600/0*YCm8DSZwoKLz8Vj4.png'>

p(x1, x2 , … , xn | yi) means the probability of a specific combination of features (an observation / row in a dataset) given a class label. We need extremely large datasets to have an estimate on the probability distribution for all different combinations of feature values. To overcome this issue, naive bayes algorithm assumes that all features are independent of each other. Furthermore, denominator (p(x1,x2, … , xn)) can be removed to simplify the equation because it only normalizes the value of conditional probability of a class given an observation ( p(yi | x1,x2, … , xn)).
The probability of a class ( p(yi) ) is very simple to calculate:

<img src='https://miro.medium.com/max/508/0*tirvpl3LU-SVEDcX.png'>

Under the assumption of features being independent, p(x1, x2 , … , xn | yi) can be written as:

<img src='https://miro.medium.com/max/679/0*TX0XkmBcywNqJ7bM.png'>

The conditional probability for a single feature given the class label (i.e. p(x1 | yi) ) can be more easily estimated from the data. The algorithm needs to store probability distributions of features for each class independently. For example, if there are 5 classes and 10 features, 50 different probability distributions need to be stored.
Adding all these up, it became an easy task for naive bayes algorithm to calculate the probability to observe a class given values of features (p(yi | x1, x2 , … , xn) )
The assumption that all features are independent makes naive bayes algorithm very fast compared to complicated algorithms. In some cases, speed is preferred over higher accuracy. On the other hand, the same assumption makes naive bayes algorithm less accurate than complicated algorithms. Speed comes at a cost!

## __06-K-Nearest Neighbors(kNN)__

K-nearest neighbors (kNN) is a supervised learning algorithm that can be used to solve both classification and regression tasks. The main idea behind kNN is that the value or class of a data point is determined by the data points around it.

kNN classifier determines the class of a data point by majority voting principle. For instance, if k is set to 5, the classes of 5 closest points are checked. Prediction is done according to the majority class. Similarly, kNN regression takes the mean value of 5 closest points. Let’s go over an example. Consider the following data points that belong to 4 different classes:

<img src='https://miro.medium.com/max/610/0*JITsPkWWA8DU62ac.png'>

Let’s see how the predicted classes change according to the k value:

<img src='https://miro.medium.com/max/610/0*OcaG0BdtFCGKOnv9.png'>

<img src='https://miro.medium.com/max/610/0*cU1E1HGhvrOeFvYN.png'>

It is very important to determine an optimal k value. If k is too low, the model is too specific and not generalized well. It also tends to be sensitive to noise. The model accomplishes a high accuracy on train set but will be a poor predictor on new, previously unseen data points. Therefore, we are likely to end up with an overfit model. On the other hand, if k is too large, the model is too generalized and not a good predictor on both train and test sets. This situation is known as underfitting.


kNN is simple and easy to interpret. It does not make any assumption so it can be implemented in non-linear tasks. kNN becomes very slow as the number of data points increases because the model needs to store all data points. Thus, it is also not memory efficient. Another downside of kNN is that it is sensitive to outliers.


## __07-k-Means Clustering__
## __08-Random Forest__
## __09-Dimensionality Reduction Algorithms(e.g. PCA)__

PCA is a dimensionality reduction algorithm which basically derives new features from the existing ones with keeping as much information as possible. PCA is an unsupervised learning algorithm but it is also widely used as a preprocessing step for supervised learning algorithms.
PCA derives new features by finding the relations among features within a dataset.

> Note: PCA is a linear dimensionality reduction algorithm. There are also non-linear methods available.
The aim of PCA is to explain the variance within the original dataset as much as possible by using less features (or columns). The new derived features are called principal components. The order of principal components is determined according to the fraction of variance of original dataset they explain.

<img src='https://miro.medium.com/max/393/0*a8S1njE_ZxJf01Pq.png'>

> The principal components are linear combinations of the features of original dataset.

The advantage of PCA is that a significant amount of variance of the original dataset is retained using much smaller number of features than the original dataset. Principal components are ordered according to the amount of variance they explain.


## __10-Gradient Boosting & AdaBoost(e,g. GBDT)__


TDSSRC=https://towardsdatascience.com/11-most-common-machine-learning-algorithms-explained-in-a-nutshell-cc6e98df93be
