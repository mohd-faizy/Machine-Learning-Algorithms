# __Machine Learning Algorithms__

## __Linear Regression__

Linear regression is a supervised learning algorithm and tries to model the relationship between a continuous target variable and one or more independent variables by fitting a linear equation to the data.
For a linear regression to be a good choice, there needs to be a linear relation between independent variable(s) and target variable. There are many tools to explore the relationship among variables such as scatter plots and correlation matrix. For example, the scatter plot below shows a positive correlation between an independent variable (x-axis) and dependent variable (y-axis). As one increases, the other one also increases.


<img src ='https://miro.medium.com/max/490/0*SJucFv9TduqDWgw7.png'>

A linear regression model tries to fit a regression line to the data points that best represents the relations or correlations. The most common technique to use is ordinary-least squares (OLE). With this method, best regression line is found by minimizing the sum of squares of the distance between data points and the regression line. For the data points above, the regression line obtained using OLE seems like:


<img src='https://miro.medium.com/max/508/0*e2N94sdwIpaNs5iE.jpeg'>

## __Logistic Regression__

## __Decision Tree__
## __Support Vector Machine(SVM)__
Support Vector Machine (SVM) is a supervised learning algorithm and mostly used for classification tasks but it is also suitable for regression tasks.
SVM distinguishes classes by drawing a decision boundary. How to draw or determine the decision boundary is the most critical part in SVM algorithms. Before creating the decision boundary, each observation (or data point) is plotted in n-dimensional space. “n” is the number of features used. For instance, if we use “length” and “width” to classify different “cells”, observations are plotted in a 2-dimensional space and decision boundary is a line. If we use 3 features, decision boundary is a plane in 3-dimensional space. If we use more than 3 features, decision boundary becomes a hyperplane which is really hard to visualize.

<img src='https://miro.medium.com/max/455/0*JgKQiYT_f74pU85_.png'>

Decision boundary is drawn in a way that the distance to support vectors are maximized. If the decision boundary is too close to a support vector, it will be highly sensitive to noises and not generalize well. Even very small changes in independent variables may cause a misclassification.
The data points are not always linearly separable like in the figure above. In these cases, SVM uses kernel trick which measures the similarity (or closeness) of data points in a higher dimensional space in order to make them linearly separable.

Kernel function is kind of a similarity measure. The inputs are original features and the output is a similarity measure in the new feature space. Similarity here means a degree of closeness. It is a costly operation to actually transform data points to a high-dimensional feature space. The algorithm does not actually transform the data points to a new, high dimensional feature space. Kernelized SVM compute decision boundaries in terms of similarity measures in a high-dimensional feature space without actually doing a transformation. I think this is why it is also called kernel trick.

SVM is especially effective in cases where number of dimensions are more than the number of samples. When finding the decision boundary, SVM uses a subset of training points rather than all points which makes it memory efficient. On the other hand, training time increases for large datasets which negatively effects the performance.

## __Naive Bayes__
## __K-Nearest Neighbors(kNN)__
## __k-Means Clustering__
## __Random Forest__
## __Dimensiopnality Reduction Algorithms(e.g. PCA)__
## __Gradient Boosting & AdaBoost(e,g. GBDT)__



TDSSRC=https://towardsdatascience.com/11-most-common-machine-learning-algorithms-explained-in-a-nutshell-cc6e98df93be
