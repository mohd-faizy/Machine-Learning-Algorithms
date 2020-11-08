# Decision Tree

Decision Tree : Decision tree is the most powerful and popular tool for classification and prediction. A Decision tree is a flowchart like tree structure, where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node (terminal node) holds a class label.

<img src='https://miro.medium.com/max/700/0*7xow30weh2lVxhsp.png'>


### How Decision tree works?
The general algorithm for decision tree can be described as follows:

:heavy_check_mark: :one: Select the best attribute that best splits or separates the data.

:heavy_check_mark: :two: Ask the relevant question.

:heavy_check_mark: :three: Follow the answer path.

:heavy_check_mark: :four: Repeat these steps until you arrive to the answer.

While constructing a decision tree, the major challenge is to identify the attribute for each root nodes in each level. This process is known as “attribute selection”. This selection can be done by using two methods.Let’s take a look at each.

:radio_button: __Information Gain__ : In order to keep our tree small, we must select an attribute which can split the data into purest form i.e, to split data distinctly. The split with highest information gain is used first and repeat the process until all children nodes are pure or the information gain is ‘0‘.

:radio_button: __Gini Index__ : Gini Index is the measurement of likelihood how often a randomly chosen element is misclassified. The attributes with lower Gini Index need to be consider for splitting or for making a decision.

## Decision Tree "Hello World!"

> [BankNote_Authentication.csv](https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data/download)

```python

# Data Collection
import pandas as pd
file = './BankNote_Authentication.csv'
data = pd.read_csv(file)
data.head()

# Data pre-processing
from sklearn.preprocessing import StandardScaler
scaler  = StandardScaler()
features = scaler.fit_transform(features)
features

# Building the model
from sklearn.model_selection import train_test_split
featureTrain, featureTest, labelTrain, labelTest = train_test_split(features, labels)

#  let’s start training our Decision Tree algorithm with data
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(featureTrain,labelTrain)

# Testing the model 
pred = model.predict(featureTest)
from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix:\n")
print(confusion_matrix(labelTest,pred))
print("\nClassification Report:\n")
print(classification_report(labelTest,pred))

# Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(labelTest,pred)

```


### __The strengths of decision tree methods are:__

- Simple to understand and to interpret. Trees can be visualised.

- Requires little data preparation. Other techniques often require data normalization, dummy variables need to be created and blank values to be removed. Note however that this module does not support missing values.

- The cost of using the tree (i.e., predicting data) is logarithmic in the number of data points used to train the tree.

- Able to handle both numerical and categorical data. Other techniques are usually specialized in analyzing datasets that have only one type of variable.

- Able to handle multi-output problems.

- Uses a white box model. If a given situation is observable in a model, the explanation for the condition is easily explained by boolean logic. By contrast, in a black box model (e.g., in an artificial neural network), results may be more difficult to interpret.

- Possible to validate a model using statistical tests. That makes it possible to account for the reliability of the model.

- Performs well even if its assumptions are somewhat violated by the true model from which the data were generated.
 

### __The disadvantages of decision trees include:__

- Decision-tree learners can create over-complex trees that do not generalize the data well. This is called overfitting. Mechanisms such as pruning (not currently supported), setting the minimum number of samples required at a leaf node or setting the maximum depth of the tree are necessary to avoid this problem.


- Decision trees can be unstable because small variations in the data might result in a completely different tree being generated. This problem is mitigated by using decision trees within an ensemble.


- The problem of learning an optimal decision tree is known to be NP-complete under several aspects of optimality and even for simple concepts. Consequently, practical decision-tree learning algorithms are based on heuristic algorithms such as the greedy algorithm where locally optimal decisions are made at each node. Such algorithms cannot guarantee to return the globally optimal decision tree. This can be mitigated by training multiple trees in an ensemble learner, where the features and samples are randomly sampled with replacement.

- There are concepts that are hard to learn because decision trees do not express them easily, such as XOR, parity or multiplexer problems.
