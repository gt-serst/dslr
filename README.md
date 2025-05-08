# dslr
## Subject
### Predict the Hogwarts house of a student based on a set of features such as: First Name, Last Name, Birthday, Best Hand, Arithmancy, etc.
### The project is divided into several components:
#### Data analysis
##### A program that takes a dataset as input and displays various statistical metrics for all numerical features.
#### Data visualization
##### A set of scripts that generate histograms, scatter plots, and pair plots to help understand the distribution, correlation, and significance of the data.
#### Logistic regression
##### Implementation of a multi-class classifier using the one-vs-all logistic regression approach.
##### Two main programs:
1. Training Program: Takes a training dataset as input, uses gradient descent to minimize the cost function, and saves the resulting coefficients to a file.
2. Prediction Program: Takes a test dataset and a file containing the learned coefficients as inputs. It outputs a file with the predicted houses for the given students.
## A bit of math and logic
### One vs. All
#### It is possible to use a softmax regression model as a multi-classifier, but here it is required to use a one-vs-all strategy. The concept is to build a binary logistic regression for each class (house), where the output represents the probability of a student belonging to that specific house (class = 1) versus not belonging to it (class = 0). This probability is calculated using the sigmoid function. Since there are four houses, four separate classifiers are trained. During prediction, each model outputs a probability score, and the predicted house is the one associated with the highest probability.
### Sigmoid function
#### The model predicts a probability for each training example using the sigmoid function.
```math
\sigma(z) = \frac{1}{1 + e^{-z}}
```
```math
z = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_n x_n = \boldsymbol{\theta}^T \boldsymbol{x}
```
Where:
- **x** is the input feature vector (including a 1 for the bias term),
- **θ** is the vector of model parameters (weights).

### Loss function
#### Binary Cross-Entropy evaluates how well the predicted probabilities align with the actual class labels. If the predicted value pp closely matches the true label yy, the resulting loss is small, reflecting an accurate prediction. On the other hand, the further the prediction is from the actual value, the larger the loss becomes, signaling a poor prediction. The use of logarithms in the formula ensures that wrong predictions are penalized more sharply than correct ones.
```math
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
```
Where:
- **m** is the number of samples,
- **y_i** is the true class label (0 or 1) for sample i,
- **p_i** is the predicted probability for sample i.
### Gradient descent
#### To reduce the error, the gradient of the loss is calculated with respect to each model parameter, indicating the direction and rate at which the parameter should be changed.
```math
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left(p_i - y_i \right) x_{ij}
```
Where:
- **m** is the number of samples,
- **y_i** is the true class label (0 or 1) for sample i,
- **p_i** is the predicted probability for sample i.
- **x_ij** is the value of feature j for sample i.
#### The parameters are then updated slightly in the opposite direction of the gradient. This process repeats over many iterations until the gradient becomes very small, suggesting that the model has reached a point where further changes won’t significantly improve performance.
```math
\theta_j = \theta_j - \alpha \cdot \frac{\partial J(\theta)}{\partial \theta_j}
```
Where:
- **θ_j** is the parameter (weight) for feature j,
- **α** is the learning rate (a small positive number controlling the step size).
