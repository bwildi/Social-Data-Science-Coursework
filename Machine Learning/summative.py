import helpers as hp
import numpy as np
import matplotlib.pyplot as plt
import time
from mnist import MNIST
import os

directory = os.getcwd() + "\\mldata"
mnist = MNIST(directory)
X_train, y_train = mnist.load_training()
X_test, y_test = mnist.load_testing()
X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

# Visualise a few of the digits
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(X_train[i].reshape((28, 28)), cmap="gray", interpolation=None)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Label: {y_train[i]}")
plt.show()

# Scale X to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# PCA
hp.mnist_pca(X_train, y_train)

# Split some cross validation data
X_train1, X_cv, y_train1, y_cv = hp.train_test_split(X_train, y_train, test_size=(1/7), random_state=6)

# Linear SVM
costs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
linear_svm_cscores = hp.svm_crosstrain(X_train1, y_train1, X_cv, y_cv, costs=costs)

hp.CrossLine(costs, "C", linear_svm_cscores)

# C = 0.1 seems optimal
linear_svm_mod = hp.svm.LinearSVC(C=0.1, random_state=20, dual=False)
linear_svm_mod.fit(X_train, y_train)

hp.ClassReport(linear_svm_mod, X_test, y_test)

# Cross validating on rbf kernel is very computationally expensive
# So we're going to use bagging to find some gamma and cost paramters
# This is also useful since Sklearn makes it easy to use multi-processing
# On each bag
# Each iteration takes 30 - 60 seconds
C = [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
G = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
rbf_svm_scores = hp.svm_crosstrain(X_train1, y_train1, X_cv, y_cv, costs=C, kernel="rbf", n=8, samp_max=0.02, gamma=G)

hp.CrossHeat(C, G, "C", "Gamma", rbf_svm_scores)

# Okay we have lots of potentially good values of C and gamma
# We will take C = 50 and gamma = 0.01
rbf_classifier = hp.svm.SVC(C=50, gamma=0.01, kernel="rbf")
rbf_classifier.fit(X_train, y_train)

# Strong performance, 98.25% accuracy
hp.ClassReport(rbf_classifier, X_test, y_test)

# Neural network time
NN = hp.NeuralNetwork(X_train, y_train)

# Let's do a forward propogation and see the initial cost
ic = NN.cost(NN.x, NN.y)
print("Pre-training cost:", ic)

# Initial cost around 0.5 
# Now let's run gradient descent and see if it goes down
# l = 1 seems to work
steps = [i for i in range(20)]
step_cost = [ic]
for i in steps:
    NN.grad_descent(l=1, steps=100)
    cost = NN.cost(NN.x, NN.y)
    step_cost.append(cost)
    print(f"Cost = {cost}")

steps = list((np.array(steps) + 1) * 100)
steps.insert(0, 0)
plt.figure(figsize=(8,8))
plt.plot(steps, step_cost)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.show(block=False)

# Score report
hp.ClassReport(NN, X_test, y_test, NN=1)

# Let's try stochastic
NN2 = hp.NeuralNetwork(X_train, y_train)
NN2.stoc_grad_descent(epochs=5, l=0.1)

# Score report
pred = NN2.predict(X_test)[1]
predicted = pred.argmax(axis=1)
print("Model score: %s\n" % sklearn.metrics.accuracy_score(y_test, predicted))
print("Classification report for classifier %s:\n%s\n"
    % (NN2, metrics.classification_report(y_test, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predicted))
hp.ClassReport(NN2, X_test, y_test, NN=1)