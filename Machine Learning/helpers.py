from sklearn import svm, metrics
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics
import math
from sklearn.ensemble import BaggingClassifier
import time
import seaborn as sns

def mnist_pca(X, y):
    '''Perform a PCA on the input data and produce 
    a couple of visualisations'''
    # Perform PCA
    pca = PCA()
    pca.fit(X)

    # Get explained vairance ratios and the information for PC1 and PC2
    exp_var = pca.explained_variance_ratio_
    exp_var = np.array([i for i in enumerate(exp_var)])
    best_pcs = pca.fit_transform(X)[:,0:2]

    # Variance explained by first 100 PCs
    print(np.sum(exp_var[:100, 1]))

    # Proportion of variance explained by each PC plot
    plt.figure(figsize=(15,5))
    plt.subplot(121)

    plt.semilogy(exp_var[:700,0], exp_var[:700,1])
    plt.title("Explained variance of each principal component")
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.ylim(0.000000001, 1)
    plt.subplots_adjust(left=0.125, right=0.9)
    plt.grid(which="major")

    plt.subplot(122)
    plt.title("Scatter plot on first two principal components")
    plt.scatter(best_pcs[:, 0], best_pcs[:,1], alpha=0.05, c=y)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show(block=False)

def svm_crosstrain(x, y, x_cv, y_cv, costs=[1], kernel="linear", gamma=["auto"], n=10, samp_max=0.1, feat_max=1.0, deg=[1], bootstrap=True, coef=0.0):
    
    # Fit SVMs and find the best one
    t0, ti = time.time(), time.time()
    scores = []
    for c in costs:
        for g in gamma:
            if kernel == "linear": mod = svm.LinearSVC(C=c, dual=False)
            else: 
                mod = BaggingClassifier(svm.SVC(C=c, gamma=(g), kernel=kernel, coef0=coef), 
                        n_estimators=n, bootstrap=bootstrap, max_samples=samp_max, max_features=feat_max, n_jobs=-1)
            mod.fit(x, y)
            score = mod.score(x_cv, y_cv)
            scores.append(score)
            print(f"\nScore for C = {c}, gamma = {g}: {score}\nTime taken: {time.time() - ti}")
            print(f"Cumulative time: {time.time() - t0}")
            ti = time.time()
        
    return scores

def CrossHeat(p1, p2, p1_name, p2_name, scores):
    heat_scores = pd.DataFrame({p1_name: np.repeat(p1, len(p2)), p2_name: p2 * len(p1), 
                                "Score": scores})
    heat_scores_pivot = heat_scores.pivot(p2_name, p1_name, "Score")

    plt.figure(figsize=(8, 8))
    sns.heatmap(heat_scores_pivot, annot=True, fmt=".4f", cmap="hot_r")
    plt.show(block=False)
    print("Best Result:\n", heat_scores[heat_scores["Score"] == max(heat_scores["Score"])])

def CrossLine(parameter, parameter_name, scores, logx=True):
    # Plot to view how the score changes with different values of c
    plt.subplot(111)
    plt.figure(figsize=(8,8))
    plt.clf()
    plt.plot(parameter, scores)
    locs, _ = plt.yticks()
    plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
    plt.ylabel('Accuracy on validation set')
    plt.xlabel(f'Cost parameter {parameter_name}')
    if logx == True: plt.xscale('log')
    plt.grid(which="minor")
    plt.grid(which="major")
    plt.show(block=False)

def ClassReport(mod, x, y, NN=0):
    if NN == 1:
        pred = mod.predict(x)[1]
        predicted = pred.argmax(axis=1)
    
    if NN == 2:
        predicted = mod.predict_classes(x)
    else: predicted = mod.predict(x)
    
    print("Model score: %s\n" % sklearn.metrics.accuracy_score(y, predicted))
    print("Classification report for classifier %s:\n%s\n"
        % (mod, metrics.classification_report(y, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(y, predicted))

class NeuralNetwork:
    '''Class that takes in some independent variables (x) with labels (y) 
    into a neural network with a single hidden layer of 100 nodes'''
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
        # Weights are initialised randomly
        # There are 101 because we are going to add bias nodes
        self.weights_1 = np.random.randn(101, len(x.T) + 1)
        self.weights_2 = np.random.randn(10, 101)
    
    def inputs(self, x):
        '''The inputs function is just to add a bias node to whatever input data we are using.'''
        i = np.ones((len(x), len(x.T) + 1), dtype=np.float64)
        i[:, :-1] = x
        return i
        
    def target(self, y):
        '''Turns multi-class labels into dummy variables. Labels should be numbers.'''
        return np.array([np.float_(y == i) for i in range(len(set(y)))], dtype=np.float64).T
    
    def predict(self, x):
        '''The forward propogation of the network, returns the activations in the hidden layer
        and in the output layer.'''
        z1 = np.float_(np.matmul(self.inputs(x), self.weights_1.T))
        alpha1 = 1 / (1 + np.exp(-z1))
        z2 = np.matmul(alpha1, self.weights_2.T)
        alpha2 = 1 / (1 + np.exp(-z2))

        # Problem is these values do not add up to 1
        # So we need to softmax them
        alp_sum = np.sum(alpha2, axis=1, dtype=np.float64)[:, np.newaxis]
        alpha2 = alpha2 / alp_sum

        return alpha1, alpha2

    def cost(self, x, y):
        '''Returns the least squared cost of the network for a given dataset'''
        return np.sum((self.predict(x)[1] - self.target(y)) ** 2)/(2 * len(x))

    def gradient(self, x, y):
        '''Returns a tuple of the gradient of the weights from the input layer to the hidden layer
        and the weights from the hidden layer to the output layer using backpropogation'''
        alpha1, alpha2 = self.predict(x)
        x1 = np.ones((x.shape[0], x.shape[1] + 1))
        x1[:, :-1] = x

        w2_delta = np.dot(alpha1.T, (y - alpha2) * alpha2 * (1 - alpha2)) / len(x)
        w1_delta = np.dot(x1.T, (np.dot((y - alpha2) * alpha2 * (1 - alpha2), self.weights_2) * alpha1 * (1 - alpha1))) / len(x)
        return w1_delta.T, w2_delta.T
    
    def grad_descent(self, l=0.00001, steps=10):
        '''Uses results from backpropogation to conduct gradient descent'''
        for step in range(steps):
            w1_delta, w2_delta = self.gradient(self.x, self.target(self.y))
            self.weights_1 += l * w1_delta
            self.weights_2 += l * w2_delta
            
            # To check progress
            if (step + 1) % 100 == 0:
                print(f"{step + 1} steps completed")
    
    def stoc_grad_descent(self, epochs=3, l=0.00001, steps_per_epoch=None):
        '''Stochastic gradient descent to achieve faster convergence'''

        # Firstly we need to shuffle the data
        z = np.zeros((len(self.x), 785))
        z[:,:-1] = self.x
        z[:, -1] = self.y
        np.random.shuffle(z)
        x, y = z[:, :-1], z[:, -1]
        if steps_per_epoch == None: rang = len(x)
        else: rang = steps_per_epoch
        for epoch in range(epochs):
            print(f"epoch {epoch + 1} begins")
            for i in range(0, rang):
                w1_delta, w2_delta = self.gradient(x=x[i,:][np.newaxis], y=self.target(y)[i, :])
                self.weights_1 += l * w1_delta
                self.weights_2 += l * w2_delta

# Tensorflow separated, since it's often on a different environment
try:
    import tensorflow as tf
    def NeuralNetwork2(X_train, y_train, X_test, y_test, nodes=[64, 64], conv=[False, False], activation=["relu", "relu"], loss="sparse_categorical_crossentropy", 
    alpha=0.001, dropout=[None, None], batch=25, epochs=3, steps_per_epoch=None, batch_normalisation=False, vs=0, kernel_size=[5, 5]):
        '''Creates a multi-layer perceptron for the x and y data. Parameters to select the number of layers as an integer,
        the number of nodes in each layer as a list, the activation function of each node as a list and a loss function'''

        # Create the model
        model = tf.keras.Sequential()
        if conv[0] == False: model.add(tf.keras.layers.Flatten())

        # Add layers
        for i, layer in enumerate(nodes):
            if conv[i] == False: model.add(tf.keras.layers.Dense(layer, activation=activation[i]))
            else: 
                if i == 0: model.add(tf.keras.layers.Conv2D(layer, kernel_size=5, activation=activation[i], input_shape=(28, 28, 1)))
                else: model.add(tf.keras.layers.Conv2D(layer, kernel_size=5, activation=activation[i]))
                model.add(tf.keras.layers.MaxPool2D((2, 2)))
                if i + 1 == len(nodes) or conv[i + 1] == False: model.add(tf.keras.layers.Flatten())
            # Option to add a dropout layer
            if dropout[i] != None:
                model.add(tf.keras.layers.Dropout(dropout[i]))
        
        if batch_normalisation == True: model.add(tf.keras.layers.BatchNormalization())

        # Add output layer
        model.add(tf.keras.layers.Dense(10, activation="softmax"))

        # compile method takes three arguments - an optimiser for gradient descent, a loss function to minimise and a metric
        # to monitor the training
        model.compile(optimizer=tf.train.AdamOptimizer(alpha), loss=loss, metrics=["accuracy"])

        # Fit the model
        if steps_per_epoch == None: history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_split=vs)
        else: history = model.fit(X_train, y_train, epochs=epochs, steps_per_epoch=steps_per_epoch, batch_size=batch, validation_split=vs)
        return model, history

    def PlotHistory(hist, metric, legend=None):
        fig, ax = plt.subplots()
        for h in hist: 
            ax.plot(np.arange(len(h.history[metric])),h.history[metric])

        plt.xlabel("Epoch")
        plt.ylabel("Accuarcy")
        if legend != None: ax.legend(legend)
        plt.grid()
        plt.show()

except: pass