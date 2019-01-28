import numpy as np
import tensorflow as tf
import helpers as hp
import sklearn

# Get the digits dataset, split it and divide by colour value to get floating point
mnist = tf.keras.datasets.mnist
(X_train0, y_train0), (X_test, y_test) = mnist.load_data()
X_train0, X_test = X_train0 / 255.0, X_test / 255.0

# Also going to split training data again for some cross-validation
X_train1, X_cv, y_train1, y_cv = hp.train_test_split(X_train0, y_train0, test_size=(1/7), random_state=6)

# Let's try for 98% test accuracy
# Start by grid searching for a good alpha and dropout value
alpha = [0.1, 0.01, 0.001, 0.0001]
dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
scores = np.zeros((len(alpha) * len(dropout), 3))
i = 0

for a in alpha:
    for d in dropout:
        NN, _ = hp.NeuralNetwork2(X_train1, y_train1, X_cv, y_cv, nodes=[1200, 1200], 
                                activation=["relu", "relu"], batch=256, epochs=3, dropout=[d, d], alpha=a)
        scores[i,:] = [a, d, NN.evaluate(X_cv, y_cv)[1]]
        i += 1

hp.CrossHeat(alpha, dropout, "Alpha", "Dropout", scores[:, 2])

# Dropout didn't really make any difference. Possible that this would matter more with more iterations
# Best alpha is clearly 0.001, we'll take dropout of 0.4 because performance is good (2nd best), and it's high
# So that could prevent overfitting down the line.
# Now let's try and find a batch length that converges quickly
batch = [32, 64, 128, 256, 512, 1024]
hist = []
for b in batch:
    NN, h = hp.NeuralNetwork2(X_train0, y_train0, X_test, y_test, nodes=[1200, 1200], alpha=0.001,
                            activation=["relu", "relu"], batch=b, epochs=5, dropout=[0.4, 0.4], vs=1/7)
    hist.append(h)

hp.PlotHistory(hist, "acc", legend=batch)

# Batch of 512 is fast and improves quickly
# This gives us a good model to work with, try 10 epochs, work on all data
NN3, hist3 = hp.NeuralNetwork2(X_train0, y_train0, X_test, y_test, nodes=[1200, 1200], 
                        activation=["relu", "relu"], batch=512, epochs=10, alpha=0.001,
                        dropout=0.4)
print(NN3.evaluate(X_test, y_test))
hp.PlotHistory([hist3], "acc")
hp.ClassReport(NN3, X_test, y_test, NN=2)

# We get something with 98.44 percent accuracy in just 10 epochs

# Data science challenge, experiment with convolution
# Using 2 hidden layers (32, 64), with a 5x5 kernel and 2x2 pooling
# This model is a bit more complex, so batch size has been reduced, alpha is unchanged
X_train2 = X_train0[:, :, :, np.newaxis]
X_test2 = X_test[:, :, :, np.newaxis]

NN4, _ = hp.NeuralNetwork2(X_train2, y_train0, X_test2, y_test, nodes=[32, 64], alpha=0.001,
                        activation=["relu", "relu"], batch=128, epochs=5, conv=[True, True], batch_normalisation=True)

hp.ClassReport(NN4, X_test2, y_test, NN=2)

#Data Science Challenge - team solution
import pandas as pd

# Added a dropout layer to the second dense output
x_train = X_train0.reshape(X_train0.shape[0], X_train0.shape[1], X_train0.shape[2], 1)
x_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
y_train_h = pd.get_dummies(y_train0)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(15, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(20, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(120, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(84, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer=tf.train.AdamOptimizer(), 
          loss='categorical_crossentropy',
          metrics=['accuracy'])
model.fit(x_train, y_train_h,
          batch_size=128,
          epochs=10,
          verbose=1,
          shuffle=True)
hp.ClassReport(model, X_test2, y_test, NN=2)