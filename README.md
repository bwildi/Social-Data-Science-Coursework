# Social-Data-Science-Coursework
Code for various projects undertaken during my Social Data Science MSc at Oxford University.

## Python for Social Data Science
This project involved scraping charity information from https://www.givewell.org/ and then finding information about these charity's Wikipedia pages using the API.

## Data Analytics at Scale
For this project, I created an implementation of the Gale-Shapley algorithm to find a stable match for this Kaggle data set: https://www.kaggle.com/c/santa-gift-matching. The first attempt is in matcher.py, and then the other three files reimplement the algorithm to run more efficiently using NumPy, Numba and Numba multiprocessing respectively.

## Machine Learning
For this project, we were tasked with conducting a variety of tasks to classify the MNIST data set. The helpers.py file contains relevant functions and classes. The main summative file is where I conduct a PCA of the MNIST data set, use SVM with a linear and and RBF kernel to classify the data, and build a 100 node single hidden layer neural network from scratch using Numpy. The tensorflow file is where I use Keras to build more complex neural networks to better classify the data.
